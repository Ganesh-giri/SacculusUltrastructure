# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 21:11:28 2024

@author: Ganesh
"""

import os
import numpy as np
import open3d as o3d
import pyvista as pv
import matplotlib.pyplot as plt

def weighted_moving_average(data, window_size=20):
    """
    Apply a weighted moving average to the data.

    Parameters:
    - data (numpy array): Input data array.
    - window_size (int): Size of the moving window.

    Returns:
    - numpy array: Weighted moving average of the data.
    """
    weights = np.arange(1, window_size + 1)
    wma = np.convolve(data, weights / weights.sum(), mode='valid')
    return np.concatenate([np.full(window_size - 1, np.nan), wma])

def process_and_plot_point_clouds(directory, num_slices=900, window_size=5):
    """
    Process point cloud files to compute the width and height of individual objects,
    apply weighted moving average, and plot the results.

    Parameters:
    - directory (str): Directory containing point cloud files.
    - num_slices (int): Number of slices to create from each mesh.
    - window_size (int): Window size for weighted moving average.

    Returns:
    - width_ch (list): List of width arrays for each object.
    - height_ch (list): List of height arrays for each object.
    - mean_width_wma (numpy array): Mean width across all objects after WMA.
    - mean_height_wma (numpy array): Mean height across all objects after WMA.
    """
    width_ch1 = []
    height_ch1 = []

    files = os.listdir(directory)

    for i in range(len(files)):
        # Read the point cloud
        file_path = os.path.join(directory, files[i])
        pcd = o3d.io.read_point_cloud(file_path)

        # Convert point cloud to numpy array
        points = np.asarray(pcd.points)
        
        # Create a PyVista mesh from the points
        target_mesh = pv.PolyData(points)
        mesh = target_mesh.delaunay_3d(alpha=300)
        
        # Number of slices
        z_min, z_max = mesh.bounds[4], mesh.bounds[5]
        z_values = np.linspace(z_min, z_max, num_slices)

        widths = []

        for z in z_values:
            # Slice the mesh with a plane at each z value
            slice_ = mesh.slice(normal=(0, 0, 1), origin=(0, 0, z))
            
            if slice_.n_points > 0:  # Ensure the slice has points
                # Get the x and y coordinates of the points in the slice
                slice_points = slice_.points
                x_coords = slice_points[:, 0]
                y_coords = slice_points[:, 1]
                
                # Calculate the width as the distance between the furthest points
                max_width = max(np.ptp(x_coords), np.ptp(y_coords))
                widths.append(max_width)
            else:
                # If no points, append 0 as the width
                widths.append(0)
        
        width_ch1.append(widths)
        height_ch1.append(z_values - z_min)  # Normalize height to start from 0
        print(f'Processed file {i+1}/{len(files)}')

    # Convert lists to numpy arrays for easier manipulation
    width_ch1 = np.array(width_ch1, dtype=object)
    height_ch1 = np.array(height_ch1, dtype=object)

    # Discard values prior to the maximum width index
    width_ch = []
    height_ch = []
    for i in range(len(width_ch1)):
        ind = np.argmax(width_ch1[i])
        width_ch.append(width_ch1[i][ind:])
        height_ch.append(height_ch1[i][ind:])

    for i in range(len(height_ch)):
        height_ch[i] = height_ch[i] - np.amin(height_ch[i])
    
    # Convert width and height from nanometers to micrometers
    width_ch = [np.array(w) / 1000 for w in width_ch]
    height_ch = [np.array(h) / 1000 for h in height_ch]

    # Apply weighted moving average
    width_ch_wma = [weighted_moving_average(np.array(w), window_size) for w in width_ch]
    height_ch_wma = [weighted_moving_average(np.array(h), window_size) for h in height_ch]

    for i in range(len(height_ch_wma)):
        height_ch_wma[i] = height_ch_wma[i] - np.amin(height_ch_wma[i])

    # Find the maximum length among the lists to pad the shorter lists
    max_length = max(len(w) for w in width_ch_wma)

    # Pad the shorter lists with NaN
    width_ch_padded = np.array([np.pad(w, (0, max_length - len(w)), 'constant', constant_values=np.nan) for w in width_ch_wma])
    height_ch_padded = np.array([np.pad(h, (0, max_length - len(h)), 'constant', constant_values=np.nan) for h in height_ch_wma])

    # Replace NaN values with 0
    width_ch_padded = np.nan_to_num(width_ch_padded, nan=0)
    height_ch_padded = np.nan_to_num(height_ch_padded, nan=0)

    # Remove columns that are all zeros
    valid_widths = ~np.all(width_ch_padded == 0.00001, axis=0)
    valid_heights = ~np.all(height_ch_padded == 0.00001, axis=0)

    # Filter out invalid columns
    filtered_width_ch_padded = width_ch_padded[:, valid_widths]
    filtered_height_ch_padded = height_ch_padded[:, valid_heights]

    # Calculate the mean width and height after WMA
    mean_width_wma = np.mean(filtered_width_ch_padded, axis=0)
    mean_height_wma = np.mean(filtered_height_ch_padded, axis=0)

    # Plot individual width vs height data with WMA
    plt.figure(figsize=(12, 8))
    for i in range(len(files)):
        plt.plot(height_ch_wma[i], width_ch_wma[i], alpha=0.5, label=f'File {i+1}')

    # Plot mean width vs mean height with WMA
    plt.plot(mean_height_wma, mean_width_wma, label='Mean Width vs Height (WMA)', color='black', linewidth=2)

    # Customize the plot
    plt.xlabel('Height (Z-axis)')
    plt.ylabel('Width')
    plt.title('Width vs Height of Point Cloud Slices with WMA')
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(True)
    plt.show()

    return width_ch, height_ch, mean_width_wma, mean_height_wma

# Example usage
directory = 'E:/EM_data_results/point_clouds/aligned_point_clouds_sample2/c1'
width_ch1_s2, height_ch1_s2, mean_width_ch1_s2, mean_height_ch1_s2 = process_and_plot_point_clouds(directory)

directory = 'E:/EM_data_results/point_clouds/aligned_point_clouds_sample2/c2'
width_ch2_s2, height_ch2_s2, mean_width_ch2_s2, mean_height_ch2_s2 = process_and_plot_point_clouds(directory)

directory = 'E:/EM_data_results/point_clouds/aligned_point_clouds_sample2/c3'
width_ch3_s2, height_ch3_s2, mean_width_ch3_s2, mean_height_ch3_s2 = process_and_plot_point_clouds(directory)

directory = 'E:/EM_data_results/Compiled_results/aligned/ch3_s2'
width_ch3_s2, height_ch3_s2, mean_width_ch3_s2, mean_height_ch3_s2 = process_and_plot_point_clouds(directory)



###### for sample 1 with corrected scaling 

import numpy as np
import open3d as o3d
import pyvista as pv
import matplotlib.pyplot as plt

def adjust_point_cloud_scaling(points, original_xy_scale=10, corrected_xy_scale=10.8398, z_scale=30):
    """
    Adjusts the scaling of a point cloud based on the corrected XY and Z scaling factors.

    Parameters:
    - points (numpy array): Array of point cloud coordinates (N x 3).
    - original_xy_scale (float): Original scaling factor for XY coordinates.
    - corrected_xy_scale (float): Corrected scaling factor for XY coordinates.
    - z_scale (float): Scaling factor for Z coordinates.

    Returns:
    - scaled_points (numpy array): Adjusted point cloud coordinates.
    """
    # Calculate the adjustment factor for XY scaling
    xy_adjustment_factor = corrected_xy_scale / original_xy_scale
    
    # Create a copy of the points array
    scaled_points = np.copy(points)
    
    # Apply the adjustment to XY coordinates
    scaled_points[:, 0] *= xy_adjustment_factor  # Scale X coordinates
    scaled_points[:, 1] *= xy_adjustment_factor  # Scale Y coordinates
    scaled_points[:, 2] *= z_scale / z_scale     # Z coordinates remain unchanged in this example

    return scaled_points

def process_and_plot_point_clouds(directory, num_slices=500, window_size=20):
    """
    Process point cloud files to compute the width and height of individual objects,
    apply weighted moving average, and plot the results.

    Parameters:
    - directory (str): Directory containing point cloud files.
    - num_slices (int): Number of slices to create from each mesh.
    - window_size (int): Window size for weighted moving average.

    Returns:
    - width_ch (list): List of width arrays for each object.
    - height_ch (list): List of height arrays for each object.
    - mean_width_wma (numpy array): Mean width across all objects after WMA.
    - mean_height_wma (numpy array): Mean height across all objects after WMA.
    """
    width_ch1 = []
    height_ch1 = []

    files = os.listdir(directory)

    for i in range(len(files)):
        # Read the point cloud
        file_path = os.path.join(directory, files[i])
        pcd = o3d.io.read_point_cloud(file_path)

        # Convert point cloud to numpy array
        points = np.asarray(pcd.points)

        # Adjust point cloud scaling
        scaled_points = adjust_point_cloud_scaling(points)

        # Create a PyVista mesh from the scaled points
        target_mesh = pv.PolyData(scaled_points)
        mesh = target_mesh.delaunay_3d(alpha=300)
        
        # Number of slices
        z_min, z_max = mesh.bounds[4], mesh.bounds[5]
        z_values = np.linspace(z_min, z_max, num_slices)

        widths = []

        for z in z_values:
            # Slice the mesh with a plane at each z value
            slice_ = mesh.slice(normal=(0, 0, 1), origin=(0, 0, z))
            
            if slice_.n_points > 0:  # Ensure the slice has points
                # Get the x and y coordinates of the points in the slice
                slice_points = slice_.points
                x_coords = slice_points[:, 0]
                y_coords = slice_points[:, 1]
                
                # Calculate the width as the distance between the furthest points
                max_width = max(np.ptp(x_coords), np.ptp(y_coords))
                widths.append(max_width)
            else:
                # If no points, append 0 as the width
                widths.append(0)
        
        width_ch1.append(widths)
        height_ch1.append(z_values - z_min)  # Normalize height to start from 0
        print(f'Processed file {i+1}/{len(files)}')

    # Convert lists to numpy arrays for easier manipulation
    width_ch1 = np.array(width_ch1, dtype=object)
    height_ch1 = np.array(height_ch1, dtype=object)

    # Discard values prior to the maximum width index
    width_ch = []
    height_ch = []
    for i in range(len(width_ch1)):
        ind = np.argmax(width_ch1[i])
        width_ch.append(width_ch1[i][ind:])
        height_ch.append(height_ch1[i][ind:])

    for i in range(len(height_ch)):
        height_ch[i] = height_ch[i] - np.amin(height_ch[i])
        
    # Convert width and height from nanometers to micrometers
    width_ch = [np.array(w) / 1000 for w in width_ch]
    height_ch = [np.array(h) / 1000 for h in height_ch]

    # Apply weighted moving average
    width_ch_wma = [weighted_moving_average(np.array(w), window_size) for w in width_ch]
    height_ch_wma = [weighted_moving_average(np.array(h), window_size) for h in height_ch]

    for i in range(len(height_ch_wma)):
        height_ch_wma[i] = height_ch_wma[i] - np.amin(height_ch_wma[i])

    # Find the maximum length among the lists to pad the shorter lists
    max_length = max(len(w) for w in width_ch_wma)

    # Pad the shorter lists with NaN
    width_ch_padded = np.array([np.pad(w, (0, max_length - len(w)), 'constant', constant_values=np.nan) for w in width_ch_wma])
    height_ch_padded = np.array([np.pad(h, (0, max_length - len(h)), 'constant', constant_values=np.nan) for h in height_ch_wma])

    # Replace NaN values with 0
    width_ch_padded = np.nan_to_num(width_ch_padded, nan=0)
    height_ch_padded = np.nan_to_num(height_ch_padded, nan=0)

    # Remove columns that are all zeros
    valid_widths = ~np.all(width_ch_padded == 0.00001, axis=0)
    valid_heights = ~np.all(height_ch_padded == 0.00001, axis=0)

    # Filter out invalid columns
    filtered_width_ch_padded = width_ch_padded[:, valid_widths]
    filtered_height_ch_padded = height_ch_padded[:, valid_heights]

    # Calculate the mean width and height after WMA
    mean_width_wma = np.mean(filtered_width_ch_padded, axis=0)
    mean_height_wma = np.mean(filtered_height_ch_padded, axis=0)

    # Plot individual width vs height data with WMA
    plt.figure(figsize=(12, 8))
    for i in range(len(files)):
        plt.plot(height_ch_wma[i], width_ch_wma[i], alpha=0.5, label=f'File {i+1}')

    # Plot mean width vs mean height with WMA
    plt.plot(mean_height_wma, mean_width_wma, label='Mean Width vs Height (WMA)', color='black', linewidth=2)

    # Customize the plot
    plt.xlabel('Height (Z-axis)')
    plt.ylabel('Width')
    plt.title('Width vs Height of Point Cloud Slices with WMA')
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(True)
    plt.show()

    return width_ch, height_ch, mean_width_wma, mean_height_wma

# Example usage
directory = 'E:/EM_data_results/point_clouds/aligned_point_clouds_sample1/c1'
width_ch1_s1, height_ch1_s1, mean_width_ch1_s1, mean_height_ch1_s1 = process_and_plot_point_clouds(directory)

directory = 'E:/EM_data_results/point_clouds/aligned_point_clouds_sample1/c2'
width_ch2_s1, height_ch2_s1, mean_width_ch2_s1, mean_height_ch2_s1 = process_and_plot_point_clouds(directory)

directory = 'E:/EM_data_results/point_clouds/aligned_point_clouds_sample1/c3'
width_ch3_s1, height_ch3_s1, mean_width_ch3_s1, mean_height_ch3_s1 = process_and_plot_point_clouds(directory)

directory = 'E:/EM_data_results/Compiled_results/aligned/ch3_s1'
width_ch3_s1, height_ch3_s1, mean_width_ch3_s1, mean_height_ch3_s1 = process_and_plot_point_clouds(directory)



################ combined plotting
width=7.204
height=5
fig, ax = plt.subplots(3, 3, squeeze=False, figsize=(width, height), dpi=500)
ax1= ax[0,0]

for i in range(0,len(width_ch1_s2)):
    ax1.plot(height_ch1_s2[i], width_ch1_s2[i], color='tab:blue', linewidth=0.5, alpha=0.5)
ax1.plot(mean_height_ch1_s2, mean_width_ch1_s2, color='blue', linewidth=1, alpha=1, label ='High RH')    

for i in range(0,len(width_ch1_s1)):
    ax1.plot(height_ch1_s1[i], width_ch1_s1[i], color='tab:red', linewidth=0.5, alpha=0.5)
ax1.plot(mean_height_ch1_s1, mean_width_ch1_s1, color='red', linewidth=1, alpha=1, label ='Low RH')    

ax1.set_xlabel('Height (Z-axis)', fontsize=7,family='Arial')
ax1.set_ylabel('Width',fontsize=7,family='Arial')
ax1.set_title('Chamber 1', fontsize=8,family='Arial')
ax1.legend(loc='upper right', fontsize='small')
ax1.tick_params(labelsize=7)
#plt.grid(True)
#plt.savefig('E:/EM_data_results/Compiled_results/chamber1_width_comparision.png', dpi=300)
plt.tight_layout() 



ax2 = ax[0,1]

for i in range(0,len(width_ch2_s2)):
    ax2.plot(height_ch2_s2[i], width_ch2_s2[i], color='tab:blue', linewidth=0.5, alpha=0.5)
ax2.plot(mean_height_ch2_s2, mean_width_ch2_s2, color='blue', linewidth=1, alpha=1, label ='High RH')    

for i in range(0,len(width_ch2_s1)):
    ax2.plot(height_ch2_s1[i], width_ch2_s1[i], color='tab:red', linewidth=0.5, alpha=0.5)
ax2.plot(mean_height_ch2_s1, mean_width_ch2_s1, color='red', linewidth=1, alpha=1, label ='Low RH')    

ax2.set_xlabel('Height (Z-axis)', fontsize=7,family='Arial')
ax2.set_ylabel('Width',fontsize=7,family='Arial')
ax2.set_title('Chamber 2', fontsize=8,family='Arial')
ax2.legend(loc='upper right', fontsize='small')
ax2.tick_params(labelsize=7)
#plt.grid(True)
#plt.savefig('E:/EM_data_results/Compiled_results/chamber1_width_comparision.png', dpi=300)
plt.tight_layout() 



ax3 =ax[0,2]
for i in range(0,len(width_ch3_s2)):
    ax3.plot(height_ch3_s2[i], width_ch3_s2[i], color='tab:blue', linewidth=0.5, alpha=0.5)
ax3.plot(mean_height_ch3_s2, mean_width_ch3_s2, color='blue', linewidth=1, alpha=1, label ='High RH')    

for i in range(0,len(width_ch3_s1)):
    ax3.plot(height_ch3_s1[i], width_ch3_s1[i], color='tab:red', linewidth=0.5, alpha=0.5)
ax3.plot(mean_height_ch3_s1, mean_width_ch3_s1, color='red', linewidth=1, alpha=1, label ='Low RH')    

ax3.set_xlabel('Height (Z-axis)', fontsize=7,family='Arial')
ax3.set_ylabel('Width',fontsize=7,family='Arial')
ax3.set_title('Chamber 3', fontsize=8,family='Arial')
ax3.legend(loc='upper right', fontsize='small')
ax3.tick_params(labelsize=7)
#plt.grid(True)
#plt.savefig('E:/EM_data_results/Compiled_results/chamber1_width_comparision.png', dpi=300)
plt.tight_layout() 
plt.show()


#### statistical analysis ######

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf



# Ensure data arrays have the same lengths
width_dry_flat = np.concatenate(width_ch3_s1)
height_dry_flat = np.concatenate(height_ch3_s1)
width_humid_flat = np.concatenate(width_ch3_s2)
height_humid_flat = np.concatenate(height_ch3_s2)




print(f'Length of width_dry_flat: {len(width_dry_flat)}')
print(f'Length of height_dry_flat: {len(height_dry_flat)}')
print(f'Length of width_humid_flat: {len(width_humid_flat)}')
print(f'Length of height_humid_flat: {len(height_humid_flat)}')

# Make sure that the lengths of the arrays are equal
if len(width_dry_flat) != len(height_dry_flat) or len(width_humid_flat) != len(height_humid_flat):
    raise ValueError("The lengths of width and height arrays must match within each group.")

# Combine the data into a single DataFrame
width_combined = np.concatenate([width_dry_flat, width_humid_flat])
height_combined = np.concatenate([height_dry_flat, height_humid_flat])
groups_combined = ['Dry'] * len(width_dry_flat) + ['Humid'] * len(width_humid_flat)

# Create an appropriate 'Object' column
object_dry = np.repeat(range(len(width_ch3_s1)), [len(arr) for arr in width_ch3_s1])
object_humid = np.repeat(range(len(width_ch3_s2)), [len(arr) for arr in width_ch3_s2])
object_combined = np.concatenate([object_dry, object_humid])

# Combine into a DataFrame
long_data = pd.DataFrame({
    'Width': width_combined,
    'Height': height_combined,
    'Group': groups_combined,
    'Object': object_combined
}).dropna()

# Ensure that 'Group' is treated as a categorical variable
long_data['Group'] = long_data['Group'].astype('category')

# Convert 'Width' and 'Height' to numeric types
long_data['Width'] = pd.to_numeric(long_data['Width'])
long_data['Height'] = pd.to_numeric(long_data['Height'])

# Check the data types to ensure they are correct
print(long_data.dtypes)

# Fit a mixed-effects model
model = smf.mixedlm("Width ~ Height * Group", long_data, groups=long_data["Object"])
result = model.fit()
print(result.summary())



from scipy.stats import t

# Generate a range of heights
heights = np.linspace(min(long_data['Height']), max(long_data['Height']), 1000)

# Create a DataFrame for prediction
prediction_data_dry = pd.DataFrame({'Height': heights, 'Group': 'Dry'})
prediction_data_humid = pd.DataFrame({'Height': heights, 'Group': 'Humid'})

# Get predictions for Dry group
predicted_dry = result.predict(prediction_data_dry)
predicted_dry = [i for i in predicted_dry if i>0]
predicted_humid = result.predict(prediction_data_humid)
predicted_humid = [i for i in predicted_humid if i>0]


# Calculate standard errors for the predictions
cov_params = result.cov_params()
se_dry = np.sqrt(cov_params.loc['Intercept', 'Intercept']
                 + heights**2 * cov_params.loc['Height', 'Height']
                 + 2 * heights * cov_params.loc['Intercept', 'Height'])
se_humid = np.sqrt(cov_params.loc['Intercept', 'Intercept']
                   + cov_params.loc['Group[T.Humid]', 'Group[T.Humid]']
                   + heights**2 * (cov_params.loc['Height', 'Height'] + cov_params.loc['Height:Group[T.Humid]', 'Height:Group[T.Humid]'])
                   + 2 * (heights * cov_params.loc['Intercept', 'Height']
                          + heights * cov_params.loc['Intercept', 'Height:Group[T.Humid]']
                          + cov_params.loc['Intercept', 'Group[T.Humid]']
                          + heights**2 * cov_params.loc['Height', 'Height:Group[T.Humid]']))

# Calculate the confidence intervals
alpha = 0.05
t_value = t.ppf(1 - alpha/2, df=result.df_resid)
ci_dry_lower = predicted_dry - t_value * se_dry[:len(predicted_dry)]
ci_dry_upper = predicted_dry + t_value * se_dry[:len(predicted_dry)]
ci_humid_lower = predicted_humid - t_value * se_humid[:len(predicted_humid)]
ci_humid_upper = predicted_humid + t_value * se_humid[:len(predicted_humid)]

# Plotting the results with confidence intervals
plt.figure(figsize=(10, 6))
plt.plot(heights[:len(predicted_dry)], predicted_dry, label='Dry', color='blue')
plt.fill_between(heights[:len(predicted_dry)], ci_dry_lower, ci_dry_upper, color='blue', alpha=0.2)
plt.plot(heights[:len(predicted_humid)], predicted_humid, label='Humid', color='green')
plt.fill_between(heights[:len(predicted_humid)], ci_humid_lower, ci_humid_upper, color='green', alpha=0.2)
plt.xlabel('Height')
plt.ylabel('Predicted Width')
plt.title('Predicted Width vs Heigh Chamber3')
plt.legend()
plt.grid(True)
plt.savefig('E:/EM_data_results/Compiled_results/pridicted_width_vs_length_chamber3.png', dpi=300)
plt.show()


##### test 


# Calculate fwhm
def calculate_fwhm(height_list, width_list):
    fwhm_comb =[]
    for i in range(0,len(height_list)):
        half_h = np.amax(height_list[i])/2
        half_h = np.where(height_list[i] < half_h)
        half_index = half_h[0][-1]
        fwhm = width_list[i][half_index]
        fwhm_comb.append(fwhm)
    return(fwhm_comb)
    


fwhm_ch1_s1 = calculate_fwhm(height_ch1_s1, width_ch1_s1)
fwhm_ch2_s1 = calculate_fwhm(height_ch2_s1, width_ch2_s1)
fwhm_ch3_s1 = calculate_fwhm(height_ch3_s1, width_ch3_s1)

fwhm_ch1_s2 = calculate_fwhm(height_ch1_s2, width_ch1_s2)
fwhm_ch2_s2 = calculate_fwhm(height_ch2_s2, width_ch2_s2)
fwhm_ch3_s2 = calculate_fwhm(height_ch3_s2, width_ch3_s2)


import matplotlib.pyplot as plt



# Create a box plot
plt.figure(figsize=(10, 6))
a1 = fwhm_ch3_s1
a2 = fwhm_ch3_s2
plt.boxplot([a1, a2], labels=['FWHM ch3 s1', 'FWHM ch3 s2'])
plt.ylabel('FWHM')
plt.title('Comparison of FWHM for ch3 s1 and ch3 s2')
plt.grid(True)
plt.savefig('E:/EM_data_results/Compiled_results/fwhm_ch1.png', dpi=300)
plt.show()

from scipy.stats import ttest_ind, mannwhitneyu

# Perform Shapiro-Wilk test for normality
from scipy.stats import shapiro

shapiro_ch3_s1 = shapiro(a1)
shapiro_ch3_s2 = shapiro(a2)

print(f'Shapiro-Wilk Test for FWHM ch3 s1: W={shapiro_ch3_s1.statistic}, p-value={shapiro_ch3_s1.pvalue}')
print(f'Shapiro-Wilk Test for FWHM ch3 s2: W={shapiro_ch3_s2.statistic}, p-value={shapiro_ch3_s2.pvalue}')

# Based on the p-value from the Shapiro-Wilk test, decide which test to use
if shapiro_ch3_s1.pvalue > 0.05 and shapiro_ch3_s2.pvalue > 0.05:
    # Use t-test if both datasets are normally distributed
    t_stat, t_pvalue = ttest_ind(a1, a2)
    print(f'T-test: t-statistic={t_stat}, p-value={t_pvalue}')
else:
    # Use Mann-Whitney U test if at least one dataset is not normally distributed
    u_stat, u_pvalue = mannwhitneyu(a1, a2)
    print(f'Mann-Whitney U test: U-statistic={u_stat}, p-value={u_pvalue}')



import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, shapiro, wilcoxon



# Descriptive Statistics
median_s1 = np.median(fwhm_ch1_s1)
iqr_s1 = np.percentile(fwhm_ch1_s1, 75) - np.percentile(fwhm_ch1_s1, 25)
median_s2 = np.median(fwhm_ch1_s2)
iqr_s2 = np.percentile(fwhm_ch1_s2, 75) - np.percentile(fwhm_ch1_s2, 25)

print(f'Median FWHM ch1 s1: {median_s1}, IQR: {iqr_s1}')
print(f'Median FWHM ch1 s2: {median_s2}, IQR: {iqr_s2}')

# Mann-Whitney U test (non-parametric)
u_stat, u_pvalue = mannwhitneyu(fwhm_ch1_s1, fwhm_ch1_s2, alternative='two-sided')
print(f'Mann-Whitney U test: U-statistic={u_stat}, p-value={u_pvalue}')

# Wilcoxon Signed-Rank Test (if samples are paired)
if len(fwhm_ch1_s1) == len(fwhm_ch1_s2):
    w_stat, w_pvalue = wilcoxon(fwhm_ch1_s1, fwhm_ch1_s2)
    print(f'Wilcoxon Signed-Rank Test: W-statistic={w_stat}, p-value={w_pvalue}')

# Box Plot
plt.figure(figsize=(10, 6))
plt.boxplot([fwhm_ch1_s1, fwhm_ch1_s2], labels=['FWHM ch1 s1', 'FWHM ch1 s2'])
plt.ylabel('FWHM')


### plot xy data points of slices 
pcd = o3d.io.read_point_cloud('E:/EM_data_results/point_clouds/aligned_point_clouds_sample1/c1/transformed_1.ply')

# Convert point cloud to numpy array
points = np.asarray(pcd.points)

# Create a PyVista mesh from the points
target_mesh = pv.PolyData(points)
mesh = target_mesh.delaunay_3d(alpha=300)

plotter = pv.Plotter()
plotter.add_mesh(mesh, color='white', edge_color='black', show_edges=True, opacity=0.5)
plotter.show_grid()
plotter.show()

z_min, z_max = mesh.bounds[4], mesh.bounds[5]
z_values = np.linspace(z_min, z_max, 500)
slice_ = mesh.slice(normal=(0, 0, 1), origin=(0, 0, z_values[350]))

slice_points = slice_.points  # Extract points
slice_points_array = np.asarray(slice_points)  # Convert to numpy array

# Plot only the xy coordinates in 2D using matplotlib
plt.figure()
plt.scatter(slice_points_array[:, 0], slice_points_array[:, 1], c='b', marker='o', s=1)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('2D Slice Points (XY Coordinates)')
plt.show()









################
#################
###################
##########################
#################################



import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


def gmm(width_1, height_1, width_2, height_2):
    
    # Ensure data arrays have the same lengths
    width_dry_flat = np.concatenate(width_1)
    height_dry_flat = np.concatenate(height_1)
    width_humid_flat = np.concatenate(width_2)
    height_humid_flat = np.concatenate(height_2)




    print(f'Length of width_dry_flat: {len(width_dry_flat)}')
    print(f'Length of height_dry_flat: {len(height_dry_flat)}')
    print(f'Length of width_humid_flat: {len(width_humid_flat)}')
    print(f'Length of height_humid_flat: {len(height_humid_flat)}')

    # Make sure that the lengths of the arrays are equal
    if len(width_dry_flat) != len(height_dry_flat) or len(width_humid_flat) != len(height_humid_flat):
        raise ValueError("The lengths of width and height arrays must match within each group.")
    
    # Combine the data into a single DataFrame
    width_combined = np.concatenate([width_dry_flat, width_humid_flat])
    height_combined = np.concatenate([height_dry_flat, height_humid_flat])
    groups_combined = ['Dry'] * len(width_dry_flat) + ['Humid'] * len(width_humid_flat)
    
    # Create an appropriate 'Object' column
    object_dry = np.repeat(range(len(width_1)), [len(arr) for arr in width_1])
    object_humid = np.repeat(range(len(width_2)), [len(arr) for arr in width_2])
    object_combined = np.concatenate([object_dry, object_humid])
    
    # Combine into a DataFrame
    long_data = pd.DataFrame({
        'Width': width_combined,
        'Height': height_combined,
        'Group': groups_combined,
        'Object': object_combined
    }).dropna()
    
    # Ensure that 'Group' is treated as a categorical variable
    long_data['Group'] = long_data['Group'].astype('category')
    
    # Convert 'Width' and 'Height' to numeric types
    long_data['Width'] = pd.to_numeric(long_data['Width'])
    long_data['Height'] = pd.to_numeric(long_data['Height'])
    
    # Check the data types to ensure they are correct
    print(long_data.dtypes)
    
    # Fit a mixed-effects model
    model = smf.mixedlm("Width ~ Height * Group", long_data, groups=long_data["Object"])
    result = model.fit()
    print(result.summary())
    
    return(result, long_data)




from scipy.stats import t

def pred_ci ( result, ld):
    

    # Generate a range of heights
    heights = np.linspace(min(ld['Height']), max(ld['Height']), 1000)
    
    # Create a DataFrame for prediction
    prediction_data_dry = pd.DataFrame({'Height': heights, 'Group': 'Dry'})
    prediction_data_humid = pd.DataFrame({'Height': heights, 'Group': 'Humid'})
    
    # Get predictions for Dry group
    predicted_dry = result.predict(prediction_data_dry)
    predicted_dry = [i for i in predicted_dry if i>0]
    predicted_humid = result.predict(prediction_data_humid)
    predicted_humid = [i for i in predicted_humid if i>0]
    
    
    # Calculate standard errors for the predictions
    cov_params = result.cov_params()
    se_dry = np.sqrt(cov_params.loc['Intercept', 'Intercept']
                     + heights**2 * cov_params.loc['Height', 'Height']
                     + 2 * heights * cov_params.loc['Intercept', 'Height'])
    se_humid = np.sqrt(cov_params.loc['Intercept', 'Intercept']
                       + cov_params.loc['Group[T.Humid]', 'Group[T.Humid]']
                       + heights**2 * (cov_params.loc['Height', 'Height'] + cov_params.loc['Height:Group[T.Humid]', 'Height:Group[T.Humid]'])
                       + 2 * (heights * cov_params.loc['Intercept', 'Height']
                              + heights * cov_params.loc['Intercept', 'Height:Group[T.Humid]']
                              + cov_params.loc['Intercept', 'Group[T.Humid]']
                              + heights**2 * cov_params.loc['Height', 'Height:Group[T.Humid]']))
    
    # Calculate the confidence intervals
    alpha = 0.05
    t_value = t.ppf(1 - alpha/2, df=result.df_resid)
    ci_dry_lower = predicted_dry - t_value * se_dry[:len(predicted_dry)]
    ci_dry_upper = predicted_dry + t_value * se_dry[:len(predicted_dry)]
    ci_humid_lower = predicted_humid - t_value * se_humid[:len(predicted_humid)]
    ci_humid_upper = predicted_humid + t_value * se_humid[:len(predicted_humid)]
    return(predicted_dry, ci_dry_lower, ci_dry_upper, predicted_humid, ci_humid_lower, ci_humid_upper)


# Calculate fwhm
def calculate_fwhm(height_list, width_list):
    fwhm_comb =[]
    for i in range(0,len(height_list)):
        half_h = np.amax(height_list[i])/2
        half_h = np.where(height_list[i] < half_h)
        half_index = half_h[0][-1]
        fwhm = width_list[i][half_index]
        fwhm_comb.append(fwhm)
    return(fwhm_comb)
    






r_ch1,ld_ch1 =gmm(width_ch1_s1, height_ch1_s1, width_ch1_s2, height_ch1_s2)
predicted_dry_ch1, ci_dry_lower_ch1, ci_dry_upper_ch1, predicted_humid_ch1, ci_humid_lower_ch1, ci_humid_upper_ch1 = pred_ci(r_ch1,ld_ch1)

r_ch2,ld_ch2 = gmm(width_ch2_s1, height_ch2_s1, width_ch2_s2, height_ch2_s2)
predicted_dry_ch2, ci_dry_lower_ch2, ci_dry_upper_ch2, predicted_humid_ch2, ci_humid_lower_ch2, ci_humid_upper_ch2 = pred_ci(r_ch2,ld_ch2)

r_ch3,ld_ch3 =gmm(width_ch3_s1, height_ch3_s1, width_ch3_s2, height_ch3_s2)
predicted_dry_ch3, ci_dry_lower_ch3, ci_dry_upper_ch3, predicted_humid_ch3, ci_humid_lower_ch3, ci_humid_upper_ch3 = pred_ci(r_ch3,ld_ch3)

fwhm_ch1_s1 = calculate_fwhm(height_ch1_s1, width_ch1_s1)
fwhm_ch2_s1 = calculate_fwhm(height_ch2_s1, width_ch2_s1)
fwhm_ch3_s1 = calculate_fwhm(height_ch3_s1, width_ch3_s1)

fwhm_ch1_s2 = calculate_fwhm(height_ch1_s2, width_ch1_s2)
fwhm_ch2_s2 = calculate_fwhm(height_ch2_s2, width_ch2_s2)
fwhm_ch3_s2 = calculate_fwhm(height_ch3_s2, width_ch3_s2)






import pandas as pd
import seaborn as sns

width=7.204
height=5
fig, ax = plt.subplots(3, 3, squeeze=False, figsize=(width, height), dpi=500)
ax1= ax[0,0]

for i in range(0,len(width_ch1_s2)):
    ax1.plot(height_ch1_s2[i], width_ch1_s2[i], color='tab:blue', linewidth=0.5, alpha=0.5)
ax1.plot(mean_height_ch1_s2, mean_width_ch1_s2, color='blue', linewidth=1, alpha=1, label ='High RH')    

for i in range(0,len(width_ch1_s1)):
    ax1.plot(height_ch1_s1[i], width_ch1_s1[i], color='tab:red', linewidth=0.5, alpha=0.5)
ax1.plot(mean_height_ch1_s1, mean_width_ch1_s1, color='red', linewidth=1, alpha=1, label ='Low RH')    

ax1.set_xlabel('Height (Z-axis)', fontsize=7,family='Arial')
ax1.set_ylabel('Width',fontsize=7,family='Arial')
ax1.set_title('Chamber 1', fontsize=8,family='Arial')
ax1.legend(loc='upper right', fontsize='x-small')
ax1.tick_params(labelsize=7)
#plt.grid(True)
#plt.savefig('E:/EM_data_results/Compiled_results/chamber1_width_comparision.png', dpi=300)
plt.tight_layout() 



ax2 = ax[0,1]

for i in range(0,len(width_ch2_s2)):
    ax2.plot(height_ch2_s2[i], width_ch2_s2[i], color='tab:blue', linewidth=0.5, alpha=0.5)
ax2.plot(mean_height_ch2_s2, mean_width_ch2_s2, color='blue', linewidth=1, alpha=1, label ='High RH')    

for i in range(0,len(width_ch2_s1)):
    ax2.plot(height_ch2_s1[i], width_ch2_s1[i], color='tab:red', linewidth=0.5, alpha=0.5)
ax2.plot(mean_height_ch2_s1, mean_width_ch2_s1, color='red', linewidth=1, alpha=1, label ='Low RH')    

ax2.set_xlabel('Height (Z-axis)', fontsize=7,family='Arial')
ax2.set_ylabel('Width',fontsize=7,family='Arial')
ax2.set_title('Chamber 2', fontsize=8,family='Arial')
ax2.legend(loc='upper right', fontsize='x-small')
ax2.tick_params(labelsize=7)
#plt.grid(True)
#plt.savefig('E:/EM_data_results/Compiled_results/chamber1_width_comparision.png', dpi=300)
plt.tight_layout() 



ax3 =ax[0,2]
for i in range(0,len(width_ch3_s2)):
    ax3.plot(height_ch3_s2[i], width_ch3_s2[i], color='tab:blue', linewidth=0.5, alpha=0.5)
ax3.plot(mean_height_ch3_s2, mean_width_ch3_s2, color='blue', linewidth=1, alpha=1, label ='High RH')    

for i in range(0,len(width_ch3_s1)):
    ax3.plot(height_ch3_s1[i], width_ch3_s1[i], color='tab:red', linewidth=0.5, alpha=0.5)
ax3.plot(mean_height_ch3_s1, mean_width_ch3_s1, color='red', linewidth=1, alpha=1, label ='Low RH')    

ax3.set_xlabel('Height (Z-axis)', fontsize=7,family='Arial')
ax3.set_ylabel('Width',fontsize=7,family='Arial')
ax3.set_title('Chamber 3', fontsize=8,family='Arial')
ax3.legend(loc='upper right', fontsize='x-small')
ax3.tick_params(labelsize=7)
#plt.grid(True)
#plt.savefig('E:/EM_data_results/Compiled_results/chamber1_width_comparision.png', dpi=300)


ax4 =ax[1,0]
ax5 =ax[1,1]
ax6 =ax[1,2]


# Boxplot for FWHM - Chamber 1
ax4.boxplot([fwhm_ch1_s1, fwhm_ch1_s2], patch_artist=True, widths=0.2,
            boxprops=dict(facecolor='tab:red', color='tab:red', alpha=0.6),
            medianprops=dict(color='black', alpha=0.6),
            whiskerprops=dict(color='black', alpha=0.6),
            capprops=dict(color='black', alpha=0.6),
            flierprops=dict(markerfacecolor='black', marker='o', markersize=2, linestyle='none', alpha=0.6))

ax4.boxplot([fwhm_ch1_s2], positions=[2], patch_artist=True, widths=0.2,
            boxprops=dict(facecolor='tab:blue', color='tab:blue', alpha=0.6),
            medianprops=dict(color='black', alpha=0.6),
            whiskerprops=dict(color='black', alpha=0.6),
            capprops=dict(color='black', alpha=0.6),
            flierprops=dict(markerfacecolor='black', marker='o', markersize=2, linestyle='none', alpha=0.6))


ax4.set_xlabel('')  # Remove X-axis label
ax4.set_ylabel('FWHM', fontsize=7, family='Arial')  # Reduce Y-axis label font size
ax4.tick_params(labelsize=7)
lim = np.amax(fwhm_ch1_s1+ fwhm_ch1_s2)
lim2= np.amin(fwhm_ch1_s1+ fwhm_ch1_s2)
ax4.set_ylim(lim2-0.1,lim+0.35 )


# Boxplot for FWHM - Chamber 2
ax5.boxplot([fwhm_ch2_s1, fwhm_ch2_s2], patch_artist=True, widths=0.2,
            boxprops=dict(facecolor='tab:red', color='tab:red', alpha=0.6),
            medianprops=dict(color='black', alpha=0.6),
            whiskerprops=dict(color='black', alpha=0.6),
            capprops=dict(color='black', alpha=0.6),
            flierprops=dict(markerfacecolor='black', marker='o', markersize=2, linestyle='none', alpha=0.6))

ax5.boxplot([fwhm_ch2_s2], positions=[2], patch_artist=True, widths=0.2,
            boxprops=dict(facecolor='tab:blue', color='tab:blue', alpha=0.6),
            medianprops=dict(color='black', alpha=0.6),
            whiskerprops=dict(color='black', alpha=0.6),
            capprops=dict(color='black', alpha=0.6),
            flierprops=dict(markerfacecolor='black', marker='o', markersize=2, linestyle='none', alpha=0.6))


ax5.set_xlabel('')  # Remove X-axis label
ax5.set_ylabel('FWHM', fontsize=7, family='Arial')  # Reduce Y-axis label font size
ax5.tick_params(labelsize=7)
lim= np.amax(fwhm_ch2_s1+ fwhm_ch2_s2)
lim2= np.amin(fwhm_ch2_s1+ fwhm_ch2_s2)
ax5.set_ylim(lim2-0.1,lim+0.35 )


# Boxplot for FWHM - Chamber 3
ax6.boxplot([fwhm_ch3_s1, fwhm_ch3_s2], patch_artist=True, widths=0.2,
            boxprops=dict(facecolor='tab:red', color='tab:red', alpha=0.6),
            medianprops=dict(color='black', alpha=0.6),
            whiskerprops=dict(color='black', alpha=0.6),
            capprops=dict(color='black', alpha=0.6),
            flierprops=dict(markerfacecolor='black', marker='o', markersize=2, linestyle='none', alpha=0.6))

ax6.boxplot([fwhm_ch3_s2], positions=[2], patch_artist=True, widths=0.2,
            boxprops=dict(facecolor='tab:blue', color='tab:blue', alpha=0.6),
            medianprops=dict(color='black', alpha=0.6),
            whiskerprops=dict(color='black', alpha=0.6),
            capprops=dict(color='black', alpha=0.6),
            flierprops=dict(markerfacecolor='black', marker='o', markersize=2, linestyle='none', alpha=0.6))


ax6.set_xlabel('')  # Remove X-axis label
ax6.set_ylabel('FWHM', fontsize=7, family='Arial')  # Reduce Y-axis label font size
ax6.tick_params(labelsize=7)
lim= np.amax(fwhm_ch3_s1+ fwhm_ch3_s2)
lim2= np.amin(fwhm_ch3_s1+ fwhm_ch3_s2)
ax6.set_ylim(lim2-0.1,lim+0.35 )


heights = np.linspace(0,6000,1200)/1000
# Plot for Chamber 1 (ax7)
ax7 = ax[2,0]
ax7.plot(heights[:len(predicted_dry_ch1)], predicted_dry_ch1, label='Low RH', color='tab:red')
ax7.fill_between(heights[:len(predicted_dry_ch1)], ci_dry_lower_ch1, ci_dry_upper_ch1, color='tab:red', alpha=0.2)
ax7.plot(heights[:len(predicted_humid_ch1)], predicted_humid_ch1, label='High RH', color='tab:blue')
ax7.fill_between(heights[:len(predicted_humid_ch1)], ci_humid_lower_ch1, ci_humid_upper_ch1, color='tab:blue', alpha=0.2)
ax7.set_xlabel('Height', fontsize=7, family='Arial')
ax7.set_ylabel('Predicted Width', fontsize=7, family='Arial')
ax7.legend(loc='upper right', fontsize='x-small')
ax7.tick_params(labelsize=7)

# Plot for Chamber 2 (ax8)
ax8 = ax[2,1]
ax8.plot(heights[:len(predicted_dry_ch2)], predicted_dry_ch2, label='Low RH', color='tab:red')
ax8.fill_between(heights[:len(predicted_dry_ch2)], ci_dry_lower_ch2, ci_dry_upper_ch2, color='tab:red', alpha=0.2)
ax8.plot(heights[:len(predicted_humid_ch2)], predicted_humid_ch2, label='High', color='tab:blue')
ax8.fill_between(heights[:len(predicted_humid_ch2)], ci_humid_lower_ch2, ci_humid_upper_ch2, color='tab:blue', alpha=0.2)
ax8.set_xlabel('Height', fontsize=7, family='Arial')
ax8.set_ylabel('Predicted Width', fontsize=7, family='Arial')
ax8.legend(loc='upper right', fontsize='x-small')
ax8.tick_params(labelsize=7)

# Plot for Chamber 3 (ax9)
ax9 = ax[2,2]
ax9.plot(heights[:len(predicted_dry_ch3)], predicted_dry_ch3, label='Low RH', color='tab:red')
ax9.fill_between(heights[:len(predicted_dry_ch3)], ci_dry_lower_ch3, ci_dry_upper_ch3, color='tab:red', alpha=0.2)
ax9.plot(heights[:len(predicted_humid_ch3)], predicted_humid_ch3, label='High RH', color='tab:blue')
ax9.fill_between(heights[:len(predicted_humid_ch3)], ci_humid_lower_ch3, ci_humid_upper_ch3, color='tab:blue', alpha=0.2)
ax9.set_xlabel('Height', fontsize=7, family='Arial')
ax9.set_ylabel('Predicted Width', fontsize=7, family='Arial')
ax9.legend(loc='upper right', fontsize='x-small')
ax9.tick_params(labelsize=7)


plt.tight_layout() 
plt.savefig('E:/EM_data_results/Compiled_results/results_fig.png', dpi=600)
plt.show()




########


a1 = fwhm_ch3_s1
a2 = fwhm_ch3_s2
plt.boxplot([a1,a2])
u_stat, p = mannwhitneyu(a1, a2)
print(f'Mann-Whitney U test: U-statistic={u_stat}, p-value={p}')
if p < 0.001:
    sig_symbol = '***'
elif p < 0.01:
    sig_symbol = '**'
elif p < 0.05:
    sig_symbol = '*'
print(sig_symbol)