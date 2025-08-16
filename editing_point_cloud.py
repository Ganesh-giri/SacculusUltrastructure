# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 20:43:47 2024

@author: Ganesh
"""

import pims
import cv2
import numpy as np
import open3d as o3d
import pyvista as pv
from matplotlib import pyplot as plt

def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    return pcd_down

# Load the image sequence
frames = pims.ImageSequence('H:/test_unet/predicted_mask2/*.png')

# Initialize a list to store contour points
all_points = []
cnts=[]
for i in range(0,len(frames)):
    img = frames[i]
    img = cv2.resize(img, (2800, 3500), interpolation=cv2.INTER_LINEAR)
    edged = cv2.Canny(img, 0, 1)
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts.append(contours)
    # Extract points from contours and add a z-coordinate based on the frame index
    for contour in contours:
        for point in contour:
            all_points.append([point[0][0], point[0][1], i * 1.0])  # z-coordinate increases with frame index

i=np.random.randint(0,len(frames))
img_copy= np.uint8(np.ones(np.shape(frames[i])))#frames[i].copy()
img_copy = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2BGR)
cv2.drawContours(img_copy, cnts[i],-1,color=(255, 0, 0), thickness=1, lineType=cv2.LINE_AA)
cv2.namedWindow('custom window', cv2.WINDOW_KEEPRATIO)
cv2.imshow('custom window',img_copy)
cv2.resizeWindow('custom window', 1000, 1000)
cv2.waitKey(0)
cv2.destroyAllWindows()



##############################

# Convert list of points to numpy array
all_points = np.array(all_points, dtype=np.float32)


original_shape = (3500, 2800)
resized_shape = (3500,2800)

# Calculate the scaling factors due to resizing
scale_x = original_shape[0] / resized_shape[0]
scale_y = original_shape[1] / resized_shape[1]

# Physical resolution in nm
resolution_xy = 10  # 10 nm in XY plane
resolution_z = 30   # 30 nm in Z plane

# Scale the point cloud
scaled_points = np.zeros_like(all_points)
scaled_points[:, 0] = all_points[:, 0] * scale_x * resolution_xy  # Scale X coordinates
scaled_points[:, 1] = all_points[:, 1] * scale_y * resolution_xy  # Scale Y coordinates
scaled_points[:, 2] = all_points[:, 2] * resolution_z             # Scale Z coordinates

# Create Open3D point cloud object
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(scaled_points)
#pcd.points = o3d.utility.Vector3dVector(all_points)
o3d.visualization.draw_geometries([pcd])






o3d.visualization.draw_geometries_with_editing([pcd])



pcd_cropped =o3d.io.read_point_cloud('H:/test_unet/isolated_sensilla_point_clouds/chamber1/cropped_1.ply')
o3d.visualization.draw_geometries([pcd_cropped])


###################################################
##########################################################
##################################################################

import numpy as np
import open3d as o3d
from sklearn.decomposition import PCA
import tempfile
import io

def transform_cone_point_cloud(points):
    # Step 1: Center the point cloud at the origin
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid

    # Step 2: Compute the principal axes using PCA
    pca = PCA(n_components=3)
    pca.fit(centered_points)
    principal_axes = pca.components_

    # Principal axis corresponding to the cone's main axis should align with the z-axis
    # To achieve this, we need a rotation matrix that aligns the first principal axis with the z-axis

    # Compute the rotation matrix
    z_axis = np.array([0, 0, 1])
    rotation_matrix = np.eye(3)

    # Align the principal axis with the z-axis
    axis_to_align = principal_axes[0]
    angle = np.arccos(np.dot(axis_to_align, z_axis) / (np.linalg.norm(axis_to_align) * np.linalg.norm(z_axis)))
    rotation_axis = np.cross(axis_to_align, z_axis)
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

    # Create the rotation matrix using Rodrigues' rotation formula
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    rotation_matrix = np.array([
        [cos_angle + rotation_axis[0]**2 * (1 - cos_angle),
         rotation_axis[0] * rotation_axis[1] * (1 - cos_angle) - rotation_axis[2] * sin_angle,
         rotation_axis[0] * rotation_axis[2] * (1 - cos_angle) + rotation_axis[1] * sin_angle],
        [rotation_axis[1] * rotation_axis[0] * (1 - cos_angle) + rotation_axis[2] * sin_angle,
         cos_angle + rotation_axis[1]**2 * (1 - cos_angle),
         rotation_axis[1] * rotation_axis[2] * (1 - cos_angle) - rotation_axis[0] * sin_angle],
        [rotation_axis[2] * rotation_axis[0] * (1 - cos_angle) - rotation_axis[1] * sin_angle,
         rotation_axis[2] * rotation_axis[1] * (1 - cos_angle) + rotation_axis[0] * sin_angle,
         cos_angle + rotation_axis[2]**2 * (1 - cos_angle)]
    ])

    # Step 3: Apply rotation
    rotated_points = np.dot(centered_points, rotation_matrix.T)

    # Step 4: Translate so that the base of the cone is in the xy-plane
    min_z = np.min(rotated_points[:, 2])
    translated_points = rotated_points - np.array([0, 0, min_z])

    return translated_points




points =np.asarray(pcd_cropped.points)  # Replace with point cloud data
    
transformed_points = transform_cone_point_cloud(points)
    
   



pcd_transformed = o3d.geometry.PointCloud()
pcd_transformed.points = o3d.utility.Vector3dVector(transformed_points)
o3d.visualization.draw_geometries([pcd_transformed])




import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull

def measure_cone_width_at_z_slice(points, z_slice, slice_width=0.1, jitter_strength=1e-5):
    """
    Measure the width of the cone at a given z-slice.
    
    Parameters:
    - points: NumPy array of shape (N, 3), where N is the number of points
    - z_slice: The z-coordinate of the slice where width is measured
    - slice_width: Width of the z-slice range (default is 0.1 units)
    - jitter_strength: Small value to add jitter to avoid collinearity issues (default is 1e-5)
    
    Returns:
    - width: The width of the cone at the given z-slice
    """
    # Filter points at the desired z-slice
    z_min = z_slice - slice_width / 2
    z_max = z_slice + slice_width / 2
    slice_points = points[(points[:, 2] >= z_min) & (points[:, 2] <= z_max)]
    
    if len(slice_points) < 3:
        # Less than 3 points cannot form a convex hull
        return None
    
    # Add jitter to avoid precision errors
    jitter = np.random.normal(scale=jitter_strength, size=slice_points[:, :2].shape)
    slice_points[:, :2] += jitter
    
    try:
        # Compute the convex hull in the 2D plane (x, y)
        hull = ConvexHull(slice_points[:, :2])  # Use only x and y for 2D convex hull
        hull_points = slice_points[hull.vertices]
        
        # Calculate the width by finding the maximum distance between hull points
        width = np.max([np.linalg.norm(p1 - p2) for i, p1 in enumerate(hull_points) for p2 in hull_points[i+1:]])
        
    except Exception as e:
        print(f"Error computing convex hull at z={z_slice}: {e}")
        return None
    
    return width

def measure_cone_width_along_z(points, slice_width=0.1):
    """
    Measure the width of the cone at every z-slice along its length.
    
    Parameters:
    - points: NumPy array of shape (N, 3), where N is the number of points
    - slice_width: Width of each z-slice (default is 0.1 units)
    
    Returns:
    - widths: List of tuples containing z-value and corresponding width
    """
    z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])
    z_slices = np.arange(z_min, z_max, slice_width)
    
    widths = []
    for z in z_slices:
        width = measure_cone_width_at_z_slice(points, z, slice_width)
        if width is not None:
            widths.append((z, width))
    
    return widths


points = np.asarray(pcd_transformed.points)
    
    # Measure widths along the cone
widths = measure_cone_width_along_z(points, slice_width=1)
    
# Print results
for z, width in widths:
    print(f"Width of the cone at z={z:.2f}: {width:.2f}")


import matplotlib.pyplot as plt

# Unpack z-values and widths for plotting
z_values, width_values = zip(*widths)

# Plot the width along z
plt.scatter(z_values, width_values, marker='o')
plt.xlabel('Z Value')
plt.ylabel('Width')
plt.title('Cone Width along Z-axis')
plt.show()


import pyvista as pv

target_mesh = pv.PolyData(points)
# Plot the mesh
mesh = target_mesh.delaunay_3d(alpha=300)


# Plot the mesh
plotter = pv.Plotter()
plotter.add_mesh(mesh, color='white', edge_color='black', show_edges=True, opacity=0.5)
plotter.show_grid()
plotter.show()


num_slices = 500  # adjust this based on your resolution needs
z_min, z_max = mesh.bounds[4], mesh.bounds[5]
z_values = np.linspace(z_min, z_max, num_slices)

widths = []

for z in z_values:
    # Create a plane at each z value
    plane = pv.Plane(center=(0, 0, z), direction=(0, 0, 1), i_size=1000, j_size=1000)
    # Slice the mesh with the plane
    slice_ = mesh.slice(normal=(0, 0, 1), origin=(0, 0, z))
    
    if slice_.n_points > 0:  # Ensure the slice has points
        # Get the x and y coordinates of the points in the slice
        points = slice_.points
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        
        # Calculate the width as the distance between the furthest points
        max_width = max(np.ptp(x_coords), np.ptp(y_coords))
        widths.append(max_width)
    else:
        # If no points, append 0 as the width
        widths.append(0)

# Plot the widths as a function of the z-axis
import matplotlib.pyplot as plt

plt.figure()
plt.plot(z_values, widths, label="Width along z-axis")
plt.xlabel("Z-axis position")
plt.ylabel("Width")
plt.legend()
plt.show()


###############################
################################

import numpy as np
import open3d as o3d

def remove_cropped_points(pcd, pcd_cropped, distance_threshold=0.01):
    """
    Remove points from `pcd` that are close to any point in `pcd_cropped`.
    
    Parameters:
    - pcd: open3d.geometry.PointCloud, original point cloud
    - pcd_cropped: open3d.geometry.PointCloud, cropped point cloud
    - distance_threshold: float, distance threshold to consider a point as close
    
    Returns:
    - pcd_diff: open3d.geometry.PointCloud, the difference point cloud
    """
    # Create KDTree for the cropped point cloud
    pcd_tree = o3d.geometry.KDTreeFlann(pcd_cropped)
    
    # Array to hold points that are not in the cropped point cloud
    diff_points = []
    
    for point in pcd.points:
        [k, idx, _] = pcd_tree.search_radius_vector_3d(point, distance_threshold)
        if k == 0:
            diff_points.append(point)
    
    # Create a new point cloud for the difference
    pcd_diff = o3d.geometry.PointCloud()
    pcd_diff.points = o3d.utility.Vector3dVector(np.array(diff_points))
    
    return pcd_diff


# Get the difference point cloud
pcd_diff = remove_cropped_points(pcd, pcd_cropped, distance_threshold=0.01)

# Visualize the difference point cloud
o3d.visualization.draw_geometries([pcd_diff])

########################################################



import numpy as np
import open3d as o3d
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

def transform_cone_point_cloud(points):
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    pca = PCA(n_components=3)
    pca.fit(centered_points)
    principal_axes = pca.components_
    z_axis = np.array([0, 0, 1])
    axis_to_align = principal_axes[2]
    angle = np.arccos(np.dot(axis_to_align, z_axis) / (np.linalg.norm(axis_to_align) * np.linalg.norm(z_axis)))
    rotation_axis = np.cross(axis_to_align, z_axis)
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    rotation_matrix = np.array([
        [cos_angle + rotation_axis[0]**2 * (1 - cos_angle),
         rotation_axis[0] * rotation_axis[1] * (1 - cos_angle) - rotation_axis[2] * sin_angle,
         rotation_axis[0] * rotation_axis[2] * (1 - cos_angle) + rotation_axis[1] * sin_angle],
        [rotation_axis[1] * rotation_axis[0] * (1 - cos_angle) + rotation_axis[2] * sin_angle,
         cos_angle + rotation_axis[1]**2 * (1 - cos_angle),
         rotation_axis[1] * rotation_axis[2] * (1 - cos_angle) - rotation_axis[0] * sin_angle],
        [rotation_axis[2] * rotation_axis[0] * (1 - cos_angle) - rotation_axis[1] * sin_angle,
         rotation_axis[2] * rotation_axis[1] * (1 - cos_angle) + rotation_axis[0] * sin_angle,
         cos_angle + rotation_axis[2]**2 * (1 - cos_angle)]
    ])
    rotated_points = np.dot(centered_points, rotation_matrix.T)
    min_z = np.min(rotated_points[:, 2])
    translated_points = rotated_points - np.array([0, 0, min_z])
    return translated_points

def measure_cone_width_at_z_slice(points, z_slice, slice_width=0.1, jitter_strength=1e-5):
    z_min = z_slice - slice_width / 2
    z_max = z_slice + slice_width / 2
    slice_points = points[(points[:, 2] >= z_min) & (points[:, 2] <= z_max)]
    if len(slice_points) < 3:
        return None
    jitter = np.random.normal(scale=jitter_strength, size=slice_points[:, :2].shape)
    slice_points[:, :2] += jitter
    try:
        hull = ConvexHull(slice_points[:, :2])
        hull_points = slice_points[hull.vertices]
        width = np.max([np.linalg.norm(p1 - p2) for i, p1 in enumerate(hull_points) for p2 in hull_points[i+1:]])
    except Exception as e:
        print(f"Error computing convex hull at z={z_slice}: {e}")
        return None
    return width

def measure_cone_width_along_z(points, slice_width=0.1):
    z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])
    z_slices = np.arange(z_min, z_max, slice_width)
    widths = []
    for z in z_slices:
        width = measure_cone_width_at_z_slice(points, z, slice_width)
        if width is not None:
            widths.append((z, width))
    return widths

def remove_cropped_points(pcd, pcd_cropped, distance_threshold=0.01):
    pcd_tree = o3d.geometry.KDTreeFlann(pcd_cropped)
    diff_points = []
    for point in pcd.points:
        [k, idx, _] = pcd_tree.search_radius_vector_3d(point, distance_threshold)
        if k == 0:
            diff_points.append(point)
    pcd_diff = o3d.geometry.PointCloud()
    pcd_diff.points = o3d.utility.Vector3dVector(np.array(diff_points))
    return pcd_diff

# Load initial point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(scaled_points)
iterations = 5  # Set the number of iterations

for i in range(iterations):
    # Visualize and edit the point cloud to crop a region
    o3d.visualization.draw_geometries_with_editing([pcd])

    # Load the cropped point cloud from the file saved after editing
    pcd_cropped = o3d.io.read_point_cloud(f'H:/test_unet/isolated_sensilla_point_clouds/chamber1/cropped_{i + 1}.ply')
    
    # Transform the cropped point cloud
    points = np.asarray(pcd_cropped.points)
    transformed_points = transform_cone_point_cloud(points)
    pcd_transformed = o3d.geometry.PointCloud()
    pcd_transformed.points = o3d.utility.Vector3dVector(transformed_points)
    o3d.visualization.draw_geometries([pcd_transformed])

    # Measure widths along the cone
    widths = measure_cone_width_along_z(transformed_points, slice_width=1)
    z_values, width_values = zip(*widths)

    # Plot the width along z
    plt.scatter(z_values, width_values, marker='o')
    plt.xlabel('Z Value')
    plt.ylabel('Width')
    plt.title(f'Cone Width along Z-axis (Iteration {i + 1})')
    plt.show()

    # Remove the cropped points from the original point cloud
    pcd = remove_cropped_points(pcd, pcd_cropped, distance_threshold=0.01)

    # Visualize the updated point cloud
    o3d.visualization.draw_geometries([pcd])

print("Process completed.")




###############################################
####################################################
########################################################

pcd_cropped = o3d.io.read_point_cloud(f'H:/test_unet/isolated_sensilla_point_clouds/chamber1/c1/cropped_1.ply')
o3d.visualization.draw_geometries([pcd_cropped])
# Transform the cropped point cloud
points = np.asarray(pcd_cropped.points)



def transform_cone_point_cloud(points, n=0):
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    pca = PCA(n_components=3)
    pca.fit(centered_points)
    principal_axes = pca.components_
    
    z_axis = np.array([0, 0, 1])
    
    # Find the principal axis most aligned with the z-axis
    dot_products = [np.abs(np.dot(axis, z_axis)) for axis in principal_axes]
    most_aligned_axis_idx = np.argmax(dot_products)
    axis_to_align = principal_axes[n]
    
    # Compute the rotation matrix
    angle = np.arccos(np.dot(axis_to_align, z_axis) / (np.linalg.norm(axis_to_align) * np.linalg.norm(z_axis)))
    rotation_axis = np.cross(axis_to_align, z_axis)
    if np.linalg.norm(rotation_axis) < 1e-10:
        rotation_axis = np.array([1, 0, 0])  # Handle the case where the vectors are parallel
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    rotation_matrix = np.array([
        [cos_angle + rotation_axis[0]**2 * (1 - cos_angle),
         rotation_axis[0] * rotation_axis[1] * (1 - cos_angle) - rotation_axis[2] * sin_angle,
         rotation_axis[0] * rotation_axis[2] * (1 - cos_angle) + rotation_axis[1] * sin_angle],
        [rotation_axis[1] * rotation_axis[0] * (1 - cos_angle) + rotation_axis[2] * sin_angle,
         cos_angle + rotation_axis[1]**2 * (1 - cos_angle),
         rotation_axis[1] * rotation_axis[2] * (1 - cos_angle) - rotation_axis[0] * sin_angle],
        [rotation_axis[2] * rotation_axis[0] * (1 - cos_angle) - rotation_axis[1] * sin_angle,
         rotation_axis[2] * rotation_axis[1] * (1 - cos_angle) + rotation_axis[0] * sin_angle,
         cos_angle + rotation_axis[2]**2 * (1 - cos_angle)]
    ])
    
    rotated_points = np.dot(centered_points, rotation_matrix.T)
    
    # Translate so that the base of the cone is in the xy-plane
    min_z = np.min(rotated_points[:, 2])
    translated_points = rotated_points - np.array([0, 0, min_z])
    
    return translated_points


zs3=[]
ws3=[]

import pyvista as pv
import matplotlib.pyplot as plt
#f'H:/test_unet/isolated_sensilla_point_clouds/chamber1/c3/cropped_8.ply'

def calc_width(path, num_slices=500, alpha=200, n=0):
    
    pcd_cropped = o3d.io.read_point_cloud(path)
    o3d.visualization.draw_geometries([pcd_cropped])
    points=np.asarray(pcd_cropped.points)
    tp = transform_cone_point_cloud(points, n) 

    pcd_transformed = o3d.geometry.PointCloud()
    pcd_transformed.points = o3d.utility.Vector3dVector(tp)
    o3d.visualization.draw_geometries([pcd_transformed])
    points2=np.asarray(pcd_transformed.points)


    target_mesh = pv.PolyData(points2)
    # Plot the mesh
    mesh = target_mesh.delaunay_3d(alpha=200)


    # Plot the mesh
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color='white', edge_color='black', show_edges=True, opacity=0.5)
    plotter.show_grid()
    plotter.show()
    

    num_slices = num_slices  #  adjust this based on resolution needs
    z_min, z_max = mesh.bounds[4], mesh.bounds[5]
    z_values = np.linspace(z_min, z_max, num_slices)

    widths = []

    for z in z_values:
        # Create a plane at each z value
        plane = pv.Plane(center=(0, 0, z), direction=(0, 0, 1), i_size=1000, j_size=1000)
        # Slice the mesh with the plane
        slice_ = mesh.slice(normal=(0, 0, 1), origin=(0, 0, z))
        
        if slice_.n_points > 0:  # Ensure the slice has points
            # Get the x and y coordinates of the points in the slice
            points = slice_.points
            x_coords = points[:, 0]
            y_coords = points[:, 1]
            
            # Calculate the width as the distance between the furthest points
            max_width = max(np.ptp(x_coords), np.ptp(y_coords))
            widths.append(max_width)
        else:
            # If no points, append 0 as the width
            widths.append(0)

# Plot the widths as a function of the z-axis


    plt.figure()
    plt.plot(z_values, widths, label="Width along z-axis")
    plt.xlabel("Z-axis position")
    plt.ylabel("Width")
    plt.legend()
    plt.show()
    return(z_values,widths, points2)


path=f'H:/test_unet/isolated_sensilla_point_clouds/chamber1/c1/cropped_1.ply'
ch1_1 = calc_width(path, 200,200,0)

path=f'H:/test_unet/isolated_sensilla_point_clouds/chamber1/c1/cropped_2.ply'
ch1_2 = calc_width(path, 200,200,0)

path=f'H:/test_unet/isolated_sensilla_point_clouds/chamber1/c1/cropped_3.ply'
ch1_3 = calc_width(path, 200,200,0)

path=f'H:/test_unet/isolated_sensilla_point_clouds/chamber1/c1/cropped_4.ply'
ch1_4 = calc_width(path, 200,200,0)

plt.plot(ch1_1[0], ch1_1[1])
plt.plot(ch1_2[0], ch1_2[1])
plt.plot(ch1_3[0], ch1_3[1])
plt.plot(ch1_4[0], ch1_4[1])


path=f'H:/test_unet/isolated_sensilla_point_clouds/chamber1/cropped_1.ply'
ch2_1 = calc_width(path, 200,200,0)

path=f'H:/test_unet/isolated_sensilla_point_clouds/chamber1/cropped_2.ply'
ch2_2 = calc_width(path, 200,200,0)

path=f'H:/test_unet/isolated_sensilla_point_clouds/chamber1/cropped_3.ply'
ch2_3 = calc_width(path, 200,200,2)

path=f'H:/test_unet/isolated_sensilla_point_clouds/chamber1/cropped_4.ply'
ch2_4 = calc_width(path, 200,200,2)

path=f'H:/test_unet/isolated_sensilla_point_clouds/chamber1/cropped_5.ply'
ch2_5 = calc_width(path, 200,200,2)

plt.plot(ch1_1[0], ch1_1[1])
plt.plot(ch1_2[0], ch1_2[1])
plt.plot(ch1_3[0], ch1_3[1])
plt.plot(ch1_4[0], ch1_4[1])





zs3.append(z_values)
ws3.append(widths)

for i in range(0,len(zs3)):
    plt.plot(zs3[i], ws3[i] )
plt.xlabel("Z-axis position")
plt.ylabel("Width")   










widths = measure_cone_width_along_z(tp, slice_width=1)
z_values, width_values = zip(*widths)

# Plot the width along z
plt.scatter(z_values, width_values, marker='o')
plt.xlabel('Z Value')
plt.ylabel('Width')
plt.show()
