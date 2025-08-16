import pims
import os
from matplotlib import pyplot as plt
frames=pims.ImageSequence('F:/Drosophilla_Antenna_IV/PluginData/SEM/Detail/LiveTracking/*.tif')

plt.imshow(frames[285])


for i in range (len(frames[285:])):
    img=frames[285+i]
    plt.imsave('F:/Drosophilla_Antenna_IV/PluginData/SEM/chamber2/' + str(i) + '.tiff', img)



frames=pims.ImageSequence('F:/Drosophilla_Antenna_IV/PluginData/SEM/Detail2/LiveTracking/*.tif')

plt.imshow(frames[0])


for i in range (len(frames[:182])):
    img=frames[i]
    plt.imsave('F:/Drosophilla_Antenna_IV/PluginData/SEM/chamber2/' + str(i+197) + '.tiff', img)



import numpy as np
from matplotlib import pyplot as plt
import cv2
import imageio

frames=pims.ImageSequence('F:/Drosophilla_Antenna_IV/PluginData/SEM/chamber2/*.tiff')



from skimage.registration import register_translation
from skimage.registration import phase_cross_correlation

from scipy.ndimage import fourier_shift
from skimage import io

movie = frames
shifts = []
corrected_shift_movie = []

for img in range(0,len(movie)):
    if img < len(movie) - 1:

        shift, error, diffphase = register_translation(movie[0], movie[img + 1])

        img_corr = fourier_shift(np.fft.fftn(movie[img + 1]), shift)
        img_corr = np.fft.ifftn(img_corr)

        shifts.append(shift)
        corrected_shift_movie.append(img_corr.real)


# for plotting the xy shifts over time

shifts = np.array(shifts)
corrected_shift_movie = np.array(corrected_shift_movie)

x_drift = [shifts[i][0] for i in range(0,shifts.shape[0])]
y_drift = [shifts[i][1] for i in range(0,shifts.shape[0])]

plt.plot(x_drift, '* g' , label = ' X drift')
plt.plot(y_drift, '--r' , label = ' Y drfit')
plt.legend()

# checking drift for the new corrected movie

movie = corrected_shift_movie
shifts_corr = []

for img in range(0,movie.shape[0]):
    if img < movie.shape[0] - 1:

        shift, error, diffphase = register_translation(movie[0], movie[img + 1])
        shifts_corr.append(shift)

shifts_corr = np.array(shifts_corr)

x_drift = [shifts_corr[i][0] for i in range(0,shifts_corr.shape[0])]
y_drift = [shifts_corr[i][1] for i in range(0,shifts_corr.shape[0])]

plt.plot(x_drift, '* g' , label = ' X drift')
plt.plot(y_drift, '--r' , label = ' Y drfit')
plt.legend()

# saving the new corrected movie

import tifffile as tiff
movie_to_save = corrected_shift_movie

with tiff.TiffWriter('E:/trial/drift_correction.tif', bigtiff=True) as tif:
        for new_image in range(movie_to_save.shape[0]):
            tif.save(movie_to_save[new_image], compress=0)
            
            
            
#########################


import numpy as np
import matplotlib.pyplot as plt
from skimage.registration import phase_cross_correlation
from scipy.ndimage import fourier_shift
from skimage.transform import rescale
from skimage import io

# Example image paths (replace with actual image paths or arrays)
image_path1 = 'path_to_image1'
image_path2 = 'path_to_image2'

# Load images (assuming they are 3D stacks)
image1 = frames[197]
image2 = frames[196]

# Pixel sizes
pixel_size1 = np.array([14.6484, 14.6484, 30.0])  # nm
pixel_size2 = np.array([10.3760, 10.3760, 30.0])  # nm

# Rescale image2 to match the pixel size of image1
scale_factors = pixel_size2 / pixel_size1
rescaled_image2 = rescale(image2, scale_factors, mode='reflect', anti_aliasing=True)



# Ensure the rescaled image has the same shape as image1 by cropping or padding
def crop_or_pad_image(image, target_shape):
    current_shape = image.shape
    pad_width = [(0, max(target_shape[i] - current_shape[i], 0)) for i in range(len(current_shape))]
    crop_width = [(max(current_shape[i] - target_shape[i], 0) // 2, max(current_shape[i] - target_shape[i], 0) - max(current_shape[i] - target_shape[i], 0) // 2) for i in range(len(current_shape))]

    padded_image = np.pad(image, pad_width, mode='constant', constant_values=0)
    slices = tuple(slice(crop_width[i][0], padded_image.shape[i] - crop_width[i][1]) for i in range(len(current_shape)))
    cropped_image = padded_image[slices]
    
    return cropped_image

rescaled_image2 = crop_or_pad_image(rescaled_image2, image1.shape)

# Now both images should have the same shape
# Register translations and correct shifts
shift, error, diffphase = phase_cross_correlation(image1, rescaled_image2)
img_corr = fourier_shift(np.fft.fftn(rescaled_image2), shift)
img_corr = np.fft.ifftn(img_corr).real

# Normalize the corrected image for display
img_corr_normalized = (img_corr - np.min(img_corr)) / (np.max(img_corr) - np.min(img_corr))

# Display the corrected image
plt.imshow(img_corr_normalized, cmap='gray')
plt.title('Corrected Image')
plt.show()

# Display original and rescaled images for comparison
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(image1, cmap='gray')
plt.title('Original Image 1')

plt.subplot(1, 3, 2)
plt.imshow(rescaled_image2, cmap='gray')
plt.title('Rescaled Image 2')

plt.subplot(1, 3, 3)
plt.imshow(img_corr_normalized, cmap='gray')
plt.title('Corrected Image 2')

plt.show()



###########################################
#rescale  images from folder 2 and save, then calculate the shift wrt to fist image from folder 2 and last image if folder 1, apply the shift transform on all the images from folder 1

import numpy as np
import matplotlib.pyplot as plt
from skimage.registration import phase_cross_correlation
from scipy.ndimage import fourier_shift
from skimage.transform import rescale
from skimage import io, img_as_ubyte
import os

import  pims 

frames_set1 = pims.ImageSequence('H:/Drosophilla_Antenna_IV/PluginData/SEM/Detail/LiveTracking/*.tif')
for  i in range(0,len(frames_set1)):
    img =  frames_set1[i]
    cv2.imwrite('H:/new/set1/' + str(i) + '.tiff', img)
    print(i)
    
frames_set2 = pims.ImageSequence('H:/Drosophilla_Antenna_IV/PluginData/SEM/Detail2/LiveTracking/*.tif')
for  i in range(0,len(frames_set2)):
    img =  frames_set2[i]
    cv2.imwrite('H:/new/set2/' + str(i) + '.tiff', img)
    print(i)


# Define file paths and folders
folder1 = "H:/new/set1"  # Replace with the actual folder path
folder2 = "H:/new/set2"  # Replace with the actual folder path
output_folder = "H:/new/results"   # Replace with the actual folder path

# Ensure output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Pixel sizes
pixel_size1 = np.array([10.3760, 10.3760, 30.0])  # nm
pixel_size2 = np.array([14.6484, 14.6484, 30.0])  # nm

# Helper function to crop or pad images to a target shape
def crop_or_pad_image(image, target_shape):
    current_shape = image.shape
    pad_width = [(0, max(target_shape[i] - current_shape[i], 0)) for i in range(len(current_shape))]
    crop_width = [(max(current_shape[i] - target_shape[i], 0) // 2, max(current_shape[i] - target_shape[i], 0) - max(current_shape[i] - target_shape[i], 0) // 2) for i in range(len(current_shape))]

    padded_image = np.pad(image, pad_width, mode='constant', constant_values=0)
    slices = tuple(slice(crop_width[i][0], padded_image.shape[i] - crop_width[i][1]) for i in range(len(current_shape)))
    cropped_image = padded_image[slices]
    
    return cropped_image

# Load the last image of the first set
image1_last_path = os.path.join(folder1, '481.tiff')  # Adjust if needed
image1_last = io.imread(image1_last_path)

# Process images from the first set and save them to the output folder
for i in range(0,481):
    image_path = os.path.join(folder1, f'{i}.tiff')
    image = io.imread(image_path)
    output_path = os.path.join(output_folder, f'aligned_image_{i:04d}.tif')
    io.imsave(output_path, img_as_ubyte(image))
    print(i)

# Load the first image of the second set
image2_first_path = os.path.join(folder2, '0.tiff')
image2_first = io.imread(image2_first_path)

# Rescale the first image of the second set
scale_factors = (pixel_size2 / pixel_size1)[:2] 
rescaled_image2_first = rescale(image2_first, scale_factors, mode='reflect', anti_aliasing=True)
rescaled_image2_first = crop_or_pad_image(rescaled_image2_first, image1_last.shape)

# Find the shift
shift, error, diffphase = phase_cross_correlation(image1_last, rescaled_image2_first)

# Process and save the shifted images from the second set
for i in range(0,846):
    image_path = os.path.join(folder2, f'{i}.tiff')
    image = io.imread(image_path)
    
    # Rescale the image
    rescaled_image = rescale(image, scale_factors, mode='reflect', anti_aliasing=True)
    rescaled_image = crop_or_pad_image(rescaled_image, image1_last.shape)
    
    # Apply the shift correction
    shifted_image = np.fft.ifftn(fourier_shift(np.fft.fftn(rescaled_image), shift)).real
    
    # Normalize and save the corrected image
    shifted_image_normalized = (shifted_image - np.min(shifted_image)) / (np.max(shifted_image) - np.min(shifted_image))
    output_path = os.path.join(output_folder, f'aligned_image_{481 + i:04d}.tif')
    io.imsave(output_path, img_as_ubyte(shifted_image_normalized))
    print( f'aligned_image_{481 + i:04d}.tif')

import pims
from matplotlib import pyplot as plt
import cv2
frames=pims.ImageSequence('F:/Drosophilla_Antenna_IV/PluginData/SEM/chamber2/result/*.tif')
for i in range(0,len(frames)):
    img=frames[i]
    if img.ndim == 3:
        if img.shape[2] == 3:  # RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.shape[2] == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            raise ValueError(f"Unexpected number of channels in image {i}: {img.shape}")
    elif img.ndim == 2:  # Already grayscale
        pass
    else:
        raise ValueError(f"Unexpected image dimensions in image {i}: {img.shape}")
    cv2.imwrite('F:/Drosophilla_Antenna_IV/PluginData/SEM/shift_corrected/' + str(i) + '.tiff', img)
    print(i)
    
    
###############
###############
######################






import numpy as np
import matplotlib.pyplot as plt
from skimage.registration import phase_cross_correlation
from scipy.ndimage import fourier_shift
from skimage.transform import rescale
from skimage import io, img_as_ubyte
import os

import  pims 

frames_set1 = pims.ImageSequence('H:/Drosophilla_Antenna_IV/PluginData/SEM/Detail/LiveTracking/*.tif')
for  i in range(0,len(frames_set1)):
    img =  frames_set1[i]
    cv2.imwrite('H:/new/set1/' + str(i) + '.tiff', img)
    print(i)
    
frames_set2 = pims.ImageSequence('H:/Drosophilla_Antenna_IV/PluginData/SEM/Detail2/LiveTracking/*.tif')
for  i in range(0,len(frames_set2)):
    img =  frames_set2[i]
    cv2.imwrite('H:/new/set2/' + str(i) + '.tiff', img)
    print(i)


# Define file paths and folders
folder1 = "H:/new/set1"  # Replace with the actual folder path
folder2 = "H:/new/set2"  # Replace with the actual folder path
output_folder = "H:/new/results"   # Replace with the actual folder path

# Ensure output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Pixel sizes
pixel_size1 = np.array([10.3760, 10.3760, 30.0])  # nm
pixel_size2 = np.array([14.6484, 14.6484, 30.0])  # nm

scale_factors = (pixel_size2 / pixel_size1)[:2] 

# Helper function to crop or pad images to a target shape
# def crop_or_pad_image(image, target_shape):
#     current_shape = image.shape
#     pad_width = [(0, max(target_shape[i] - current_shape[i], 0)) for i in range(len(current_shape))]
#     crop_width = [(max(current_shape[i] - target_shape[i], 0) // 2, max(current_shape[i] - target_shape[i], 0) - max(current_shape[i] - target_shape[i], 0) // 2) for i in range(len(current_shape))]

#     padded_image = np.pad(image, pad_width, mode='constant', constant_values=0)
#     slices = tuple(slice(crop_width[i][0], padded_image.shape[i] - crop_width[i][1]) for i in range(len(current_shape)))
#     cropped_image = padded_image[slices]
    
#     return cropped_image

def crop_or_pad_image(image, target_shape):
    current_shape = image.shape
    pad_width = [(0, max(target_shape[i] - current_shape[i], 0)) for i in range(len(current_shape))]
    
    # Adjust crop_width to crop from the bottom instead of evenly
    crop_width = [(0, max(current_shape[i] - target_shape[i], 0)) if i == 0 else (max(current_shape[i] - target_shape[i], 0) // 2, max(current_shape[i] - target_shape[i], 0) - max(current_shape[i] - target_shape[i], 0) // 2) for i in range(len(current_shape))]

    padded_image = np.pad(image, pad_width, mode='constant', constant_values=0)
    slices = tuple(slice(crop_width[i][0], padded_image.shape[i] - crop_width[i][1]) for i in range(len(current_shape)))
    cropped_image = padded_image[slices]
    
    return cropped_image


# Load the last image of the first set
image1_last_path = os.path.join(folder1, '481.tiff')  # Adjust if needed
image1_last = io.imread(image1_last_path)

# Load the first image of the second set
image2_first_path = os.path.join(folder2, '0.tiff')  # Adjust if needed
image2_first = io.imread(image1_last_path)

# Process images from the first set and save them to the output folder
for i in range(0,847):
    image_path = os.path.join(folder2, f'{i}.tiff')
    image = io.imread(image_path)
    output_path = os.path.join(output_folder, f'aligned_image_{481+i:04d}.tif')
    rescaled_image2_first = rescale(image, scale_factors, mode='reflect', anti_aliasing=True)
    rescaled_image2_first = crop_or_pad_image(rescaled_image2_first, image1_last.shape)
    io.imsave(output_path, img_as_ubyte( rescaled_image2_first))
    print(i)

# Load the last image of the first set
image1_last_path = os.path.join(folder1, '481.tiff')  # Adjust if needed
image1_last = io.imread(image1_last_path)


rescaled_image2_first =io.imread('H:/new/results/aligned_image_0481.tif')
# Find the shift
shift, error, diffphase = phase_cross_correlation(rescaled_image2_first,image1_last)

# Process and save the shifted images from the second set
for i in range(0,480):
    image_path = os.path.join(folder1, f'{i}.tiff')
    image = io.imread(image_path)
    

    
    # Apply the shift correction
    shifted_image = np.fft.ifftn(fourier_shift(np.fft.fftn(image), shift)).real
    
    # Normalize and save the corrected image
    shifted_image_normalized = (shifted_image - np.min(shifted_image)) / (np.max(shifted_image) - np.min(shifted_image))
    output_path = os.path.join(output_folder, f'aligned_image_{i:04d}.tif')
    io.imsave(output_path, img_as_ubyte(shifted_image_normalized))
    print( f'aligned_image_{i:04d}.tif')
    

    
f=pims.ImageSequence('H:/new/results/*.tif')
np.amax(f[0])
#######

def crop_or_pad_image(image, target_shape):
    current_shape = image.shape
    pad_width = [(0, max(target_shape[i] - current_shape[i], 0)) for i in range(len(current_shape))]
    
    # Adjust crop_width to crop from the bottom instead of evenly
    crop_width = [(0, max(current_shape[i] - target_shape[i], 0)) if i == 0 else (max(current_shape[i] - target_shape[i], 0) // 2, max(current_shape[i] - target_shape[i], 0) - max(current_shape[i] - target_shape[i], 0) // 2) for i in range(len(current_shape))]

    padded_image = np.pad(image, pad_width, mode='constant', constant_values=0)
    slices = tuple(slice(crop_width[i][0], padded_image.shape[i] - crop_width[i][1]) for i in range(len(current_shape)))
    cropped_image = padded_image[slices]
    
    return cropped_image

i=400
image_path = os.path.join(folder2, f'{i}.tiff')
image = io.imread(image_path)
rescaled_image2_first = rescale(image, scale_factors, mode='reflect', anti_aliasing=True)
rescaled_image2_first = crop_or_pad_image(rescaled_image2_first, image1_last.shape)
io.imsave(output_path, img_as_ubyte( rescaled_image2_first))
