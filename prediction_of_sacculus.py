# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 13:48:41 2024

@author: Ganesh
"""

from UNET import simple_unet_model   #Use normal unet model
from tensorflow.keras.utils import normalize
import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

from scipy.signal import find_peaks
from scipy import ndimage

IMG_WIDTH=256
IMG_HEIGHT=256
IMG_CHANNELS=1

import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

def preprocess_image(image_path, size):
    """
    Load and preprocess the image.
    """
    image = cv2.imread(image_path, 0)  # Read in grayscale
    image = Image.fromarray(image)
    image = image.resize((size, size))
    image = np.array(image)
    image = normalize(image, axis=1)
    image = np.expand_dims(image, axis=-1)  # Expand dims to match model input
    image = np.expand_dims(image, axis=0)  # Expand dims to add batch size
    return image

def predict_image(model, image_path, size=256):
    """
    Predict the mask for a given image using the trained model.
    """
    preprocessed_image = preprocess_image(image_path, size)
    prediction = model.predict(preprocessed_image)
    predicted_mask = np.squeeze(prediction, axis=0)  # Remove batch dimension
    predicted_mask = np.squeeze(predicted_mask, axis=-1)  # Remove channel dimension
    return predicted_mask

def display_results(image_path, predicted_mask):
    """
    Display the original image and its predicted mask side by side.
    """
    original_image = cv2.imread(image_path, 0)
    original_image = cv2.resize(original_image, (predicted_mask.shape[1], predicted_mask.shape[0]))
    
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.title('Original Image')
    plt.imshow(original_image, cmap='gray')
    
    plt.subplot(122)
    plt.title('Predicted Mask')
    plt.imshow(predicted_mask, cmap='gray')
    
    plt.show()

# Example usage:

def get_model():
    return simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

model1 = get_model()    
model2 = get_model()

model1.load_weights('F:/test_unet/sacculus_segmentation_20_epochs.hdf5')
model2.load_weights('F:/test_unet/sacculus_segmentation_25_epochs.hdf5')

image_p = "H:/cropped_sacculus/"
image_p = 'H:/new/results'


import glob

image_list = []
for file in glob.glob(image_p +"/*.tif"):
    file = file.replace('\\', '/')
    image_list.append(file)



i=920
for i in range(0,len(image_list)):
    image_path = image_list[i]
    predicted_mask15 = predict_image(model1, image_path )
    predicted_mask25 = predict_image(model2, image_path )
    
    predicted_mask=predicted_mask15*predicted_mask25
    
   
    predicted_mask = predicted_mask/np.amax(predicted_mask)
    #display_results(image_path, predicted_mask)
    
    threshold = 0.3

    # Apply the threshold to get a binary mask
    binary_mask = (predicted_mask > threshold).astype(np.uint8)

    # Optionally, scale the binary mask to 0-255 for visualization
    binary_mask_visual = binary_mask * 255
    plt.imshow(binary_mask_visual)
 

    cv2.imwrite( 'H:/new/predicted_mask/' + str(i) +'.png' , binary_mask_visual)
    #cv2.imwrite('H:/test_unet/outline/' + str(i) +'.png' , mask)
    print(i)
     

     ######################################
     
     
     
    input_size = (256, 256) 
     # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
     
     # Draw contours on the original image
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    original_image = cv2.resize(original_image, input_size)
    contour_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)  # Convert to BGR for color display
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 1)  # Draw contours in green
     
     # Display the original image with contours
     
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_image, cmap='gray')
     
    plt.subplot(1, 2, 2)
    plt.title("Image with Contours")
    plt.imshow(contour_image)
    plt.show()



#############


########################
for i in range(0,len(image_list)):
    image_path = image_list[i]
    predicted_mask20 = predict_image(model1, image_path )
    predicted_mask25 = predict_image(model2, image_path )
    
    predicted_mask=predicted_mask20*predicted_mask25
    
   
    predicted_mask = predicted_mask/np.amax(predicted_mask)
    display_results(image_path, predicted_mask)
    
    threshold = 0.1

    # Apply the threshold to get a binary mask
    binary_mask = (predicted_mask > threshold).astype(np.uint8)

    # Optionally, scale the binary mask to 0-255 for visualization
    binary_mask_visual = binary_mask * 255
    plt.imshow(binary_mask_visual)
 

    cv2.imwrite('H:/test_unet/predicted_mask2/' + str(i) +'.png' , binary_mask_visual)
    #cv2.imwrite('H:/test_unet/outline/' + str(i) +'.png' , mask)
    print(i)

#################
import random
predictions = model1.predict(X_test, verbose=1)

n=10
predicted_mask=predictions[n]
predicted_mask=predicted_mask/np.amax(predicted_mask)
predicted_mask=(predicted_mask>0.1)*1
predicted_mask=np.reshape(predicted_mask, (256,256))

ground_truth = y_test[n] 
ground_truth = ground_truth/np.amax(ground_truth)
ground_truth=np.reshape(ground_truth, (256,256))

flat_pred = predicted_mask.flatten()
flat_ground = ground_truth.flatten()

intersection = np.sum(flat_ground * flat_pred)
union = np.sum(flat_ground) + np.sum(flat_pred) - intersection

iou=intersection/union



# Function to calculate IoU
def calculate_iou(y_true, y_pred, threshold=0.5):
    y_pred = y_pred / np.amax(y_pred)
    y_pred_binary = (y_pred > threshold).astype(int)
    
    y_true = y_true / np.amax(y_true)
    y_true_binary = (y_true > threshold).astype(int)
    
    intersection = np.sum(y_true_binary * y_pred_binary)
    union = np.sum(y_true_binary) + np.sum(y_pred_binary) - intersection
    
    return intersection / union

# Compute IoU for each test image
ious = []
for i in range(len(X_test)):
    iou = calculate_iou(y_test[i], predictions[i])
    ious.append(iou)
    print(f"Image {i}: IoU = {iou}")

# Calculate the average IoU
average_iou = np.mean(ious)
print(f"Average IoU: {average_iou}")

# Visualize predictions for a few random test images
for _ in range(5):  # Visualize 5 test images
    image_number = random.randint(0, len(X_test)-1)
    
    predicted_mask = predictions[image_number] / np.amax(predictions[image_number])
    predicted_mask = (predicted_mask > 0.1).astype(int)
    predicted_mask = np.reshape(predicted_mask, (256, 256))
    
    ground_truth = y_test[image_number] / np.amax(y_test[image_number])
    ground_truth = np.reshape((ground_truth > 0.1).astype(int), (256, 256))

    plt.figure(figsize=(12, 6))
    plt.subplot(131)
    plt.title('Test Image')
    plt.imshow(np.reshape(X_test[image_number], (256, 256)), cmap='gray')
    plt.subplot(132)
    plt.title('Ground Truth')
    plt.imshow(np.reshape(ground_truth, (256, 256)), cmap='gray')
    plt.subplot(133)
    plt.title('Prediction')
    plt.imshow(np.reshape(predicted_mask, (256, 256)), cmap='gray')
    plt.show()






plt.subplot(131)
plt.title('Test Image')
plt.imshow(np.reshape(X_test[n], (256, 256)), cmap='gray')
plt.subplot(132)
plt.title('Ground Truth')
plt.imshow(np.reshape(ground_truth, (256, 256)), cmap='gray')
plt.subplot(133)
plt.title('Prediction')
plt.imshow(np.reshape(predicted_mask, (256, 256)) , cmap='gray')  # Apply threshold
plt.show()




threshold = 0.1

# Apply the threshold to get a binary mask
binary_mask = (predicted_mask > threshold).astype(np.uint8)

# Optionally, scale the binary mask to 0-255 for visualization
binary_mask_visual = binary_mask * 255

# Display the original and binary masks
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Predicted Mask (Probability)')
plt.imshow(predicted_mask, cmap='gray')

plt.subplot(1, 2, 2)
plt.title('Binary Mask (Thresholded)')
plt.imshow(binary_mask_visual, cmap='gray')
plt.show()
