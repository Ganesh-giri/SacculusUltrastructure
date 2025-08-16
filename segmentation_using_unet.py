from UNET import simple_unet_model
from tensorflow.keras.utils import normalize
import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

# image_directory = 'H:/test_unet/images_to_train/downsampled/images/'
# mask_directory = 'H:/test_unet/images_to_train/downsampled/labels/'

# image_directory = 'H:/new/train/aug_images/'  #### sanole 2
# mask_directory = 'H:/new/train/aug_labels/'


image_directory = "F:/test_unet/images_to_train/aug_images_combined/"  #### sample1 and 2 combined 
mask_directory = "F:/test_unet/images_to_train/aug_labels_combined/"


SIZE = 256
image_dataset = []
mask_dataset = []

images = os.listdir(image_directory)
for i, image_name in enumerate(images):
    image = cv2.imread(image_directory+image_name, 0)
    image = Image.fromarray(image)
    image = image.resize((SIZE, SIZE))
    image_dataset.append(np.array(image))
    print(i)

masks = os.listdir(mask_directory)
for i, image_name in enumerate(masks):
    image = cv2.imread(mask_directory+image_name, 0)
    image = Image.fromarray(image)
    image = image.resize((SIZE, SIZE))
    mask_dataset.append(np.array(image))
    print(i)

image_dataset = np.expand_dims(normalize(np.array(image_dataset), axis=1), 3)
mask_dataset = np.expand_dims((np.array(mask_dataset)), 3) / 255.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size=0.10, random_state=0)

import random
image_number = random.randint(0, len(X_train))
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(np.reshape(X_train[image_number], (256, 256)), cmap='gray')
plt.subplot(122)
plt.imshow(np.reshape(y_train[image_number], (256, 256)), cmap='gray')
plt.show()

IMG_HEIGHT = image_dataset.shape[1]
IMG_WIDTH = image_dataset.shape[2]
IMG_CHANNELS = image_dataset.shape[3]

import tensorflow as tf

def iou_metric(y_true, y_pred, threshold=0.1, smooth=1e-6):
    # Normalize y_pred and y_true to be between 0 and 1
    y_pred = y_pred / tf.reduce_max(y_pred)
    y_true = y_true / tf.reduce_max(y_true)
    
    y_pred = tf.cast(y_pred > threshold, tf.float32)
    y_true = tf.cast(y_true > threshold, tf.float32)
    
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou

def get_model():
    model = simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', iou_metric])
    return model

model = get_model()

from tensorflow.keras.utils import plot_model

# Visualize the model
plot_model(model, to_file='E:/EM_data_results/Compiled_results/model.png', show_shapes=False, show_layer_names=False, dpi=100)
plot_model(model, to_file='E:/EM_data_results/Compiled_results/model2.png', show_shapes=True, show_layer_names=True, dpi=50, figsize=(10, 10))
img = plt.imread('E:/EM_data_results/Compiled_results/model.png')
plt.figure(figsize=(10, 10))
plt.imshow(img)
plt.axis('off')
plt.show()


from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

checkpoint = ModelCheckpoint('E:/EM_data_results/Compiled_results/Sacculus_seg_best_model.h5', monitor='iou_metric', verbose=1, 
                             save_best_only=True, mode='max')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6, verbose=1)
callbacks = [checkpoint, reduce_lr]


history = model.fit(X_train, y_train, 
                    batch_size=1, 
                    epochs=50, 
                    validation_data=(X_test, y_test), 
                    shuffle=False,
                    callbacks=callbacks)


def plot_history(history):
    plt.figure(figsize=(15, 4))
    
    # Plot accuracy
    plt.subplot(121)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.legend()
    
    # Plot IoU
    plt.subplot(122)
    plt.plot(history.history['iou_metric'], label='Training IoU')
    plt.plot(history.history['val_iou_metric'], label='Validation IoU')
    plt.title('Training and Validation IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.ylim(0, 1)
    plt.legend()

    # Save the figure
    plt.savefig('training_history.png', dpi=600, bbox_inches='tight')
    plt.show()

# Call the function to plot and save the history
plot_history(history)

import pandas as pd
# convert the history.history dict to a pandas DataFrame:     
hist_df = pd.DataFrame(history.history) 
hist_csv_file = 'E:/EM_data_results/Compiled_results/history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

##############
def get_model():
    return simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

model = get_model()    
model.load_weights('E:/EM_data_results/Compiled_results/Sacculus_seg_best_model.h5')


# def get_model():
#     return simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# model = get_model()    
# model=get_model()

# model.load_weights('H:/test_unet/sacculus_segmentation_20_epochs.hdf5')

#Evaluate the model
import random
predictions = model.predict(X_test, verbose=1)

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
    plt.savefig('E:/EM_data_results/Compiled_results/prediction' + str(image_number) +'.png', dpi=300)
    plt.show()


import pims
frames=pims.ImageSequence("F:/test_unet/images_to_train/aug_images_combined/*.png"  )
plt.imshow(frames[178])
img=frames[178]
img = Image.fromarray(img)
img = img.resize((256, 256))

    
from scipy import signal 
weights = model.layers[1].get_weights()[0]  # Get the weights of the first convolutional layer
kernel0 =  weights[:, :, 0, 0]
kernel1 =  weights[:, :, 0, 1]
kernel2 =  weights[:, :, 0, 2]
kernel3 =  weights[:, :, 0, 3]
kernel4 =  weights[:, :, 0, 4]
kernel5 =  weights[:, :, 0, 5]
 
    
kernels =[kernel0,kernel1,kernel2,kernel3,kernel4, kernel5]
for i in range(0,len(kernels)):
    conv = signal.convolve2d(img, kernels[i], boundary='symm', mode='same')
    
    plt.figure(figsize=(12, 6))
    plt.subplot(131)
    plt.imshow(img, cmap='gray')
    plt.title('Original image')
    
    plt.subplot(132)
    plt.imshow(kernels[i], cmap='gray')
    plt.title('Kernel')
    
    
    plt.subplot(133)
    plt.imshow(conv, cmap='gray')
    plt.title('convolved image')
    plt.savefig('E:/EM_data_results/Compiled_results/example_convolution' + str(i) +'.png', dpi=300)
    


conv = signal.convolve2d(img, kernels[4], boundary='symm', mode='same')
# Applying ReLU activation
relu_feature_map = np.maximum(0, conv)

# Visualizing the feature maps
plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.title('Feature Map Before ReLU')
plt.imshow(conv, cmap='gray', interpolation='none')
plt.colorbar()

plt.subplot(122)
plt.title('Feature Map After ReLU')
plt.imshow(relu_feature_map, cmap='gray', interpolation='none')
plt.colorbar()

plt.show()


# Apply Max Pooling (2x2 pooling with stride 2)
max_pooled_image = tf.nn.max_pool2d(
    np.expand_dims(np.expand_dims(relu_feature_map, axis=0), axis=-1), 
    ksize=2, strides=2, padding='SAME'
).numpy()[0, :, :, 0]

# Plot the original image, ReLU feature map, and max pooled image
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(img, cmap='gray')

plt.subplot(1, 3, 2)
plt.title("ReLU Feature Map")
plt.imshow(relu_feature_map, cmap='gray')

plt.subplot(1, 3, 3)
plt.title("Max Pooled Image")
plt.imshow(max_pooled_image, cmap='gray')

plt.show()




# Load the image
frames = pims.ImageSequence("F:/test_unet/images_to_train/aug_images_combined/*.png")
img = frames[178]
img = Image.fromarray(img).resize((256, 256))
img = np.array(img)

# Get weights of the first convolutional layer
weights = model.layers[19].get_weights()[0]  # Get the weights of the first convolutional layer
kernel = weights[:, :, 0, 0]  # Choose one kernel for demonstration (e.g., kernel5)

# Apply convolution
conv = signal.convolve2d(img, kernel, boundary='symm', mode='same')

# Apply ReLU activation
relu_feature_map = np.maximum(0, conv)

# Apply Max Pooling (2x2 pooling with stride 2)
max_pooled_image = tf.nn.max_pool2d(
    np.expand_dims(np.expand_dims(relu_feature_map, axis=0), axis=-1),
    ksize=2, strides=2, padding='SAME'
).numpy()[0, :, :, 0]

# Plot the original image, one of the convolved images, ReLU activated image, and max pooled image
plt.figure(figsize=(16,10))

# Original Image
plt.subplot(2, 2, 1)
plt.title("Original Image")
plt.imshow(img, cmap='gray')


# Convolved Image
plt.subplot(2, 2, 2)
plt.title("Convolved Image")
plt.imshow(conv, cmap='gray')


# ReLU Activated Image
plt.subplot(2, 2, 3)
plt.title("ReLU Activated Image")
plt.imshow(relu_feature_map, cmap='gray')


# Max Pooled Image
plt.subplot(2, 2, 4)
plt.title("Max Pooled Image")
plt.imshow(max_pooled_image, cmap='gray')
plt.savefig('E:/EM_data_results/Compiled_results/conv_process.png', dpi=300)


plt.tight_layout()
plt.show()
plt.savefig('E:/EM_data_results/Compiled_results/conv_process.png', dpi=300)



####################
width = 7.205
height = 2
plt.figure(figsize=(width, height))

# Plot accuracy
plt.subplot(121)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy', fontsize=8,family='Arial')
plt.xlabel('Epoch', fontsize=7,family='Arial')
plt.ylabel('Accuracy', fontsize=7,family='Arial')
plt.ylim(0, 1)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)

plt.legend(prop={'size': 7})

# Plot IoU
plt.subplot(122)
plt.plot(history.history['iou_metric'], label='Training IoU')
plt.plot(history.history['val_iou_metric'], label='Validation IoU')
plt.title('Training and Validation IoU', fontsize=8,family='Arial')
plt.xlabel('Epoch', fontsize=7,family='Arial')
plt.ylabel('IoU', fontsize=7,family='Arial')
plt.ylim(0, 1)
plt.legend(prop={'size': 7})
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)

image_number = 49

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

image_number = 91
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


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# Set figure size for the entire plot
width, height = 7.205, 6  # Adjust the height as needed to fit all subplots
plt.figure(figsize=(width, height))

# Create a GridSpec with 3 rows and 6 columns
gs = GridSpec(3, 6)

# 1. Plot Training and Validation Accuracy (spans 3 columns)
plt.subplot(gs[0, 0:3])  # First row, columns 0 to 3
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy', fontsize=8, family='Arial')
plt.xlabel('Epoch', fontsize=7, family='Arial')
plt.ylabel('Accuracy', fontsize=7, family='Arial')
plt.ylim(0, 1)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
plt.legend(prop={'size': 7})

# 2. Plot Training and Validation IoU (spans 3 columns)
plt.subplot(gs[0, 3:6])  # First row, columns 3 to 6
plt.plot(history.history['iou_metric'], label='Training IoU')
plt.plot(history.history['val_iou_metric'], label='Validation IoU')
plt.title('Training and Validation IoU', fontsize=8, family='Arial')
plt.xlabel('Epoch', fontsize=7, family='Arial')
plt.ylabel('IoU', fontsize=7, family='Arial')
plt.ylim(0, 1)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
plt.legend(prop={'size': 7})

# 3. Plot Test Image, Ground Truth, and Prediction for image_number = 49
image_number = 26

predicted_mask = predictions[image_number] / np.amax(predictions[image_number])
predicted_mask = (predicted_mask > 0.1).astype(int)
predicted_mask = np.reshape(predicted_mask, (256, 256))

ground_truth = y_test[image_number] / np.amax(y_test[image_number])
ground_truth = np.reshape((ground_truth > 0.1).astype(int), (256, 256))

plt.subplot(gs[1, 0:2])  # Second row, columns 0 to 2
plt.title('Test Image 49', fontsize=8, family='Arial')
plt.imshow(np.reshape(X_test[image_number], (256, 256)), cmap='gray')
plt.axis('off')

plt.subplot(gs[1, 2:4])  # Second row, columns 2 to 4
plt.title('Ground Truth 49', fontsize=8, family='Arial')
plt.imshow(np.reshape(ground_truth, (256, 256)), cmap='gray')
plt.axis('off')

plt.subplot(gs[1, 4:6])  # Second row, columns 4 to 6
plt.title('Prediction 49', fontsize=8, family='Arial')
plt.imshow(np.reshape(predicted_mask, (256, 256)), cmap='gray')
plt.axis('off')

# 4. Plot Test Image, Ground Truth, and Prediction for image_number = 91
image_number = 91

predicted_mask = predictions[image_number] / np.amax(predictions[image_number])
predicted_mask = (predicted_mask > 0.1).astype(int)
predicted_mask = np.reshape(predicted_mask, (256, 256))

ground_truth = y_test[image_number] / np.amax(y_test[image_number])
ground_truth = np.reshape((ground_truth > 0.1).astype(int), (256, 256))

plt.subplot(gs[2, 0:2])  # Third row, columns 0 to 2
plt.title('Test Image 91', fontsize=8, family='Arial')
plt.imshow(np.reshape(X_test[image_number], (256, 256)), cmap='gray')
plt.axis('off')

plt.subplot(gs[2, 2:4])  # Third row, columns 2 to 4
plt.title('Ground Truth 91', fontsize=8, family='Arial')
plt.imshow(np.reshape(ground_truth, (256, 256)), cmap='gray')
plt.axis('off')

plt.subplot(gs[2, 4:6])  # Third row, columns 4 to 6
plt.title('Prediction 91', fontsize=8, family='Arial')
plt.imshow(np.reshape(predicted_mask, (256, 256)), cmap='gray')
plt.axis('off')

# Adjust layout to avoid overlap
plt.tight_layout()
plt.savefig('E:/EM_data_results/Compiled_results/fig2.png', dpi=600, bbox_inches='tight')
# Show the entire figure
plt.show()








# Save the figure
plt.savefig('E:/EM_data_results/Compiled_results/training_history2.png', dpi=600, bbox_inches='tight')
plt.show()
