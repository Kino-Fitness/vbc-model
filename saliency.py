import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2  # For Gaussian blur
from utils import process_data

test_df = pd.read_pickle('saved/dataframes/test_df.pkl')
X_test_front_images, X_test_back_images, X_test_tabular, Y_test_body_fat = process_data(test_df)

X_test_front_images_tensor = tf.convert_to_tensor(X_test_front_images, dtype=tf.float32)
X_test_back_images_tensor = tf.convert_to_tensor(X_test_back_images, dtype=tf.float32)
X_test_tabular_tensor = tf.convert_to_tensor(X_test_tabular, dtype=tf.float32)

model_path = './saved/models/model_0.keras'
loaded_model = tf.keras.models.load_model(model_path)

outputs = ['body_fat']
output_index = 0  # Index of the output for which you want to create the saliency map

def normalize(x):
    """Normalize the input tensor to the range [0, 1]."""
    return (x - tf.reduce_min(x)) / (tf.reduce_max(x) - tf.reduce_min(x))

def saliency_map(model, X_front, X_back, X_tabular, output_index):
    with tf.GradientTape() as tape:
        tape.watch(X_front)
        tape.watch(X_back)
        tape.watch(X_tabular)
        outputs = model([X_front, X_back, X_tabular])
        output = outputs[output_index]
    
    gradients = tape.gradient(output, [X_front, X_back, X_tabular])
    saliency_front = tf.reduce_sum(tf.abs(gradients[0]), axis=-1)
    saliency_back = tf.reduce_sum(tf.abs(gradients[1]), axis=-1)
    saliency_tabular = gradients[2]  # Do not reduce sum for tabular data
    
    return saliency_front, saliency_back, saliency_tabular

saliency_front, saliency_back, saliency_tabular = saliency_map(loaded_model, X_test_front_images_tensor, X_test_back_images_tensor, X_test_tabular_tensor, output_index)

# Normalize and apply Gaussian blur to saliency maps
saliency_front = normalize(saliency_front)
saliency_back = normalize(saliency_back)

saliency_front_blurred = cv2.GaussianBlur(saliency_front.numpy()[0], (5, 5), 0)
saliency_back_blurred = cv2.GaussianBlur(saliency_back.numpy()[0], (5, 5), 0)

# Plotting the saliency maps
plt.figure(figsize=(16, 6))  # Increase the figure size
plt.subplot(1, 3, 1)
plt.imshow(saliency_front_blurred, cmap='hot')
plt.title('Front Image Saliency Map')
plt.axis('off')
# Add a colorbar for the heat map
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(saliency_back_blurred, cmap='hot')
plt.title('Back Image Saliency Map')
plt.axis('off')
# Add a colorbar for the heat map
plt.colorbar()

plt.subplot(1, 3, 3)
plt.bar(range(len(saliency_tabular.numpy()[0])), saliency_tabular.numpy()[0])
plt.title('Tabular Data Saliency Map')
plt.xlabel('Feature Index')
plt.ylabel('Saliency Value')

# Modify the x-axis labels
plt.xticks(range(len(saliency_tabular.numpy()[0])), ['Height', 'Weight', 'Waist/Hips Ratio'])
plt.show()