import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, BatchNormalization, Flatten, concatenate
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, train_test_split
from utils import process_data
import os
import gc
import psutil
import subprocess

class MemoryCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # System memory
        memory = psutil.virtual_memory()
        print(f"Epoch {epoch}")
        print(f"System memory used: {memory.percent}%")
        
        # GPU memory
        try:
            result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'])
            gpu_memory = [int(x) for x in result.decode('ascii').split(',')]
            print(f"GPU memory used: {gpu_memory[0]} MB / {gpu_memory[1]} MB")
        except:
            print("Unable to fetch GPU memory")

# Settings
pd.set_option('display.max_rows', None)
np.set_printoptions(threshold=100)
# tf.debugging.set_log_device_placement(True)
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Load data
df = pd.read_pickle('saved/dataframes/df.pkl')
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
front_images, back_images, tabular, body_fat = process_data(train_df)
X_test_front_images, X_test_back_images, X_test_tabular, Y_test_body_fat = process_data(test_df)

def print_memory_size(array, name):
    size_bytes = array.nbytes
    size_mb = size_bytes / (1024 ** 2)  # Convert bytes to megabytes
    print(f"{name} size in megabytes: {size_mb:.2f}")

# Print memory sizes for each array
print_memory_size(front_images, "Front Images")
print_memory_size(back_images, "Back Images")
print_memory_size(tabular, "Tabular Data")
print_memory_size(body_fat, "Body Fat")

print_memory_size(X_test_front_images, "X Test Front Images")
print_memory_size(X_test_back_images, "X Test Back Images")
print_memory_size(X_test_tabular, "X Test Tabular")
print_memory_size(Y_test_body_fat, "Y Test Body Fat")

outputs = ['body_fat']

# Define model creation function
def create_model(image_shape, num_tabular_features):
    base_model = MobileNetV2(input_shape=image_shape, include_top=False, weights='imagenet')
    
    # Define a function to process images through the base model
    def process_image(image_input):
        x = base_model(image_input)
        x = GlobalAveragePooling2D()(x)
        x = Flatten()(x)
        return BatchNormalization()(x)
    
    front_image_input = Input(shape=image_shape, name='front_image_input')
    back_image_input = Input(shape=image_shape, name='back_image_input')
    tabular_input = Input(shape=(num_tabular_features,), name='tabular_input')
    
    front_image_features = process_image(front_image_input)
    back_image_features = process_image(back_image_input)

    tabular_features = Flatten()(tabular_input)
    tabular_features = Dense(32, activation='relu')(tabular_features)
    tabular_features = BatchNormalization()(tabular_features)

    combined_features = concatenate([front_image_features, back_image_features, tabular_features])

    output_layers = []

    for output in outputs:
        output_layer = Dense(1, activation='linear', name='output_' + output)(combined_features)
        output_layers.append(output_layer)

    model = Model(
        inputs=[front_image_input, back_image_input, tabular_input], 
        outputs=output_layers
    )

    model.compile(optimizer=AdamW(learning_rate=.0001, weight_decay=.0001), 
                  loss=['mse'],
                  metrics=['mae'])
    
    return model
def train_ensemble_cv(n_models, X_front, X_back, X_tabular, Y, 
                      image_shape, num_tabular_features, n_splits=5):
    histories = []

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for i in range(n_models):
        print(f"Training model {i+1}/{n_models}")

        for fold, (train_index, val_index) in enumerate(kf.split(X_tabular)):
            print(f"  Training on fold {fold + 1}/{n_splits}")
            
            # Split the data
            X_train_front, X_val_front = X_front[train_index], X_front[val_index]
            X_train_back, X_val_back = X_back[train_index], X_back[val_index]
            X_train_tabular, X_val_tabular = X_tabular[train_index], X_tabular[val_index]
            Y_train, Y_val = Y[train_index], Y[val_index]
            
            # Create and train the model
            model = create_model(image_shape, num_tabular_features)
            model_path = f'./saved/models/model_{i}_fold_{fold}.keras'
            checkpoint = ModelCheckpoint(model_path, 
                                         monitor='val_loss', 
                                         save_best_only=True, 
                                         mode='min', 
                                         verbose=1)
            early_stopping = EarlyStopping(monitor='loss', 
                                           patience=20, 
                                           mode='min', 
                                           verbose=1)
            
            history = model.fit(
                [X_train_front, X_train_back, X_train_tabular],
                {'output_body_fat': Y_train},
                epochs=2,
                batch_size=4,
                verbose=1,
                callbacks=[checkpoint, early_stopping, MemoryCallback()],
                validation_data=([X_val_front, X_val_back, X_val_tabular], {'output_body_fat': Y_val})
            ) 
            histories.append(history)

            del model
            tf.keras.backend.clear_session()
            gc.collect()

    return histories

# Define ensemble prediction function
def ensemble_predict(ground_truth, X_front, X_back, X_tabular):
    predictions = []
    weight_sum = 0

    model_paths = os.listdir('./saved/models/')
    for i, model_path in enumerate(model_paths):
        model = tf.keras.models.load_model(os.path.join('./saved/models/', model_path))
        
        predictions_body_fat = model.predict([X_front, X_back, X_tabular]).flatten()
        mae = mean_absolute_error(ground_truth, predictions_body_fat)
        print(f"  Model {i+1}/{len(model_paths)} MAE: {round(mae, 2)}")
        weight = 1 / (mae ** 10)
        weighted_pred = predictions_body_fat * weight
        weight_sum += weight

        predictions.append(weighted_pred)
        del model
        tf.keras.backend.clear_session()
    
    weighted_avg_prediction = np.sum(predictions, axis=0) / weight_sum
    return np.round(weighted_avg_prediction, 1)

def print_evaluation(predictions, ground_truth):
    mae = mean_absolute_error(ground_truth, predictions)
    print(f"Total MAE: {round(mae, 2)}")
    print(f"Predictions: {predictions}")
    print(f"Ground Truth: {ground_truth}")
    errors = predictions_body_fat - ground_truth
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=10)
    plt.title("Distribution of Errors")
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.show()

def plot_histories(histories):
    plt.figure(figsize=(10, 6))
    for i, history in enumerate(histories):
        plt.plot(history.history['loss'], label=f"Model {i} Train")
    plt.title("Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()

    plt.figure(figsize=(10, 6))
    for i, history in enumerate(histories):
        plt.plot(history.history['val_loss'], label=f"Model {i} Validation")
    plt.title("Validation Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()

# Main execution
image_shape = (224, 224, 3)
num_tabular_features = tabular.shape[1] 
n_models = 1
n_splits = 5

# Delete every past file in saved/models
folder_path = './saved/models/'
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    os.remove(file_path)

histories = train_ensemble_cv(n_models, front_images, back_images, tabular, body_fat, image_shape, num_tabular_features, n_splits)

predictions_body_fat = ensemble_predict(Y_test_body_fat, X_test_front_images, X_test_back_images, X_test_tabular)

predictions = {
    'body_fat': predictions_body_fat
}

ground_truth = {
    'body_fat': Y_test_body_fat
}

print_evaluation(predictions_body_fat, Y_test_body_fat)
# plot_histories(histories)


