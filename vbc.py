import pandas as pd
import numpy as np
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

# Settings
pd.set_option('display.max_rows', None)
np.set_printoptions(threshold=100)
# tf.debugging.set_log_device_placement(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Load data
df = pd.read_pickle('saved/dataframes/df.pkl')
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
front_images, back_images, tabular, body_fat = process_data(train_df)
X_test_front_images, X_test_back_images, X_test_tabular, Y_test_body_fat = process_data(test_df)

# Delete every past file in saved/models
folder_path = './saved/models/'
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    os.remove(file_path)

outputs = ['body_fat']

# Define model creation function
def create_model(image_shape, num_tabular_features):
    base_model = MobileNetV2(input_shape=image_shape, include_top=False, weights='imagenet')
    
    front_image_input = Input(shape=image_shape, name='front_image_input')
    front_image_x = base_model(front_image_input)
    front_image_x = GlobalAveragePooling2D()(front_image_x)
    front_image_features = Flatten()(front_image_x)
    front_image_features = BatchNormalization()(front_image_features)

    back_image_input = Input(shape=image_shape, name='back_image_input')
    back_image_x = base_model(back_image_input)
    back_image_x = GlobalAveragePooling2D()(back_image_x)
    back_image_features = Flatten()(back_image_x)
    back_image_features = BatchNormalization()(back_image_features)

    tabular_input = Input(shape=(num_tabular_features,), name='tabular_input')
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

    model.compile(optimizer=AdamW(learning_rate=.001, weight_decay=.001), 
                  loss=['mse'],
                  metrics=['mae'])
    
    return model
def train_ensemble_cv(n_models, X_front, X_back, X_tabular, Y, 
                      image_shape, num_tabular_features, n_splits=5):
    weights = [1] * n_splits
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
                                           patience=10, 
                                           mode='min', 
                                           verbose=1)
            
            model.fit(
                [X_train_front, X_train_back, X_train_tabular],
                {'output_body_fat': Y_train},
                epochs=100,
                batch_size=8,
                verbose=1,
                callbacks=[checkpoint, early_stopping],
                validation_data=([X_val_front, X_val_back, X_val_tabular], {'output_body_fat': Y_val})
            ) 

            # Save model path and delete model from memory
            # model_paths.append(model_path)
            del model
            
            # # Load the best model for this fold
            # loaded_model = tf.keras.models.load_model(model_path)

            # # Predict on the validation fold
            # predictions_body_fat = loaded_model.predict([X_val_front, X_val_back, X_val_tabular]).flatten()
            # mae = mean_absolute_error(Y_val, predictions_body_fat)
            # weights.append(1 / mae)

            # Delete loaded model from memory
            # del loaded_model
            tf.keras.backend.clear_session()
            gc.collect()

    return weights

# Define ensemble prediction function
def ensemble_predict(weights, X_front, X_back, X_tabular):
    predictions = []
    model_paths = os.listdir('./saved/models/')
    for i, model_path in enumerate(model_paths):
        model = tf.keras.models.load_model(os.path.join('./saved/models/', model_path))
        pred = model.predict([X_front, X_back, X_tabular])
        weighted_pred = pred.flatten() * weights[i]
        predictions.append(weighted_pred)
        del model
        tf.keras.backend.clear_session()
    
    weighted_avg_prediction = np.sum(predictions, axis=0) / np.sum(weights)
    return np.round(weighted_avg_prediction, 1)

def print_evaluation(predictions, ground_truth):
    mae = mean_absolute_error(ground_truth, predictions)
    print(f"Total MAE: {round(mae, 2)}")
    print(f"Predictions: {predictions}")
    print(f"Ground Truth: {ground_truth}")

# Main execution
image_shape = (224, 224, 3)
num_tabular_features = tabular.shape[1] 
n_models = 1
n_splits = 4

weights = train_ensemble_cv(n_models, front_images, back_images, tabular, body_fat, image_shape, num_tabular_features, n_splits)

predictions_body_fat = ensemble_predict(weights, X_test_front_images, X_test_back_images, X_test_tabular)

predictions = {
    'body_fat': predictions_body_fat
}

ground_truth = {
    'body_fat': Y_test_body_fat
}

print_evaluation(predictions_body_fat, Y_test_body_fat)
