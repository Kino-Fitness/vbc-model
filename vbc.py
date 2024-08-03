import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, BatchNormalization, Flatten, concatenate
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import mean_absolute_error
from utils import Scalar2Gaussian, process_data, s2g

train_df = pd.read_pickle('saved/dataframes/train_df.pkl')
val_df = pd.read_pickle('saved/dataframes/val_df.pkl')
test_df = pd.read_pickle('saved/dataframes/test_df.pkl')

X_train_front_images, X_train_back_images, X_train_tabular, Y_train_body_fat = process_data(train_df)
X_val_front_images, X_val_back_images, X_val_tabular, Y_val_body_fat = process_data(val_df)
X_test_front_images, X_test_back_images, X_test_tabular, Y_test_body_fat = process_data(test_df)

# Model definition and training functions
def l1_l2_l3_loss(y_true, y_pred, output):
    l1_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    decoded_y_true = tf.expand_dims(s2g[output].decode_tensor(vector=y_true), -1)
    decoded_y_pred = tf.expand_dims(s2g[output].decode_tensor(vector=y_pred), -1)
    
    l2_loss = tf.keras.losses.MeanSquaredError()(decoded_y_true, decoded_y_pred)
    
    total_loss = l1_loss + l2_loss
    return total_loss

def loss_wrapper(output):
    def loss_fn(y_true, y_pred):
        return l1_l2_l3_loss(y_true, y_pred, output)
    return loss_fn

def loss_fn(y_true, y_pred):
    return l1_l2_l3_loss(y_true, y_pred, output)

outputs = ['body_fat']

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
        output_layer = Dense(1000, activation='softmax', name='output_' + output)(combined_features)
        output_layers.append(output_layer)

    model = Model(
        inputs=[front_image_input, back_image_input, tabular_input], 
        outputs=output_layers
    )

    losses = {f'output_{output}': loss_wrapper(output) for output in outputs}

    model.compile(optimizer=Adam(learning_rate=.00001),
                  loss=losses,
                  metrics=['mae'])
    
    return model

# Ensemble learning functions
def create_bootstrap_sample(X_front, X_back, X_tabular, Y):
    n_samples = X_front.shape[0]
    indices = np.random.choice(n_samples, n_samples, replace=True)
    return X_front[indices], X_back[indices], X_tabular[indices], Y[indices]

def train_ensemble(n_models, X_train_front, X_train_back, X_train_tabular, Y_train, 
                   X_val_front, X_val_back, X_val_tabular, Y_val, 
                   image_shape, num_tabular_features):
    models = []
    for i in range(n_models):
        print(f"Training model {i+1}/{n_models}")
        X_boot_front, X_boot_back, X_boot_tabular, Y_boot = create_bootstrap_sample(
            X_train_front, X_train_back, X_train_tabular, Y_train)
        
        model = create_model(image_shape, num_tabular_features)\
        

        model_path = f'./saved/models/model_{i}.keras'
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
            [X_boot_front, X_boot_back, X_boot_tabular],
            {'output_body_fat': Y_boot},
            epochs=1,
            batch_size=16,
            verbose=1,
            callbacks=[checkpoint, early_stopping],
            validation_data=([X_val_front, X_val_back, X_val_tabular], {'output_body_fat': Y_val})
        )

        custom_objects={
            'loss_fn': loss_fn,
            'l1_l2_l3_loss': l1_l2_l3_loss,
            'Scalar2Gaussian': Scalar2Gaussian,
            's2g': s2g
        }
        
        loaded_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        models.append(loaded_model)
    
    return models

def ensemble_predict(models, X_front, X_back, X_tabular):
    predictions = []
    for model in models:
        pred = model.predict([X_front, X_back, X_tabular])
        predictions.append(pred)
    
    # Average the predictions
    avg_prediction = np.mean(predictions, axis=0)
    return avg_prediction

# Main execution
image_shape = (224, 224, 3)
num_tabular_features = 4

# Train the ensemble
n_models = 3  # Number of models in the ensemble
ensemble = train_ensemble(n_models, X_train_front_images, X_train_back_images, X_train_tabular, Y_train_body_fat, X_val_front_images, X_val_back_images, X_val_tabular, Y_val_body_fat, image_shape, num_tabular_features)

# Make predictions using the ensemble
predictions_body_fat = ensemble_predict(ensemble, X_test_front_images, X_test_back_images, X_test_tabular)
predictions_body_fat.shape

def decode_scalar(vector, output):
    return [np.round(s2g[output].decode(prediction), 1) for prediction in vector]

def print_predictions(predictions, ground_truth):
    mae = {}
    variance = {}
    for output in outputs:
        y_actual = decode_scalar(eval(f'Y_test_{output}'), output)
        y_pred = predictions[output]
        mae[output] = mean_absolute_error(y_actual, y_pred)
        variance[output] = np.mean(np.abs(np.subtract(y_actual, y_pred)) / y_actual)

    total_mae = sum(mae.values()) / len(mae)
    print(f"Total MAE: {round(total_mae, 2)}")
    for output in outputs:
        print(f"MAE {output.capitalize()}: {round(mae[output], 2)}")

    total_variance = sum(variance.values()) / len(variance)
    print(f"Total Variance: {round(total_variance, 2)}")
    for output in outputs:
        print(f"Variance {output.capitalize()}: {round(variance[output], 2)}")
        
    print(f"Predictions: {predictions}")
    print(f"Ground Truth: {ground_truth}")


predictions = {
    'body_fat': decode_scalar(predictions_body_fat, 'body_fat'),
}

ground_truth = {
    'body_fat': decode_scalar(Y_test_body_fat, 'body_fat'),
}

print_predictions(predictions, ground_truth)

