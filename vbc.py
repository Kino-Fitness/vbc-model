import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, train_test_split
from utils import process_data, augment_data
import os
import gc
import concurrent.futures
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

# Settings
pd.set_option('display.max_rows', None)
np.set_printoptions(threshold=100)

# Load data
df = pd.read_pickle('saved/dataframes/df.pkl')
df = df[df['Gender'] == 'Male']
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

front_images, back_images, tabular, body_fat, muscle_mass, bone_mass, bone_density = process_data(train_df)
X_test_front_images, X_test_back_images, X_test_tabular, Y_test_body_fat, Y_test_muscle_mass, Y_test_bone_mass, Y_test_bone_density = process_data(test_df)

# Define the custom PyTorch model
class MultiInputModel(nn.Module):
    def __init__(self, num_tabular_features, outputs):
        super(MultiInputModel, self).__init__()
        
        # Load pretrained MobileNetV2 and remove the final classification layer
        self.feature_extractor = models.mobilenet_v2(pretrained=True).features

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Batch normalization layer for image features
        self.image_bn = nn.BatchNorm1d(1280)  # 1280 is the number of features from MobileNetV2

        # Dense layers for tabular data
        self.tabular_dense1 = nn.Linear(num_tabular_features, 32)
        self.tabular_bn = nn.BatchNorm1d(32)

        # Combined features dimension
        combined_features_dim = 1280 * 2 + 32

        # Define output layers for multiple predictions
        self.output_layers = nn.ModuleList([
            nn.Linear(combined_features_dim, 1) for _ in outputs
        ])

    def process_image(self, x):
        x = self.feature_extractor(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.image_bn(x)
        return x

    def forward(self, front_image_input, back_image_input, tabular_input):
        # Process images
        front_image_features = self.process_image(front_image_input)
        back_image_features = self.process_image(back_image_input)
        
        # Process tabular data
        tabular_features = F.relu(self.tabular_dense1(tabular_input))
        tabular_features = self.tabular_bn(tabular_features)
        
        # Combine all features
        combined_features = torch.cat([front_image_features, back_image_features, tabular_features], dim=1)
        
        # Generate outputs for each target
        outputs = [output_layer(combined_features) for output_layer in self.output_layers]
        
        return outputs

# Define model creation function
def create_model(num_tabular_features, outputs):
    model = MultiInputModel(num_tabular_features, outputs)
    return model

# Create a DataLoader for given datasets
def create_dataloader(X_front, X_back, X_tabular, Y, batch_size, shuffle=True):
    tensor_dataset = TensorDataset(X_front, X_back, X_tabular, *Y)
    dataloader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

# Function to train and validate the model for one fold
def train_fold(fold, train_index, val_index, X_front, X_back, X_tabular, Y, num_tabular_features, batch_size=32, num_epochs=500):
    device = torch.device(f'cuda:{fold % torch.cuda.device_count()}' if torch.cuda.is_available() else 'cpu')
    print(f"  Training on fold {fold} on {device}")
    
    # Split the data into training and validation sets
    X_train_front, X_val_front = X_front[train_index], X_front[val_index]
    X_train_back, X_val_back = X_back[train_index], X_back[val_index]
    X_train_tabular, X_val_tabular = X_tabular[train_index], X_tabular[val_index]
    Y_train = [y[train_index] for y in Y]
    Y_val = [y[val_index] for y in Y]
    
    # Convert data to PyTorch tensors
    X_train_front = torch.tensor(X_train_front, dtype=torch.float32).permute(0, 3, 1, 2)
    X_val_front = torch.tensor(X_val_front, dtype=torch.float32).permute(0, 3, 1, 2)
    X_train_back = torch.tensor(X_train_back, dtype=torch.float32).permute(0, 3, 1, 2)
    X_val_back = torch.tensor(X_val_back, dtype=torch.float32).permute(0, 3, 1, 2)
    X_train_tabular = torch.tensor(X_train_tabular, dtype=torch.float32)
    X_val_tabular = torch.tensor(X_val_tabular, dtype=torch.float32)
    Y_train = [torch.tensor(y, dtype=torch.float32).unsqueeze(1) for y in Y_train]
    Y_val = [torch.tensor(y, dtype=torch.float32).unsqueeze(1) for y in Y_val]

    # Move tensors to device
    X_train_front, X_train_back, X_train_tabular = X_train_front.to(device), X_train_back.to(device), X_train_tabular.to(device)
    Y_train = [y.to(device) for y in Y_train]
    X_val_front, X_val_back, X_val_tabular = X_val_front.to(device), X_val_back.to(device), X_val_tabular.to(device)
    Y_val = [y.to(device) for y in Y_val]

    # Create DataLoaders for training and validation
    train_loader = create_dataloader(X_train_front, X_train_back, X_train_tabular, Y_train, batch_size)
    val_loader = create_dataloader(X_val_front, X_val_back, X_val_tabular, Y_val, batch_size, shuffle=False)

    # Create the model and move it to the device
    model = create_model(num_tabular_features, outputs=['body_fat', 'muscle_mass', 'bone_mass', 'bone_density']).to(device)
    
    # Define optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.001)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_epoch = 0
    patience = 50
    best_model_path = f'./saved/models/model_fold_{fold}.pt'
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for front, back, tabular, *targets in train_loader:
            optimizer.zero_grad()
            outputs = model(front, back, tabular)
            loss = sum(criterion(outputs[i], targets[i]) for i in range(len(outputs)))
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * front.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for front, back, tabular, *targets in val_loader:
                outputs = model(front, back, tabular)
                loss = sum(criterion(outputs[i], targets[i]) for i in range(len(outputs)))
                val_loss += loss.item() * front.size(0)
        
        val_loss /= len(val_loader.dataset)
        
        print(f'Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}')
        
        # Check for early stopping and save the best model
        if val_loss < best_val_loss:
            print(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saving model.")
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)
        
        elif epoch - best_epoch >= patience:
            print("Early stopping due to no improvement in validation loss.")
            break

    # Clean up
    del model
    torch.cuda.empty_cache()
    gc.collect()

    return best_val_loss

def train_ensemble_cv(n_models, X_front, X_back, X_tabular, Y, 
                      num_tabular_features, n_splits):
    histories = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=torch.cuda.device_count()) as executor:
        futures = []
        
        for i in range(n_models):
            print(f"Training model {i+1}/{n_models}")

            for fold, (train_index, val_index) in enumerate(kf.split(X_tabular)):
                # Submit the training of each fold to the thread pool
                futures.append(executor.submit(train_fold, fold, train_index, val_index,
                                               X_front, X_back, X_tabular, Y, num_tabular_features))

        for future in concurrent.futures.as_completed(futures):
            histories.append(future.result())

    return histories

# Define ensemble prediction function
def ensemble_predict(ground_truth, X_front, X_back, X_tabular, num_tabular_features, outputs=['body_fat', 'muscle_mass', 'bone_mass', 'bone_density']):
    predictions = []
    weight_sum = 0

    model_paths = os.listdir('./saved/models/')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert data to PyTorch tensors
    X_front = torch.tensor(X_front, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
    X_back = torch.tensor(X_back, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
    X_tabular = torch.tensor(X_tabular, dtype=torch.float32).to(device)
    ground_truth = [torch.tensor(gt, dtype=torch.float32).to(device) for gt in ground_truth]
    
    for i, model_path in enumerate(model_paths):
        # Load the model
        model = create_model(num_tabular_features, outputs)
        model.load_state_dict(torch.load(os.path.join('./saved/models/', model_path)))
        model.to(device)
        model.eval()

        with torch.no_grad():
            # Make predictions
            output_preds = model(X_front, X_back, X_tabular)
        
        # Calculate MAE for each target and combine them
        mae = sum(mean_absolute_error(ground_truth[i].cpu().numpy(), output_preds[i].cpu().numpy()) for i in range(len(outputs)))
        print(f"  Model {i+1}/{len(model_paths)} MAE: {round(mae, 2)}")
        
        # Calculate weight and weighted prediction
        weight = 1 / (mae ** 2 + 1e-8)
        weighted_preds = [output_preds[i] * weight for i in range(len(output_preds))]
        weight_sum += weight

        predictions.append([wp.cpu().numpy() for wp in weighted_preds])

    # Calculate weighted average prediction
    weighted_avg_predictions = [np.sum([p[i] for p in predictions], axis=0) / weight_sum for i in range(len(outputs))]
    return [np.round(wp, 1) for wp in weighted_avg_predictions]

# Main execution
num_tabular_features = tabular.shape[1] 
n_models = 1
n_splits = 4

# Delete every past file in saved/models
folder_path = './saved/models/'
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    os.remove(file_path)

Y = [body_fat, muscle_mass, bone_mass, bone_density]

histories = train_ensemble_cv(n_models, front_images, back_images, tabular, Y, num_tabular_features, n_splits)

predictions_body_fat, predictions_muscle_mass, predictions_bone_mass, predictions_bone_density = ensemble_predict(
    [Y_test_body_fat, Y_test_muscle_mass, Y_test_bone_mass, Y_test_bone_density], 
    X_test_front_images, 
    X_test_back_images, 
    X_test_tabular, 
    num_tabular_features=num_tabular_features,  
    outputs=['body_fat', 'muscle_mass', 'bone_mass', 'bone_density'] 
)

predictions = {
    'body_fat': predictions_body_fat,
    'muscle_mass': predictions_muscle_mass,
    'bone_mass': predictions_bone_mass,
    'bone_density': predictions_bone_density
}

ground_truth = {
    'body_fat': Y_test_body_fat,
    'muscle_mass': Y_test_muscle_mass,
    'bone_mass': Y_test_bone_mass,
    'bone_density': Y_test_bone_density
}

for key in predictions.keys():
    mae = mean_absolute_error(ground_truth[key], predictions[key])
    print(f"Final MAE on test set for {key}: {mae}")

def calculate_variance(ground_truth, predictions):
    variances = {}
    
    for metric in ground_truth:
        # Reshape predictions to match ground truth dimensions
        pred_values = predictions[metric].reshape(-1)
        true_values = ground_truth[metric]
        
        variance = np.sum(np.abs(true_values - pred_values) / true_values) / len(true_values)
        variances[metric] = variance
    
    return variances

variances = calculate_variance(ground_truth, predictions)
print(variances)