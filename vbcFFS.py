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
from scipy import stats

# Define the metrics as a global constant
OUTPUT_METRICS = ['body_fat', 'muscle_mass', 'bone_mass', 'bone_density']

# Settings
pd.set_option('display.max_rows', None)
np.set_printoptions(threshold=100)

# Load data
df = pd.read_pickle('saved/dataframes/df.pkl')
df = df[df['Gender'] == 'Male']
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

front_images, back_images, tabular, body_fat, muscle_mass, bone_mass, bone_density = process_data(train_df)
X_test_front_images, X_test_back_images, X_test_tabular, Y_test_body_fat, Y_test_muscle_mass, Y_test_bone_mass, Y_test_bone_density = process_data(test_df)
class MultiInputModel(nn.Module):
    def __init__(self, num_tabular_features, outputs):
        super(MultiInputModel, self).__init__()
        
        # Load pretrained MobileNetV2 and remove the final classification layer
        self.feature_extractor = models.mobilenet_v2(pretrained=True).features

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Batch normalization layer for image features
        self.image_bn = nn.BatchNorm1d(1280, momentum=0.01, eps=1e-3)  # Modified momentum and eps

        # Dense layers for tabular data
        self.tabular_dense1 = nn.Linear(num_tabular_features, 32)
        self.tabular_bn = nn.BatchNorm1d(32, momentum=0.01, eps=1e-3)  # Modified momentum and eps

        # Combined features dimension
        combined_features_dim = 1280 * 2 + 32

        # Define output layers for multiple predictions
        self.output_layers = nn.ModuleList([nn.Linear(combined_features_dim, 1) for _ in outputs])
    
    
    def process_image(self, x):
        x = self.feature_extractor(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        if x.size(0) > 1:  # Only apply batch norm if batch size > 1
            x = self.image_bn(x)
        return x

    def forward(self, front_image_input, back_image_input, tabular_input):
        # Process images
        front_image_features = self.process_image(front_image_input)
        back_image_features = self.process_image(back_image_input)
        
        # Process tabular data
        tabular_features = F.relu(self.tabular_dense1(tabular_input))
        if tabular_input.size(0) > 1:  # Only apply batch norm if batch size > 1
            tabular_features = self.tabular_bn(tabular_features)
        
        # Combine all features
        combined_features = torch.cat([front_image_features, back_image_features, tabular_features], dim=1)
        
        # Generate outputs for each target
        outputs = [output_layer(combined_features) for output_layer in self.output_layers]
        
        return outputs
def calculate_confidence_intervals(predictions, ground_truth, confidence_level=0.95):
        if torch.is_tensor(predictions):
            predictions = predictions.cpu().numpy()
        if torch.is_tensor(ground_truth):
            ground_truth = ground_truth.cpu().numpy()
    
        errors = np.abs(predictions - ground_truth)
    
        X_bar = np.mean(errors)               
        S = np.std(errors, ddof=1)            
        n = len(errors)                        

        z_value = stats.norm.ppf((1 + confidence_level) / 2)
    
        margin_of_error = z_value * (S / np.sqrt(n))
    
        lower_bound = X_bar - margin_of_error
        upper_bound = X_bar + margin_of_error
    
        return {
            'mean': X_bar,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'std_dev': S,
            'sample_size': n,
            'z_value': z_value,
            'margin_of_error': margin_of_error
        }
# Define model creation function
def create_model(num_tabular_features):
    model = MultiInputModel(num_tabular_features, outputs=OUTPUT_METRICS)
    return model

# Create a DataLoader for given datasets
def create_dataloader(X_front, X_back, X_tabular, Y, batch_size, shuffle=True, drop_last=False):
    tensor_dataset = TensorDataset(X_front, X_back, X_tabular, *Y)
    dataloader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return dataloader

def train_fold(fold, train_index, val_index, X_front, X_back, X_tabular, Y, num_tabular_features, batch_size=32, num_epochs=8):
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
    train_loader = create_dataloader(X_train_front, X_train_back, X_train_tabular, Y_train, batch_size, drop_last=True)
    val_loader = create_dataloader(X_val_front, X_val_back, X_val_tabular, Y_val, batch_size, shuffle=False, drop_last=True)

    # Check if we have enough samples for validation
    if len(val_loader) == 0:
        print(f"Warning: Validation set is too small for the current batch size ({batch_size}). Adjusting batch size.")
        new_batch_size = len(X_val_front) // 2  # Use half of the validation set size as the new batch size
        val_loader = create_dataloader(X_val_front, X_val_back, X_val_tabular, Y_val, new_batch_size, shuffle=False, drop_last=True)
        if len(val_loader) == 0:
            print(f"Error: Validation set is too small even with adjusted batch size. Cannot proceed with training.")
            return None

    # Create the model and move it to the device
    model = create_model(num_tabular_features).to(device)
    
    # Define optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.001)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_epoch = 0
    patience = 50
    best_model_path = f'./saved/models/model_fold_{fold}.pt'
    
    # Lists to store training and validation losses
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        num_train_samples = 0
        for front, back, tabular, *targets in train_loader:
            if front.size(0) == 1:  # Skip batches with only one sample
                continue
            optimizer.zero_grad()
            outputs = model(front, back, tabular)
            loss = sum(criterion(outputs[i], targets[i]) for i in range(len(outputs)))
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * front.size(0)
            num_train_samples += front.size(0)
        
        if num_train_samples == 0:
            print(f"Warning: No valid training batches in epoch {epoch + 1}. Skipping this epoch.")
            continue
        
        train_loss /= num_train_samples
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        num_val_samples = 0
        with torch.no_grad():
            for front, back, tabular, *targets in val_loader:
                if front.size(0) == 1:  # Skip batches with only one sample
                    continue
                outputs = model(front, back, tabular)
                loss = sum(criterion(outputs[i], targets[i]) for i in range(len(outputs)))
                val_loss += loss.item() * front.size(0)
                num_val_samples += front.size(0)
        
        if num_val_samples == 0:
            print(f"Warning: No valid validation batches in epoch {epoch + 1}. Skipping validation for this epoch.")
            continue
        
        val_loss /= num_val_samples
        val_losses.append(val_loss)
        
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

    return train_losses, val_losses

def train_cv(X_front, X_back, X_tabular, Y, num_tabular_features, n_splits):
    histories = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    max_workers = max(torch.cuda.device_count(), 1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []

        for fold, (train_index, val_index) in enumerate(kf.split(X_tabular)):
            futures.append(executor.submit(train_fold, fold, train_index, val_index, X_front, X_back, X_tabular, Y, num_tabular_features))

        for future in concurrent.futures.as_completed(futures):
            histories.append(future.result())

    return histories
def process_features(df, feature_combination):
    """Process different feature combinations"""
    features = []
    
    if 'waist/hips' in feature_combination:
        features.append(df['Waist'] / df['Hip (bone)'])
    if 'waist & hips' in feature_combination:
        features.extend([df['Waist'], df['Hip (bone)']])
    if 'height' in feature_combination:
        features.append(df['Height'])
    if 'weight' in feature_combination:
        features.append(df['Weight'])
    
    return np.column_stack(features)
# Main execution
FEATURE_COMBINATIONS = [
    ['waist/hips'],
    ['waist & hips'],
    ['height'],
    ['weight'],
    ['height', 'weight'],
    ['height', 'weight', 'waist & hips'],
    ['height', 'weight', 'waist/hips']
]
n_splits = 2
results_all = {}
for feature_combo in FEATURE_COMBINATIONS:
    print(f"\nTraining with features: {feature_combo}")
    
    tabular = process_features(train_df, feature_combo)
    X_test_tabular = process_features(test_df, feature_combo)
    num_tabular_features = tabular.shape[1]
    
    folder_path = './saved/models/'
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        os.remove(file_path)

    Y = [body_fat, muscle_mass, bone_mass, bone_density]

    histories = train_cv(front_images, back_images, tabular, Y, num_tabular_features, n_splits)

    # Predictions
    model_paths = os.listdir('./saved/models/')
    device = torch.device('cpu')

    X_front = torch.tensor(X_test_front_images, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
    X_back = torch.tensor(X_test_back_images, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
    X_tabular = torch.tensor(X_test_tabular, dtype=torch.float32).to(device)
    ground_truth = [torch.tensor(gt, dtype=torch.float32).to(device) for gt in [Y_test_body_fat, Y_test_muscle_mass, Y_test_bone_mass, Y_test_bone_density]]

    preds = []
    for i, model_path in enumerate(model_paths):
        model = create_model(num_tabular_features)
        model.load_state_dict(torch.load(os.path.join('./saved/models/', model_path)))
        model.to(device)
        model.eval()

        with torch.no_grad():
            preds.append([t.squeeze() for t in model(X_front, X_back, X_tabular)])

    weights = torch.tensor([1] * 8)  
    weighted_preds = []

    for i in range(len(preds[0])):
        weighted_sum = sum(w * lst[i] for w, lst in zip(weights, preds))
        weighted_avg_pred = weighted_sum / weights.sum()
        weighted_preds.append(weighted_avg_pred)

    metrics_data = []
    feature_results = {}
    for pred, act, metric in zip(weighted_preds, ground_truth, OUTPUT_METRICS):
        mae = torch.mean(torch.abs(pred - act))
        variance = torch.sum(torch.abs(act - pred) / act) / len(act)
        ci = calculate_confidence_intervals(pred.cpu(), act.cpu())
        print(f"{metric} for {feature_combo}, MAE: {round(mae.item(), 3)} Variance: {round(variance.item(), 3)}")
        feature_results[metric] = {'mae': mae.item(), 'variance': variance.item(), 'ci': ci}
    
    results_all[str(feature_combo)] = feature_results

# After feature combination loop, plot overall results
plt.figure(figsize=(15, 10))
feature_names = list(results_all.keys())

for idx, metric in enumerate(OUTPUT_METRICS):
    plt.subplot(2, 2, idx+1)
    mae_values = [results_all[feat][metric]['mae'] for feat in feature_names]
    ci_values = [results_all[feat][metric]['ci'] for feat in feature_names]
    
    plt.errorbar(
        range(len(feature_names)),
        mae_values,
        yerr=[[v['mean'] - v['lower_bound'] for v in ci_values],
              [v['upper_bound'] - v['mean'] for v in ci_values]],
        fmt='o',
        capsize=5
    )
    
    plt.title(f'{metric} by Feature Combination')
    plt.xticks(range(len(feature_names)), feature_names, rotation=45)
    plt.ylabel('MAE')
    plt.grid(True)

plt.tight_layout()
plt.show()