import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold,GroupKFold, train_test_split
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
df = pd.read_pickle('saved/dataframes/df1.pkl')
df = df[df['Gender'] == 'Male']
subject_indices = df.index.values
train_indices, test_indices = next(GroupKFold(n_splits=5).split(df, groups=subject_indices))
train_df = df.iloc[train_indices]
test_df = df.iloc[test_indices]
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Process training data
front_images, back_images, tabular_basic, tabular_full, body_fat, muscle_mass, bone_mass, bone_density = process_data(train_df)
# Process test data
X_test_front_images, X_test_back_images, X_test_tabular_basic, X_test_tabular_full, Y_test_body_fat, Y_test_muscle_mass, Y_test_bone_mass, Y_test_bone_density = process_data(test_df)
class MultiInputModel(nn.Module):
    def __init__(self, num_tabular_features_basic, num_tabular_features_full, outputs):
        super(MultiInputModel, self).__init__()
        
        # Load pretrained MobileNetV2 and remove the final classification layer
        self.feature_extractor = models.mobilenet_v2(pretrained=True).features
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.image_bn = nn.BatchNorm1d(1280, momentum=0.01, eps=1e-3)

        # Separate dense layers for basic and full tabular data
        self.tabular_basic_dense = nn.Linear(num_tabular_features_basic, 32)
        self.tabular_basic_bn = nn.BatchNorm1d(32, momentum=0.01, eps=1e-3)
        
        self.tabular_full_dense = nn.Linear(num_tabular_features_full, 32)
        self.tabular_full_bn = nn.BatchNorm1d(32, momentum=0.01, eps=1e-3)

        # Combined features dimensions
        combined_features_dim = 1280 * 2 + 32  # Same size for both paths

        # Separate output layers for body fat vs other metrics
        self.body_fat_output = nn.Linear(combined_features_dim, 1)
        self.other_outputs = nn.ModuleList([nn.Linear(combined_features_dim, 1) for _ in range(len(outputs)-1)])
    
    def process_image(self, x):
        x = self.feature_extractor(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        if x.size(0) > 1:
            x = self.image_bn(x)
        return x

    def forward(self, front_image_input, back_image_input, tabular_basic_input, tabular_full_input):
        # Process images
        front_image_features = self.process_image(front_image_input)
        back_image_features = self.process_image(back_image_input)
        
        # Process basic tabular data (for body fat)
        tabular_basic_features = F.relu(self.tabular_basic_dense(tabular_basic_input))
        if tabular_basic_input.size(0) > 1:
            tabular_basic_features = self.tabular_basic_bn(tabular_basic_features)
            
        # Process full tabular data (for other metrics)
        tabular_full_features = F.relu(self.tabular_full_dense(tabular_full_input))
        if tabular_full_input.size(0) > 1:
            tabular_full_features = self.tabular_full_bn(tabular_full_features)
        
        # Create combined features for body fat
        combined_basic = torch.cat([front_image_features, back_image_features, tabular_basic_features], dim=1)
        # Create combined features for other metrics
        combined_full = torch.cat([front_image_features, back_image_features, tabular_full_features], dim=1)
        
        # Generate body fat prediction using basic features
        body_fat_output = self.body_fat_output(combined_basic)
        
        # Generate other predictions using full features
        other_outputs = [layer(combined_full) for layer in self.other_outputs]
        
        # Combine all outputs
        return [body_fat_output] + other_outputs
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
def create_model(num_tabular_features, num_tabular_features_full):
    model = MultiInputModel(num_tabular_features,num_tabular_features_full, outputs=OUTPUT_METRICS)
    return model

# Create a DataLoader for given datasets
def create_dataloader(X_front, X_back, X_tabular_basic, X_tabular_full, Y, batch_size, shuffle=True, drop_last=False):
    tensor_dataset = TensorDataset(X_front, X_back, X_tabular_basic, X_tabular_full, *Y)
    dataloader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return dataloader


def train_fold(fold, train_index, val_index, X_front, X_back, X_tabular_basic, X_tabular_full, Y, 
               num_tabular_features_basic, num_tabular_features_full, batch_size=32, num_epochs=1000):
    device = torch.device(f'cuda:{fold % torch.cuda.device_count()}' if torch.cuda.is_available() else 'cpu')
    print(f"Training on fold {fold} on {device}")
    
    # Split data for both basic and full features
    X_train_front, X_val_front = X_front[train_index], X_front[val_index]
    X_train_back, X_val_back = X_back[train_index], X_back[val_index]
    X_train_tabular_basic, X_val_tabular_basic = X_tabular_basic[train_index], X_tabular_basic[val_index]
    X_train_tabular_full, X_val_tabular_full = X_tabular_full[train_index], X_tabular_full[val_index]
    Y_train = [y[train_index] for y in Y]
    Y_val = [y[val_index] for y in Y]
    
    # Convert to tensors
    X_train_front = torch.tensor(X_train_front, dtype=torch.float32).permute(0, 3, 1, 2)
    X_val_front = torch.tensor(X_val_front, dtype=torch.float32).permute(0, 3, 1, 2)
    X_train_back = torch.tensor(X_train_back, dtype=torch.float32).permute(0, 3, 1, 2)
    X_val_back = torch.tensor(X_val_back, dtype=torch.float32).permute(0, 3, 1, 2)
    X_train_tabular_basic = torch.tensor(X_train_tabular_basic, dtype=torch.float32)
    X_val_tabular_basic = torch.tensor(X_val_tabular_basic, dtype=torch.float32)
    X_train_tabular_full = torch.tensor(X_train_tabular_full, dtype=torch.float32)
    X_val_tabular_full = torch.tensor(X_val_tabular_full, dtype=torch.float32)
    Y_train = [torch.tensor(y, dtype=torch.float32).unsqueeze(1) for y in Y_train]
    Y_val = [torch.tensor(y, dtype=torch.float32).unsqueeze(1) for y in Y_val]

    # Move to device
    X_train_front, X_train_back = X_train_front.to(device), X_train_back.to(device)
    X_train_tabular_basic = X_train_tabular_basic.to(device)
    X_train_tabular_full = X_train_tabular_full.to(device)
    Y_train = [y.to(device) for y in Y_train]
    
    X_val_front, X_val_back = X_val_front.to(device), X_val_back.to(device)
    X_val_tabular_basic = X_val_tabular_basic.to(device)
    X_val_tabular_full = X_val_tabular_full.to(device)
    Y_val = [y.to(device) for y in Y_val]

    # Create dataloaders
    train_loader = create_dataloader(X_train_front, X_train_back, X_train_tabular_basic, 
                                   X_train_tabular_full, Y_train, batch_size, drop_last=True)
    val_loader = create_dataloader(X_val_front, X_val_back, X_val_tabular_basic, 
                                 X_val_tabular_full, Y_val, batch_size, shuffle=False, drop_last=True)
    
    if len(val_loader) == 0:
        print(f"Warning: Validation set too small for batch size {batch_size}")
        new_batch_size = len(X_val_front) // 2
        val_loader = create_dataloader(X_val_front, X_val_back, X_val_tabular_basic, 
                                     X_val_tabular_full, Y_val, new_batch_size, shuffle=False, drop_last=True)
        if len(val_loader) == 0:
            print("Error: Validation set too small even with adjusted batch size")
            return None

    # Create model with both feature sets
    model = create_model(num_tabular_features_basic, num_tabular_features_full).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_epoch = 0
    patience = 50
    best_model_path = f'./saved/models/model_fold_{fold}.pt'
    
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        num_train_samples = 0
        
        for front, back, tabular_basic, tabular_full, *targets in train_loader:
            if front.size(0) == 1:
                continue
                
            optimizer.zero_grad()
            outputs = model(front, back, tabular_basic, tabular_full)
            loss = sum(criterion(outputs[i], targets[i]) for i in range(len(outputs)))
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item() * front.size(0)
            num_train_samples += front.size(0)
        
        if num_train_samples == 0:
            print(f"Warning: No valid training batches in epoch {epoch + 1}")
            continue
        
        train_loss /= num_train_samples
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0.0
        num_val_samples = 0
        
        with torch.no_grad():
            for front, back, tabular_basic, tabular_full, *targets in val_loader:
                if front.size(0) == 1:
                    continue
                outputs = model(front, back, tabular_basic, tabular_full)
                loss = sum(criterion(outputs[i], targets[i]) for i in range(len(outputs)))
                val_loss += loss.item() * front.size(0)
                num_val_samples += front.size(0)
        
        if num_val_samples == 0:
            print(f"Warning: No valid validation batches in epoch {epoch + 1}")
            continue
        
        val_loss /= num_val_samples
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - LR: {scheduler.get_last_lr()[0]:.6f}')
        
        if val_loss < best_val_loss:
            print(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}")
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)
        elif epoch - best_epoch >= patience:
            print("Early stopping triggered")
            break

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return train_losses, val_losses

def train_cv(X_front, X_back, X_tabular_basic, X_tabular_full, Y, subject_indices, num_tabular_features_basic, num_tabular_features_full, n_splits):
    histories = []
    group_kf = GroupKFold(n_splits=n_splits)
    max_workers = max(torch.cuda.device_count(), 1)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []

        # Use GroupKFold with indices as groups to ensure subjects stay together
        for fold, (train_index, val_index) in enumerate(group_kf.split(X_tabular_basic, groups=subject_indices)):
            futures.append(
                executor.submit(
                    train_fold, fold, train_index, val_index, 
                    X_front, X_back, X_tabular_basic, X_tabular_full, Y,
                    num_tabular_features_basic, num_tabular_features_full
                )
            )

        for future in concurrent.futures.as_completed(futures):
            histories.append(future.result())

    return histories

# Main execution
train_subject_indices = train_df.index.values
tabular_basic = np.array(tabular_basic)
tabular_full = np.array(tabular_full)

num_tabular_features_basic = tabular_basic.shape[1]
num_tabular_features_full = tabular_full.shape[1]

# Clear existing models
folder_path = './saved/models/'
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    os.remove(file_path)

# Prepare targets
Y = [body_fat, muscle_mass, bone_mass, bone_density]

# Train with cross-validation
histories = train_cv(front_images, back_images, tabular_basic, tabular_full, Y, 
                    train_subject_indices, num_tabular_features_basic, 
                    num_tabular_features_full, n_splits=5)

# Predictions
model_paths = os.listdir('./saved/models/')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Convert test data
X_front = torch.tensor(X_test_front_images, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
X_back = torch.tensor(X_test_back_images, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
X_tabular_basic = torch.tensor(X_test_tabular_basic, dtype=torch.float32).to(device)
X_tabular_full = torch.tensor(X_test_tabular_full, dtype=torch.float32).to(device)
ground_truth = [torch.tensor(gt, dtype=torch.float32).to(device) for gt in [Y_test_body_fat, Y_test_muscle_mass, Y_test_bone_mass, Y_test_bone_density]]

preds = []

for i, model_path in enumerate(model_paths):
    model = create_model(num_tabular_features_basic, num_tabular_features_full)
    model.load_state_dict(torch.load(os.path.join('./saved/models/', model_path)))
    model.to(device)
    model.eval()

    with torch.no_grad():
        preds.append([t.squeeze() for t in model(X_front, X_back, X_tabular_basic, X_tabular_full)])

weights = torch.tensor([1] * len(model_paths), device=device)
weighted_preds = []

for i in range(len(preds[0])):
    weighted_sum = sum(w * lst[i] for w, lst in zip(weights, preds))
    weighted_avg_pred = weighted_sum / weights.sum()
    weighted_preds.append(weighted_avg_pred)

# Calculate metrics
metrics_data = []
feature_results = {}
for pred, act, metric in zip(weighted_preds, ground_truth, OUTPUT_METRICS):
    mae = torch.mean(torch.abs(pred - act))
    variance = torch.sum(torch.abs(act - pred) / act) / len(act)
    ci = calculate_confidence_intervals(pred.cpu(), act.cpu())
    print(f"{metric}, MAE: {round(mae.item(), 3)} Variance: {round(variance.item(), 3)}")
    feature_results[metric] = {'mae': mae.item(), 'variance': variance.item(), 'ci': ci}

# Format output data
output_data = []
for metric in OUTPUT_METRICS:
    metric_data = {
        "metric": metric,
        "mae": feature_results[metric]['mae'],
        "variance": feature_results[metric]['variance'],
        "confidence_interval": {
            "mean": feature_results[metric]['ci']['mean'],
            "lower_bound": feature_results[metric]['ci']['lower_bound'],
            "upper_bound": feature_results[metric]['ci']['upper_bound'],
            "error_margins": {
                "lower": feature_results[metric]['ci']['mean'] - feature_results[metric]['ci']['lower_bound'],
                "upper": feature_results[metric]['ci']['upper_bound'] - feature_results[metric]['ci']['mean']
            }
        }
    }
    output_data.append(metric_data)
# After making predictions, collect the body fat data with IDs
test_subject_indices = test_df.index.values

# The first element of weighted_preds corresponds to body fat predictions
body_fat_predictions = weighted_preds[0].cpu().numpy()
actual_body_fat = Y_test_body_fat

# Create individual predictions data
individual_predictions = []
for idx, (subject_id, predicted, actual) in enumerate(zip(test_subject_indices, body_fat_predictions, actual_body_fat)):
    prediction_data = {
        "subject_id": int(subject_id),
        "estimated_body_fat": float(predicted),
        "actual_body_fat": float(actual)
    }
    individual_predictions.append(prediction_data)

# Update the output data structure to include both summary metrics and individual predictions
print(output_data)
import json
def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
final_output = {
    "summary_metrics": output_data,
    "individual_predictions": individual_predictions
}

# Save to JSON
with open('metrics_data.json', 'w') as f:
    json.dump(final_output, f, indent=2, default=convert_numpy_types)



# Plot Loss
'''plt.figure(figsize=(10, 6))

for i, (train_loss, val_loss) in enumerate(histories):
    plt.plot(train_loss, label=f'Train Loss Fold {i+1}')
    plt.plot(val_loss, label=f'Val Loss Fold {i+1}')

plt.title('Training and Validation Loss Across Folds')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

for m in metrics_data:
    print(f"Metric: {m['metric']}")
    print(f"Mean: {m['ci']['mean']}")
    print(f"Lower: {m['ci']['lower_bound']}")
    print(f"Upper: {m['ci']['upper_bound']}")
# Plot confidence intervals
plt.figure(figsize=(12, 6))

plt.errorbar(
    x=range(len(metrics_data)),
    y=[float(m['ci']['mean']) for m in metrics_data],
    yerr=[[float(m['ci']['mean'] - m['ci']['lower_bound']) for m in metrics_data],
          [float(m['ci']['upper_bound'] - m['ci']['mean']) for m in metrics_data]],
    fmt='o',
    capsize=5,
    capthick=2,
    elinewidth=2,
    markersize=8
)

plt.margins(y=0.2)

plt.title('Prediction Errors with 95% Confidence Intervals')
plt.xticks(range(len(OUTPUT_METRICS)), OUTPUT_METRICS, rotation=45)
plt.ylabel('Mean Absolute Error')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
'''