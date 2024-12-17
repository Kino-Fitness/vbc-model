import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats
import os
import json
from utils import process_data  

# Define the metrics as a global constant
OUTPUT_METRICS = ['body_fat', 'muscle_mass', 'bone_mass', 'bone_density']

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
    
def create_model(num_tabular_features, num_tabular_features_full):
    model = MultiInputModel(num_tabular_features,num_tabular_features_full, outputs=OUTPUT_METRICS)
    return model

def calculate_confidence_intervals(predictions, ground_truth, confidence_level=0.95):
    """Calculate confidence intervals for predictions"""
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

    return {
        'mean': X_bar,
        'lower_bound': X_bar - margin_of_error,
        'upper_bound': X_bar + margin_of_error,
        'std_dev': S,
        'sample_size': n,
        'z_value': z_value,
        'margin_of_error': margin_of_error
    }
df = pd.read_pickle('saved/dataframes/df2.pkl')
df = df[df['Gender'] == 'Male']
X_test_front_images, X_test_back_images, X_test_tabular_basic, X_test_tabular_full, Y_test_body_fat, Y_test_muscle_mass, Y_test_bone_mass, Y_test_bone_density = process_data(df)

X_test_tabular_basic = np.array(X_test_tabular_basic)
X_test_tabular_full = np.array(X_test_tabular_full)

num_tabular_features_basic = X_test_tabular_basic.shape[1]
num_tabular_features_full = X_test_tabular_full.shape[1]

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


print("\nIndividual Predictions:")
print("Index | Estimated Body Fat | Actual Body Fat")
print("-" * 50)

indices = df.index.values
body_fat_predictions = weighted_preds[0].cpu().numpy()
actual_body_fat = Y_test_body_fat

for idx, pred, actual in zip(indices, body_fat_predictions, actual_body_fat):
    print(f"{idx} | {pred:.2f}% | {actual:.2f}%")

    # Calculate metrics
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
with open('metrics_dataProduction.json', 'w') as f:
    json.dump(output_data, f, indent=2, default=convert_numpy_types)
