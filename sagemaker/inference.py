import torch
import torch.nn as nn
from model import MultiInputModel, OUTPUT_METRICS
import pandas as pd
import numpy as np

def model_fn(model_dir):
    # Load the model
    model = MultiInputModel(num_tabular_features=32, outputs=OUTPUT_METRICS)
    model.load_state_dict(torch.load(f"{model_dir}/model.pth", map_location='cpu'))
    model.eval()
    return model

def input_fn(request_body, content_type):
    if content_type == "application/json":
        data = pd.DataFrame.from_dict(request_body)
        return torch.tensor(data.values, dtype=torch.float32)
    raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    with torch.no_grad():
        predictions = model(input_data, input_data, input_data)  # Dummy input for demo
    return predictions
