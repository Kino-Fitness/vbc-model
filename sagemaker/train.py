import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from model import MultiInputModel, OUTPUT_METRICS

# Load data
def load_data():
    df = pd.read_pickle('saved/dataframes/df1.pkl')
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    return train_df, test_df

def train_model(train_df, num_epochs=10, batch_size=16):
    model = MultiInputModel(num_tabular_features=32, outputs=OUTPUT_METRICS)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    model.train()

    # Example training loop
    for epoch in range(num_epochs):
        for front, back, tabular, target in DataLoader(train_df, batch_size=batch_size):
            optimizer.zero_grad()
            outputs = model(front, back, tabular)
            loss = sum(criterion(outputs[i], target[i]) for i in range(len(outputs)))
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), 'saved/model.pth')

if __name__ == "__main__":
    train_df, _ = load_data()
    train_model(train_df)
