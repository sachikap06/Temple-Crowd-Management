import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import os
from lstm_model import CrowdLSTM, MODEL_PATH, SEQ_LENGTH, MAX_COUNT

# ==========================================
# EFFICIENT BATCHED LSTM TRAINING SCRIPT
# ==========================================

def generate_temple_sequences(num_samples=1200):
    """
    Generates diverse crowd arrivals across various crowd levels (0 to 45),
    including steady low-density crowd periods (0 to 3 people).
    """
    np.random.seed(42)
    t = np.linspace(0, 10 * np.pi, num_samples)
    base = 15 + 12 * np.sin(t) + 6 * np.cos(2.5 * t) + 4 * np.sin(0.5 * t)
    noise = np.random.normal(0, 1.0, num_samples)
    counts = np.clip(base + noise, 0, 48)

    # Add realistic low-density steady periods (0, 1, 2, 3 people)
    low_counts = [0, 1, 2, 3]
    for i in range(0, num_samples - 20, 40):
        target_val = float(low_counts[(i // 40) % len(low_counts)])
        counts[i : i + 20] = target_val

    return counts.astype(np.float32)


def prepare_dataset(data, seq_length=SEQ_LENGTH, horizon=5):
    X, y = [], []
    scaled = data / MAX_COUNT
    for i in range(len(scaled) - seq_length - horizon):
        X.append(scaled[i : i + seq_length])
        y.append(scaled[i + seq_length + horizon - 1])
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    X_tensor = torch.tensor(X).unsqueeze(-1)  # (N, SEQ_LENGTH, 1)
    y_tensor = torch.tensor(y).unsqueeze(-1)  # (N, 1)
    return X_tensor, y_tensor


def train_model(epochs=80, batch_size=32, lr=0.003):
    print("Preparing training data...")
    synthetic = generate_temple_sequences(800)
    
    csv_file = "output/crowd_data.csv"
    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file)
            if len(df) > 5:
                csv_counts = df["People_Count"].values.astype(np.float32)
                raw_counts = np.concatenate([synthetic, csv_counts])
            else:
                raw_counts = synthetic
        except Exception:
            raw_counts = synthetic
    else:
        raw_counts = synthetic

    X_train, y_train = prepare_dataset(raw_counts)
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Dataset shape: X={X_train.shape}, y={y_train.shape}")
    
    model = CrowdLSTM()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"Training PyTorch LSTM for {epochs} epochs...")
    model.train()
    
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for bx, by in loader:
            optimizer.zero_grad()
            outputs = model(bx)
            loss = criterion(outputs, by)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * bx.size(0)
            
        epoch_loss = running_loss / len(X_train)
        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch [{epoch}/{epochs}] - Loss: {epoch_loss:.6f}")

    os.makedirs("output", exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"[OK] Trained LSTM model successfully saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train_model(epochs=80)
