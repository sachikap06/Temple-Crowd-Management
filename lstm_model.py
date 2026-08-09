import torch
import torch.nn as nn
import numpy as np
import os

# ==========================================
# PYTORCH LSTM CROWD PREDICTION MODEL
# ==========================================

class CrowdLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=2, output_size=1):
        super(CrowdLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0
        )
        
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# ==========================================
# PREPROCESSING & HELPER UTILITIES
# ==========================================

MODEL_PATH = "output/lstm_model.pth"
SEQ_LENGTH = 10
MAX_COUNT = 50.0  # Normalization scaling factor


def scale_data(data):
    """Normalize crowd count array to [0, 1] range."""
    return np.array(data, dtype=np.float32) / MAX_COUNT


def inverse_scale_data(scaled_data):
    """Convert normalized network output back to actual crowd count."""
    return np.clip(np.round(scaled_data * MAX_COUNT), 0, 100).astype(int)


def load_trained_model():
    """Load pre-trained PyTorch LSTM model weights."""
    model = CrowdLSTM()
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
            model.eval()
            return model
        except Exception as e:
            print(f"Error loading model weights: {e}")
    
    model.eval()
    return model


def predict_future_crowd(history_counts, model=None, forecast_horizon=10):
    """
    Given a sequence of past people counts, predict future crowd count 10 seconds ahead
    combining PyTorch LSTM neural output with short-term trend dynamics.
    """
    if len(history_counts) == 0:
        return 0

    current_val = float(history_counts[-1])

    # Pad sequence if history is shorter than SEQ_LENGTH window
    if len(history_counts) < SEQ_LENGTH:
        padded = np.pad(history_counts, (SEQ_LENGTH - len(history_counts), 0), mode='edge')
    else:
        padded = np.array(history_counts[-SEQ_LENGTH:], dtype=np.float32)

    if model is None:
        model = load_trained_model()

    scaled = scale_data(padded)
    input_tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

    with torch.no_grad():
        output = model(input_tensor)
        raw_pred = output.item() * MAX_COUNT

    # Compute sequence momentum trend
    if len(padded) >= 3:
        short_trend = (padded[-1] - padded[-3]) / 2.0
    else:
        short_trend = 0.0

    # Dynamic blend: LSTM neural pattern + live momentum anchor
    blend_pred = 0.65 * raw_pred + 0.35 * (current_val + short_trend * 1.5)
    final_count = int(np.clip(np.round(blend_pred), 0, 100))
    return final_count
