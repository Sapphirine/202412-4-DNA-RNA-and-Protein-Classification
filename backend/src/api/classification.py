import torch
from torch import nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load global variables for preprocessing and classification
with open('./src/api/pt_model/label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

with open('./src/api/pt_model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

tag_map = {0: 'DNA', 1: 'Protein', 2: 'Protein#DNA', 3: 'Protein#DNA#RNA', 4: 'Protein#RNA', 5: 'RNA'}
device = 'cpu'

class TransformerClassifier(nn.Module):
    """Transformer-based model for sequence classification."""
    def __init__(self, input_dim, num_classes, num_heads=4, num_layers=2, hidden_dim=128, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """Forward pass for Transformer model."""
        if x.dim() == 2:  # Add batch dimension if necessary
            x = x.unsqueeze(1)
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = self.fc(x[:, -1, :])  # Use the last time step for classification
        return x

class RNNClassifier(nn.Module):
    """Simple RNN model for sequence classification."""
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=1):
        super(RNNClassifier, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """Forward pass for RNN model."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

class LSTMClassifier(nn.Module):
    """LSTM-based model for sequence classification."""
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=1, dropout=0.5):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """Forward pass for LSTM model."""
        if x.dim() == 2:  # Add batch dimension if necessary
            x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class BaseCNN(nn.Module):
    """CNN-based model for sequence classification."""
    def __init__(self, input_dim, num_classes, dropout_rate=0.5):
        super(BaseCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU()

        self.flatten_dim = self._get_flatten_dim(input_dim)  # Compute flatten dimension dynamically
        self.fc1 = nn.Linear(self.flatten_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(p=dropout_rate)

    def _get_flatten_dim(self, input_dim):
        """Compute the output dimension after convolutions and pooling."""
        x = torch.zeros(1, 1, input_dim)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        return x.numel()

    def forward(self, x):
        """Forward pass for CNN model."""
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten for fully connected layers
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def preprocess_data(sample_data, iftensor):
    """Preprocess input data for model compatibility."""
    df = pd.DataFrame([sample_data])

    # Add sequence length as a feature
    df['sequence_length'] = df['sequence'].apply(len)

    # Encode categorical features
    for col in ['chainId', 'structureId']:
        if col in label_encoders:
            encoder = label_encoders[col]
            unseen_labels = set(df[col]) - set(encoder.classes_)
            if unseen_labels:
                print(f"warning: {col} unknown tag: {unseen_labels}")
                encoder.classes_ = np.append(encoder.classes_, list(unseen_labels))
            df[col] = encoder.transform(df[col])
        else:
            raise ValueError(f"compiler miss: {col}")

    # Scale numerical features
    feature_columns = ['residueCount', 'sequence_length']
    df[feature_columns] = scaler.transform(df[feature_columns])

    processed_data = df[['residueCount', 'sequence_length', 'chainId', 'structureId']]

    if iftensor:
        out = torch.tensor(processed_data.values, dtype=torch.float32)
    else:
        out = processed_data

    return out

def classify_with_model(model_name, raw_data):
    """Classify input data using the specified model."""
    model = None
    result = None

    if model_name.lower() == 'cnn':
        data = preprocess_data(raw_data, True)
        model = BaseCNN(input_dim=4, num_classes=6).to(device)
        model.load_state_dict(torch.load('./src/api/pt_model/best_CNN.pth', map_location=torch.device(device)))
        model.eval()
        with torch.no_grad():
            result = model(data).argmax(dim=1).numpy()
            result = tag_map.get(result[0])

    elif model_name.lower() == 'lstm':
        data = preprocess_data(raw_data, True)
        model = LSTMClassifier(input_dim=4, hidden_dim=160, num_classes=6, num_layers=2, dropout=0.3189801757552085).to(device)
        model.load_state_dict(torch.load('./src/api/pt_model/best_lstm_model.pth', map_location=torch.device(device)))
        model.eval()
        data = data.unsqueeze(0)
        with torch.no_grad():
            result = model(data).argmax(dim=1).numpy()
            result = tag_map.get(result[0])

    elif model_name.lower() == 'rnn':
        data = preprocess_data(raw_data, True)
        model = RNNClassifier(input_dim=4, hidden_dim=192, num_classes=6, num_layers=3).to(device)
        model.load_state_dict(torch.load('./src/api/pt_model/best_rnn_model.pth', map_location=torch.device(device)))
        model.eval()
        data = data.unsqueeze(0)
        with torch.no_grad():
            result = model(data).argmax(dim=1).numpy()
            result = tag_map.get(result[0])

    elif model_name.lower() == 'transformer':
        data = preprocess_data(raw_data, True)
        model = TransformerClassifier(
            input_dim=4,
            num_classes=6,
            num_heads=8,
            num_layers=3,
            hidden_dim=256,
            dropout=0.3220021040383345
        ).to(device)
        model.load_state_dict(torch.load('./src/api/pt_model/best_Transformer_model.pth', map_location=torch.device(device)))
        model.eval()
        data = data.unsqueeze(0)
        with torch.no_grad():
            result = model(data).argmax(dim=1).numpy()
            result = tag_map.get(result[0])

    elif model_name.lower() == 'kmeans':
        data = preprocess_data(raw_data, False)
        with open('./src/api/pt_model/kmeans_model.pkl', 'rb') as f:
            model = pickle.load(f)
        result = model.predict(data)
        result = tag_map.get(result[0])

    elif model_name.lower() == 'random_forest':
        data = preprocess_data(raw_data, False)
        with open('./src/api/pt_model/randomforest_model.pkl', 'rb') as f:
            model = pickle.load(f)
        result = model.predict(data)
        result = tag_map.get(result[0])

    else:
        raise ValueError(f"unknown: {model_name}")

    return result

# test
sample_raw_data = {"structureId": "2B3C", "chainId": "A", "sequence": "GCTAGCTA", "residueCount": 8, "macromoleculeType": "DNA/RNA Hybrid"}

classification_result = classify_with_model('kmeans', sample_raw_data)
print("classification_result:", classification_result)
