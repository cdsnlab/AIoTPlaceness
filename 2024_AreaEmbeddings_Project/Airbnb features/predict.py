import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import csv

########################
# Utility Functions
########################

def load_data(embeddings_path, labels_path):
    raw_embeddings = np.loadtxt(embeddings_path, delimiter=',')
    raw_labels = np.loadtxt(labels_path, delimiter=',')
    raw_embeddings = raw_embeddings.reshape(67, 424, 3130)
    raw_labels = raw_labels.reshape(67, 424, 3)
    return raw_embeddings, raw_labels

def get_horizon(mode):
    if mode == '1m':
        return 1
    elif mode == '4m':
        return 4
    elif mode == '6m':
        return 6
    else:
        raise ValueError("Invalid mode")
    
def get_dynamic_paths(base_path, mode, seq_len, is_checkpoint=True):
    """
    Generate dynamic paths for saving and loading files, incorporating seq_len.
    """
    base_dir, raw_file_name = os.path.split(base_path)
    file_base, _ = os.path.splitext(raw_file_name)
    sub_dir = 'checkpoints' if is_checkpoint else 'norm_stats'
    dir_path = os.path.join(base_dir, sub_dir, mode)
    os.makedirs(dir_path, exist_ok=True)
    suffix = f"_seqlen{seq_len}_best_checkpoint.pth" if is_checkpoint else f"_seqlen{seq_len}_normalization_stats.npz"
    return os.path.join(dir_path, f"{file_base}{suffix}")

########################
# Dataset with Sliding Window
########################

class SlidingWindowTimeSeriesDataset(Dataset):
    def __init__(self, embeddings, labels, mode='1m', seq_len=12, subset='train', train_months=50, val_months=10):
        """
        embeddings: (N, 424, 2362)
        labels: (N, 424, 3)
        mode: '1m', '4m', or '6m'
        seq_len: input sequence length
        subset: 'train', 'val', or 'test'
        train_months: number of months used for training
        val_months: number of months used for validation
        """
        horizon = get_horizon(mode)
        N = embeddings.shape[0]

        # Split:
        # Train: [0 : train_months]
        # Val: [train_months : train_months+val_months]
        # Test: [train_months+val_months : N]
        if subset == 'train':
            data_embeddings = embeddings[:train_months]
            data_labels = labels[:train_months]
        elif subset == 'val':
            data_embeddings = embeddings[train_months:train_months+val_months]
            data_labels = labels[train_months:train_months+val_months]
        elif subset == 'test':
            data_embeddings = embeddings[train_months+val_months:]
            data_labels = labels[train_months+val_months:]
        else:
            raise ValueError("subset must be 'train', 'val', or 'test'")

        self.subset = subset
        self.mode = mode
        self.seq_len = seq_len
        self.horizon = horizon
        self.raw_embeddings = data_embeddings
        self.raw_labels = data_labels

        total_data = data_embeddings.shape[0]
        self.max_start = total_data - seq_len
        self.num_samples = max(0, self.max_start)

        # Compute normalization stats for training set
        if subset == 'train':
            self.emb_mean = data_embeddings.mean()
            self.emb_std = data_embeddings.std() + 1e-8
            self.lbl_mean = data_labels.mean()
            self.lbl_std = data_labels.std() + 1e-8
        else:
            # Will be set later from train stats
            self.emb_mean = None
            self.emb_std = None
            self.lbl_mean = None
            self.lbl_std = None

    def set_normalization_stats(self, emb_mean, emb_std, lbl_mean, lbl_std):
        self.emb_mean = emb_mean
        self.emb_std = emb_std
        self.lbl_mean = lbl_mean
        self.lbl_std = lbl_std

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = self.raw_embeddings[idx : idx+self.seq_len]
        y = self.raw_labels[idx : idx+self.seq_len]

        x = (x - self.emb_mean) / self.emb_std
        y = (y - self.lbl_mean) / self.lbl_std

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

########################
# Model Definition
########################

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(1)

    def forward(self, x):
        seq_len = x.size(0)
        return x + self.pe[:seq_len, :].to(x.device)

class EmbeddingTransformer(nn.Module):
    def __init__(self, input_dims=(23, 35, 2304), hidden_dim=36, output_dim=3,
                 num_encoder_layers=4, nhead=4, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.input_dims = input_dims
        d1, d2, d3 = input_dims

        # Always use a linear layer to transform input_dim to hidden_dim=36
        self.fc1 = nn.Linear(d1, hidden_dim)
        self.fc2 = nn.Linear(d2, hidden_dim)
        self.fc3 = nn.Linear(d3, hidden_dim)

        self.input_dim = hidden_dim * 3  # 36*3=108, even number
        self.pos_encoder = PositionalEncoding(self.input_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.input_dim, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.fc_out = nn.Linear(self.input_dim, output_dim)

    def forward(self, x):
        # x: (batch_size, seq_len, 424, features) or (seq_len, 424, features)
        if x.dim() == 4:
            # (batch_size, seq_len, 424, features)
            batch_size, seq_len, num_nodes, feat_dim = x.size()
            x = x.permute(1, 0, 2, 3)  # (seq_len, batch_size, 424, features)
        else:
            # (seq_len, 424, features)
            seq_len, num_nodes, feat_dim = x.size()
            x = x.unsqueeze(1) # (seq_len, 1, 424, features)
            batch_size = 1

        d1, d2, d3 = self.input_dims
        # Slice the input according to input_dims
        x1 = x[..., :d1]
        x2 = x[..., d1:d1+d2]
        x3 = x[..., d1+d2:]

        # Apply FC layers (always)
        x1 = self.fc1(x1)
        x2 = self.fc2(x2)
        x3 = self.fc3(x3)

        x_cat = torch.cat([x1, x2, x3], dim=-1)  # (seq_len, batch_size, 424, 108)

        seq_len, bsz, num_nodes, d_model = x_cat.size()
        x_cat = x_cat.reshape(seq_len, bsz * num_nodes, d_model)

        x_pe = self.pos_encoder(x_cat)
        x_trans = self.transformer_encoder(x_pe)
        out = self.fc_out(x_trans)  # (seq_len, bsz*424, 3)
        out = out.reshape(seq_len, bsz, num_nodes, 3)
        return out

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

########################
# Training Function with Validation
########################

def train_model(full_embeddings, full_labels, embeddings_path, mode='1m', seq_len=12, 
                train_months=50, val_months=10, epochs=100, batch_size=16, lr=1e-5, 
                device='cuda:3', input_dims=(23,35,2304)):
    # Create datasets
    train_dataset = SlidingWindowTimeSeriesDataset(full_embeddings, full_labels, mode=mode, 
                                                   seq_len=seq_len, subset='train', 
                                                   train_months=train_months, val_months=val_months)
    val_dataset = SlidingWindowTimeSeriesDataset(full_embeddings, full_labels, mode=mode, 
                                                 seq_len=seq_len, subset='val', 
                                                 train_months=train_months, val_months=val_months)
    test_dataset = SlidingWindowTimeSeriesDataset(full_embeddings, full_labels, mode=mode, 
                                                  seq_len=seq_len, subset='test', 
                                                  train_months=train_months, val_months=val_months)

    # Set normalization stats for val and test
    val_dataset.set_normalization_stats(train_dataset.emb_mean, train_dataset.emb_std, 
                                        train_dataset.lbl_mean, train_dataset.lbl_std)
    test_dataset.set_normalization_stats(train_dataset.emb_mean, train_dataset.emb_std, 
                                         train_dataset.lbl_mean, train_dataset.lbl_std)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = EmbeddingTransformer(input_dims=input_dims, hidden_dim=36).to(device)
    model.apply(init_weights)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Dynamic paths for saving the best model and normalization stats
    checkpoint_path = get_dynamic_paths(embeddings_path, mode, seq_len, is_checkpoint=True)
    norm_stats_path = get_dynamic_paths(embeddings_path, mode, seq_len, is_checkpoint=False)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_pred = model(x)
            y = y.permute(1, 0, 2, 3)
            loss = criterion(y_pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                y_val_pred = model(x_val)
                y_val = y_val.permute(1, 0, 2, 3)
                val_loss += criterion(y_val_pred, y_val).item()

        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Best model updated and saved to {checkpoint_path}!")

    # Save normalization stats
    np.savez(norm_stats_path,
             emb_mean=train_dataset.emb_mean,
             emb_std=train_dataset.emb_std,
             lbl_mean=train_dataset.lbl_mean,
             lbl_std=train_dataset.lbl_std)
    print(f"Normalization stats saved to {norm_stats_path}.")

    return val_loader, model

########################
# Testing Function (Load Best Model and Evaluate)
########################

def test_model(full_embeddings, full_labels, embeddings_path, mode='1m', seq_len=12, 
               train_months=50, val_months=10, batch_size=16, device='cuda:3', input_dims=(23,35,2304)):
    # Paths
    checkpoint_path = get_dynamic_paths(embeddings_path, mode, seq_len, is_checkpoint=True)
    checkpoint_name = os.path.basename(checkpoint_path)  # Extract checkpoint name from path
    norm_stats_path = get_dynamic_paths(embeddings_path, mode, seq_len, is_checkpoint=False)
    results_path = os.path.join(os.path.dirname(embeddings_path), 'results.txt')

    # Load normalization stats
    stats = np.load(norm_stats_path, allow_pickle=True)
    emb_mean = stats['emb_mean']
    emb_std = stats['emb_std']
    lbl_mean = stats['lbl_mean']
    lbl_std = stats['lbl_std']

    test_dataset = SlidingWindowTimeSeriesDataset(full_embeddings, full_labels, mode=mode, 
                                                  seq_len=seq_len, subset='test', 
                                                  train_months=train_months, val_months=val_months)
    test_dataset.set_normalization_stats(emb_mean, emb_std, lbl_mean, lbl_std)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = EmbeddingTransformer(input_dims=input_dims, hidden_dim=36).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    predictions = []
    actuals = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            y = y.permute(1, 0, 2, 3)  # Align shapes
            predictions.append(y_pred.cpu().numpy())
            actuals.append(y.cpu().numpy())

    predictions = np.concatenate(predictions, axis=1)
    actuals = np.concatenate(actuals, axis=1)

    # Initialize metrics
    metrics = {"MSE": [], "MAE": [], "RMSE": [], "MAPE": []}

    # Compute metrics for combined shape (6, 6, 424, 3)
    mse = np.mean((predictions - actuals) ** 2)
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((predictions - actuals) / (actuals + 1e-8))) * 100

    metrics["MSE"].append(mse)
    metrics["MAE"].append(mae)
    metrics["RMSE"].append(rmse)
    metrics["MAPE"].append(mape)

    # Compute metrics for each feature separately (6, 6, 424, 1)
    for feature_idx in range(3):
        pred_feature = predictions[..., feature_idx]  # (6, 6, 424, 1)
        actual_feature = actuals[..., feature_idx]  # (6, 6, 424, 1)

        mse = np.mean((pred_feature - actual_feature) ** 2)
        mae = np.mean(np.abs(pred_feature - actual_feature))
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((pred_feature - actual_feature) / (actual_feature + 1e-8))) * 100

        metrics["MSE"].append(mse)
        metrics["MAE"].append(mae)
        metrics["RMSE"].append(rmse)
        metrics["MAPE"].append(mape)

    # Print metrics
    print("Combined Metrics (6, 6, 424, 3):")
    print(f"MSE: {metrics['MSE'][0]:.4f}, MAE: {metrics['MAE'][0]:.4f}, RMSE: {metrics['RMSE'][0]:.4f}, MAPE: {metrics['MAPE'][0]:.4f}")

    for feature_idx in range(3):
        print(f"Metrics for Feature {feature_idx + 1} (6, 6, 424, 1):")
        print(f"MSE: {metrics['MSE'][feature_idx + 1]:.4f}, MAE: {metrics['MAE'][feature_idx + 1]:.4f}, RMSE: {metrics['RMSE'][feature_idx + 1]:.4f}, MAPE: {metrics['MAPE'][feature_idx + 1]:.4f}")

    # Save metrics to results.txt
    with open(results_path, 'a') as f:
        f.write(f"-----------------------\n")
        f.write(f"Checkpoint: {checkpoint_name}\n")  # Add checkpoint name
        f.write(f"Mode: {mode}, SeqLen: {seq_len}\n")
        f.write(f"Combined Metrics (6, 6, 424, 3):\n")
        f.write(f"MSE: {metrics['MSE'][0]:.4f}, MAE: {metrics['MAE'][0]:.4f}, RMSE: {metrics['RMSE'][0]:.4f}, MAPE: {metrics['MAPE'][0]:.4f}\n")
        for feature_idx in range(3):
            f.write(f"Metrics for Feature {feature_idx + 1} (6, 6, 424, 1):\n")
            f.write(f"MSE: {metrics['MSE'][feature_idx + 1]:.4f}, MAE: {metrics['MAE'][feature_idx + 1]:.4f}, RMSE: {metrics['RMSE'][feature_idx + 1]:.4f}, MAPE: {metrics['MAPE'][feature_idx + 1]:.4f}\n")
        f.write(f"-----------------------\n")

    print(f"Metrics saved to {results_path}.")

########################
# Main Usage
########################

def find_non_float_entries(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row_idx, row in enumerate(reader):
            for col_idx, value in enumerate(row):
                try:
                    float(value)  # Attempt to convert to float
                except ValueError:
                    print(f"Non-convertible value found at row {row_idx + 1}, column {col_idx + 1}: '{value}'")

if __name__ == "__main__":
    embeddings_path = 'quang/time_series_prediction/raw_embeddings/AirBnB_Embeddings.csv'
    labels_path = 'quang/time_series_prediction/labels.csv'
    find_non_float_entries(embeddings_path)
    full_embeddings, full_labels = load_data(embeddings_path, labels_path)

    mode = '1m'
    seq_len = 6
    train_months = 43
    val_months = 12
    epochs = 100
    batch_size = 16
    lr = 7e-6
    device = 'cuda:3' if torch.cuda.is_available() else 'cpu'

    # Change input_dims as needed:
    # input_dims = (35,35,35)
    input_dims = (23,35,3072)
    # input_dims = (23,35,2304)

    # Train and validate
    # test_loader, last_model = train_model(full_embeddings, full_labels, embeddings_path=embeddings_path,
    #                                       mode=mode, seq_len=seq_len, train_months=train_months,
    #                                       val_months=val_months, epochs=epochs, batch_size=batch_size,
    #                                       lr=lr, device=device, input_dims=input_dims)

    # Test using the saved best model
    # test_model(full_embeddings, full_labels, embeddings_path=embeddings_path,
    #            mode=mode, seq_len=seq_len, train_months=train_months, val_months=val_months,
    #            batch_size=batch_size, device=device, input_dims=input_dims)
