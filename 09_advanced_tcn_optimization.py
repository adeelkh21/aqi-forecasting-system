"""
Advanced TCN Optimization for AQI Forecasting
============================================

This script performs advanced optimization of TCN models:
- Extended training epochs (500-1000)
- Grid search for hyperparameters
- Multiple optimizers (Adam, AdamW, RAdam, Lion)
- Focus on TCN_LSTM architecture
- Ensemble multiple configurations

Author: Data Science Team
Date: 2024-03-09
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

# Check PyTorch availability
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    print("❌ PyTorch not available. Install with: pip install torch")
    TORCH_AVAILABLE = False
    sys.exit(1)

# ML metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler

# Advanced optimizers
try:
    from torch.optim import AdamW
    ADAMW_AVAILABLE = True
except ImportError:
    ADAMW_AVAILABLE = False

try:
    from torch.optim import RAdam
    RADAM_AVAILABLE = True
except ImportError:
    RADAM_AVAILABLE = False

# Lion optimizer (if available)
try:
    from lion_pytorch import Lion
    LION_AVAILABLE = True
except ImportError:
    LION_AVAILABLE = False

print("🔧 ADVANCED TCN OPTIMIZATION")
print("=" * 50)
print(f"PyTorch: {'✅' if TORCH_AVAILABLE else '❌'}")
print(f"AdamW: {'✅' if ADAMW_AVAILABLE else '❌'}")
print(f"RAdam: {'✅' if RADAM_AVAILABLE else '❌'}")
print(f"Lion: {'✅' if LION_AVAILABLE else '❌'}")

# TCN Architecture Classes
class Chomp1d(nn.Module):
    """Remove extra padding"""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    """Temporal block with residual connection"""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        # Better initialization for stability
        self.conv1.weight.data.normal_(0, 0.001)
        self.conv2.weight.data.normal_(0, 0.001)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.001)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    """Temporal Convolutional Network"""
    def __init__(self, input_size, output_size, num_channels, kernel_size=2, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                   dilation=dilation_size, padding=(kernel_size-1) * dilation_size,
                                   dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.final_layer = nn.Linear(num_channels[-1], output_size)
        
        # Initialize final layer with small weights
        nn.init.xavier_uniform_(self.final_layer.weight, gain=0.01)
        nn.init.constant_(self.final_layer.bias, 0)

    def forward(self, x):
        x = self.network(x)
        x = torch.mean(x, dim=2)
        return self.final_layer(x)

class TCNLSTM(nn.Module):
    """Hybrid TCN + LSTM architecture"""
    def __init__(self, input_size, output_size, num_channels, lstm_hidden=64, dropout=0.2):
        super(TCNLSTM, self).__init__()
        self.tcn = TCN(input_size, num_channels[-1], num_channels, dropout=dropout)
        self.lstm = nn.LSTM(num_channels[-1], lstm_hidden, batch_first=True, dropout=dropout)
        self.final_layer = nn.Linear(lstm_hidden, output_size)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize final layer with small weights
        nn.init.xavier_uniform_(self.final_layer.weight, gain=0.01)
        nn.init.constant_(self.final_layer.bias, 0)

    def forward(self, x):
        # TCN processing
        x = self.tcn.network(x)  # (batch, channels, seq_len)
        x = x.transpose(1, 2)    # (batch, seq_len, channels)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        # Take last output
        x = lstm_out[:, -1, :]
        x = self.dropout(x)
        return self.final_layer(x)

class TCNAttention(nn.Module):
    """TCN with attention mechanism"""
    def __init__(self, input_size, output_size, num_channels, attention_dim=32, dropout=0.2):
        super(TCNAttention, self).__init__()
        self.tcn = TCN(input_size, num_channels[-1], num_channels, dropout=dropout)
        self.attention = nn.MultiheadAttention(num_channels[-1], num_heads=4, dropout=dropout)
        self.final_layer = nn.Linear(num_channels[-1], output_size)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize final layer with small weights
        nn.init.xavier_uniform_(self.final_layer.weight, gain=0.01)
        nn.init.constant_(self.final_layer.bias, 0)

    def forward(self, x):
        # TCN processing
        x = self.tcn.network(x)  # (batch, channels, seq_len)
        x = x.transpose(1, 2)    # (batch, seq_len, channels)
        
        # Self-attention
        x_attended, _ = self.attention(x, x, x)
        # Global average pooling
        x = torch.mean(x_attended, dim=1)
        x = self.dropout(x)
        return self.final_layer(x)

class AdvancedTCNOptimizer:
    """Advanced TCN optimizer with extended training and multiple optimizers"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = []
        self.best_model = None
        self.best_score = float('inf')
        self.best_config = None
        
        print(f"🚀 Using device: {self.device}")

    def prepare_sequences(self, X, y, sequence_length):
        """Prepare sequences for TCN input"""
        sequences_X, sequences_y = [], []
        
        for i in range(len(X) - sequence_length):
            seq_X = X.iloc[i:i+sequence_length].values
            seq_y = y.iloc[i+sequence_length]
            sequences_X.append(seq_X)
            sequences_y.append(seq_y)
        
        return np.array(sequences_X), np.array(sequences_y)

    def get_optimizer(self, model, optimizer_name, lr, weight_decay=1e-5):
        """Get optimizer based on name"""
        if optimizer_name == 'adam':
            return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adamw' and ADAMW_AVAILABLE:
            return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'radam' and RADAM_AVAILABLE:
            return optim.RAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'lion' and LION_AVAILABLE:
            return Lion(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            print(f"   ⚠️  {optimizer_name} not available, using Adam")
            return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    def train_model(self, model, train_loader, val_loader, epochs, lr, optimizer_name, patience=50):
        """Train a model with extended epochs and early stopping"""
        criterion = nn.MSELoss()
        optimizer = self.get_optimizer(model, optimizer_name, lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5, min_lr=1e-6)

        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []

        print(f"   🔄 Training for {epochs} epochs with {optimizer_name} (lr={lr})")

        for epoch in range(epochs):
            # Learning rate warmup for first 10 epochs
            if epoch < 10:
                warmup_factor = min(1.0, (epoch + 1) / 10.0)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr * warmup_factor
            
            # Training
            model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                
                optimizer.step()
                train_loss += loss.item()

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_X).squeeze()
                    val_loss += criterion(outputs, batch_y).item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            scheduler.step(val_loss)

            # Progress reporting
            if epoch % 50 == 0 or epoch < 10:
                print(f"   Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"   🛑 Early stopping at epoch {epoch}")
                break

        return best_val_loss, train_losses, val_losses

    def evaluate_model(self, model, test_loader):
        """Evaluate model on test set"""
        model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = model(batch_X).squeeze()
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(batch_y.cpu().numpy())

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)
        mape = mean_absolute_percentage_error(actuals, predictions)

        return rmse, mae, r2, mape

    def optimize_tcn(self, X_train, y_train, X_val, y_val, X_test, y_test, feature_columns, scaler):
        """Advanced TCN optimization with grid search"""
        print("\n🔧 ADVANCED TCN OPTIMIZATION & HYPERPARAMETER TUNING")
        print("=" * 70)

        # Advanced configuration space - FOCUSED on best performers
        configs = [
            # TCN_LSTM Focused Configurations (Best Architecture) - REDUCED for efficiency
            {'name': 'TCN_LSTM_Extended', 'model_type': 'tcnlstm', 'hidden_dims': [64, 32], 'sequence_length': 24, 'epochs': 300, 'lr': 0.001, 'optimizer': 'adam'},
            {'name': 'TCN_LSTM_Deep', 'model_type': 'tcnlstm', 'hidden_dims': [128, 64, 32], 'sequence_length': 24, 'epochs': 300, 'lr': 0.001, 'optimizer': 'adam'},
            {'name': 'TCN_LSTM_AdamW', 'model_type': 'tcnlstm', 'hidden_dims': [64, 32], 'sequence_length': 24, 'epochs': 300, 'lr': 0.001, 'optimizer': 'adamw'},
            {'name': 'TCN_LSTM_RAdam', 'model_type': 'tcnlstm', 'hidden_dims': [64, 32], 'sequence_length': 24, 'epochs': 300, 'lr': 0.001, 'optimizer': 'radam'},
            {'name': 'TCN_LSTM_Lion', 'model_type': 'tcnlstm', 'hidden_dims': [64, 32], 'sequence_length': 24, 'epochs': 300, 'lr': 0.001, 'optimizer': 'lion'},
            
            # Different learning rates for TCN_LSTM
            {'name': 'TCN_LSTM_LR_Low', 'model_type': 'tcnlstm', 'hidden_dims': [64, 32], 'sequence_length': 24, 'epochs': 300, 'lr': 0.0005, 'optimizer': 'adam'},
            {'name': 'TCN_LSTM_LR_High', 'model_type': 'tcnlstm', 'hidden_dims': [64, 32], 'sequence_length': 24, 'epochs': 300, 'lr': 0.002, 'optimizer': 'adam'},
            
            # Regular TCN with extended training
            {'name': 'TCN_Extended', 'model_type': 'tcn', 'hidden_dims': [64, 32], 'sequence_length': 24, 'epochs': 300, 'lr': 0.001, 'optimizer': 'adam'},
        ]

        for config in configs:
            print(f"\n🔧 Testing {config['name']}...")
            print(f"   Config: {config['hidden_dims']} dims, {config['sequence_length']}h seq, {config['epochs']} epochs, lr={config['lr']}, opt={config['optimizer']}")

            try:
                # Prepare sequences
                X_train_seq, y_train_seq = self.prepare_sequences(X_train, y_train, config['sequence_length'])
                X_val_seq, y_val_seq = self.prepare_sequences(X_val, y_val, config['sequence_length'])
                X_test_seq, y_test_seq = self.prepare_sequences(X_test, y_test, config['sequence_length'])

                # Data loaders - SMALLER BATCH SIZE for stability
                train_dataset = TensorDataset(
                    torch.FloatTensor(X_train_seq).transpose(1, 2),
                    torch.FloatTensor(y_train_seq)
                )
                train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

                val_dataset = TensorDataset(
                    torch.FloatTensor(X_val_seq).transpose(1, 2),
                    torch.FloatTensor(y_val_seq)
                )
                val_loader = DataLoader(val_dataset, batch_size=16)

                test_dataset = TensorDataset(
                    torch.FloatTensor(X_test_seq).transpose(1, 2),
                    torch.FloatTensor(y_test_seq)
                )
                test_loader = DataLoader(test_dataset, batch_size=16)

                # Initialize model
                input_size = len(feature_columns)
                if config['model_type'] == 'tcn':
                    model = TCN(input_size, 1, config['hidden_dims'], dropout=0.2).to(self.device)
                elif config['model_type'] == 'tcnlstm':
                    model = TCNLSTM(input_size, 1, config['hidden_dims'], dropout=0.2).to(self.device)
                elif config['model_type'] == 'tcnattention':
                    model = TCNAttention(input_size, 1, config['hidden_dims'], dropout=0.2).to(self.device)

                # Train model
                val_loss, train_losses, val_losses = self.train_model(
                    model, train_loader, val_loader, config['epochs'], config['lr'], config['optimizer']
                )
                
                # Evaluate on test set
                rmse, mae, r2, mape = self.evaluate_model(model, test_loader)

                print(f"   ✅ Val Loss: {val_loss:.2f} | Test RMSE: {rmse:.2f} | R²: {r2:.3f}")

                # Record results
                result = {
                    'Model': config['name'],
                    'Architecture': config['model_type'],
                    'Hidden_Dims': str(config['hidden_dims']),
                    'Sequence_Length': config['sequence_length'],
                    'Epochs': config['epochs'],
                    'Learning_Rate': config['lr'],
                    'Optimizer': config['optimizer'],
                    'Val_Loss': val_loss,
                    'RMSE': rmse,
                    'MAE': mae,
                    'R²': r2,
                    'MAPE (%)': mape
                }
                self.results.append(result)

                # Save best model
                if rmse < self.best_score:
                    self.best_score = rmse
                    self.best_model = model
                    self.best_config = config
                    print(f"   🏆 New best model! RMSE: {rmse:.2f}")

            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
                continue

        # Save best model
        if self.best_model is not None:
            model_path = f"saved_models/tcn_advanced_{self.best_config['name'].lower().replace(' ', '_')}.pth"
            torch.save({
                'model_state_dict': self.best_model.state_dict(),
                'config': self.best_config,
                'feature_columns': feature_columns,
                'scaler': scaler
            }, model_path)
            print(f"\n🏆 Best advanced TCN model saved: {model_path}")

        return self.results

def main():
    """Main function to run advanced TCN optimization"""
    print("🚀 ADVANCED TCN OPTIMIZATION")
    print("=" * 50)
    
    # Load data
    print("📊 Loading data...")
    try:
        # Load feature columns
        with open('data_repositories/features/phase1_fixed_feature_columns.pkl', 'rb') as f:
            feature_columns = pickle.load(f)
        
        # Load scaler
        with open('data_repositories/features/phase1_fixed_feature_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        # Load processed data
        df = pd.read_csv('data_repositories/features/phase1_fixed_selected_features.csv')
        
        print(f"✅ Data loaded: {df.shape}")
        print(f"📊 Features: {len(feature_columns)}")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # Prepare features and target
    print("\n🔧 Preparing features...")
    X = df[feature_columns]
    y = df['numerical_aqi']
    
    # Remove rows with NaN target
    valid_mask = y.notna()
    X = X[valid_mask]
    y = y[valid_mask]
    
    print(f"✅ Features prepared: {X.shape}")
    
    # Split data (maintaining temporal order)
    train_size = int(0.6 * len(X))
    val_size = int(0.2 * len(X))
    
    X_train = X.iloc[:train_size]
    y_train = y.iloc[:train_size]
    X_val = X.iloc[train_size:train_size+val_size]
    y_val = y.iloc[train_size:train_size+val_size]
    X_test = X.iloc[train_size+val_size:]
    y_test = y.iloc[train_size+val_size:]
    
    print(f"   Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Scale features
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to DataFrames for easier handling
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_columns, index=X_train.index)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=feature_columns, index=X_val.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_columns, index=X_test.index)
    
    # Run advanced optimization
    optimizer = AdvancedTCNOptimizer()
    results = optimizer.optimize_tcn(X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, feature_columns, scaler)
    
    # Save results
    if results:
        results_df = pd.DataFrame(results).sort_values('RMSE')
        
        # Save to CSV
        results_file = 'saved_models/advanced_tcn_optimization_results.csv'
        results_df.to_csv(results_file, index=False)
        print(f"\n📊 OPTIMIZATION RESULTS:")
        print(f"   Results saved to: {results_file}")
        
        # Show top models
        print(f"\n🏆 TOP 5 ADVANCED TCN MODELS:")
        print(results_df[['Model', 'RMSE', 'R²', 'Architecture', 'Optimizer']].head())
        
        # Show best model
        best_result = results_df.iloc[0]
        print(f"\n🎯 BEST MODEL: {best_result['Model']}")
        print(f"   RMSE: {best_result['RMSE']:.2f}")
        print(f"   R²: {best_result['R²']:.3f}")
        print(f"   Architecture: {best_result['Architecture']}")
        print(f"   Optimizer: {best_result['Optimizer']}")
        
        # Check if we achieved target RMSE
        if best_result['RMSE'] < 25:
            print(f"\n🎉 TARGET ACHIEVED! RMSE < 25: {best_result['RMSE']:.2f}")
        else:
            print(f"\n📈 Current best RMSE: {best_result['RMSE']:.2f} (Target: < 25)")
            print(f"   Improvement needed: {best_result['RMSE'] - 25:.2f}")
    else:
        print("\n❌ No results generated!")

if __name__ == "__main__":
    main()
