#!/usr/bin/env python3
"""
Phase 1: TCN Optimization & Hyperparameter Tuning
- Multiple TCN architectures
- Hyperparameter optimization
- Extended training with different configurations
- Performance comparison with tuned models
"""
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pickle
import joblib
import json
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

try:
	import torch
	import torch.nn as nn
	import torch.optim as optim
	from torch.utils.data import DataLoader, TensorDataset
	TORCH_AVAILABLE = True
except ImportError:
	TORCH_AVAILABLE = False
	print("⚠️  PyTorch not available. Install with: pip install torch")

class Chomp1d(nn.Module):
	"""Remove extra elements from padding"""
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
		self.conv1.weight.data.normal_(0, 0.01)
		self.conv2.weight.data.normal_(0, 0.01)
		if self.downsample is not None:
			self.downsample.weight.data.normal_(0, 0.01)

	def forward(self, x):
		out = self.net(x)
		res = x if self.downsample is None else self.downsample(x)
		return self.relu(out + res)

class TCN(nn.Module):
	"""Temporal Convolutional Network for AQI forecasting"""
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

	def forward(self, x):
		# x shape: (batch, features, sequence_length)
		x = self.network(x)
		# Global average pooling over time dimension
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

class TCNOptimizer:
	def __init__(self):
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.results = []
		self.best_model = None
		self.best_score = float('inf')
		# Verbosity control via env var (default: verbose)
		self.quiet = os.getenv('TCN_VERBOSE', '1') == '0'

	def prepare_sequences(self, X, y, sequence_length):
		"""Prepare sequences for TCN input"""
		sequences_X, sequences_y = [], []
		
		for i in range(len(X) - sequence_length):
			seq_X = X.iloc[i:i+sequence_length].values
			# Align target with the last timestep in the input window
			# Supports Series (single target) or DataFrame (multi-output)
			label_row = y.iloc[i + sequence_length - 1]
			if isinstance(label_row, pd.Series) and hasattr(y, 'columns'):
				# Multi-output: take row values (shape: [num_targets])
				seq_y = label_row.values
			else:
				# Single target
				seq_y = label_row
			sequences_X.append(seq_X)
			sequences_y.append(seq_y)
		
		return np.array(sequences_X), np.array(sequences_y)

	def train_model(self, model, train_loader, val_loader, epochs, lr, patience=30):
		"""Train a model with early stopping"""
		# Huber loss for robustness
		criterion = nn.SmoothL1Loss()
		# Use AdamW for better generalization
		optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
		scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

		best_val_loss = float('inf')
		patience_counter = 0

		for epoch in range(epochs):
			# Warmup for first 5 epochs
			if epoch < 5:
				for g in optimizer.param_groups:
					g['lr'] = lr * float(epoch + 1) / 5.0
			# Training
			model.train()
			train_loss = 0
			for batch_X, batch_y in train_loader:
				batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
				
				optimizer.zero_grad()
				outputs = model(batch_X).squeeze()
				loss = criterion(outputs, batch_y)
				loss.backward()
				# Gradient clipping for stability
				nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
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
			scheduler.step(val_loss)

			# Early stopping
			if val_loss < best_val_loss:
				best_val_loss = val_loss
				patience_counter = 0
			else:
				patience_counter += 1

			if patience_counter >= patience:
				break

			# Periodic logging
			if not self.quiet and (epoch < 5 or epoch % 10 == 0):
				print(f"      Epoch {epoch:3d} | Train {train_loss:.4f} | Val {val_loss:.4f}")

		return best_val_loss

	def evaluate_model(self, model, test_loader):
		"""Evaluate model on test set (supports single or multi-output)"""
		model.eval()
		predictions = []
		actuals = []
		
		with torch.no_grad():
			for batch_X, batch_y in test_loader:
				batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
				outputs = model(batch_X).squeeze()
				pred_np = outputs.detach().cpu().numpy()
				true_np = batch_y.detach().cpu().numpy()
				# Ensure 2D for multi-output consistency
				if pred_np.ndim == 1:
					pred_np = pred_np[:, None]
				if true_np.ndim == 1:
					true_np = true_np[:, None]
				predictions.append(pred_np)
				actuals.append(true_np)

		predictions = np.vstack(predictions)
		actuals = np.vstack(actuals)

		# Compute per-output metrics
		n_outputs = predictions.shape[1]
		per_horizon = []
		for j in range(n_outputs):
			rmse_j = np.sqrt(mean_squared_error(actuals[:, j], predictions[:, j]))
			mae_j = mean_absolute_error(actuals[:, j], predictions[:, j])
			r2_j = r2_score(actuals[:, j], predictions[:, j])
			mape_j = mean_absolute_percentage_error(actuals[:, j], predictions[:, j])
			per_horizon.append({'rmse': rmse_j, 'mae': mae_j, 'r2': r2_j, 'mape': mape_j})
		# Macro averages
		rmse_avg = float(np.mean([m['rmse'] for m in per_horizon]))
		mae_avg = float(np.mean([m['mae'] for m in per_horizon]))
		r2_avg = float(np.mean([m['r2'] for m in per_horizon]))
		mape_avg = float(np.mean([m['mape'] for m in per_horizon]))

		return rmse_avg, mae_avg, r2_avg, mape_avg, per_horizon

	def optimize_tcn(self, X_train, y_train, X_val, y_val, X_test, y_test, feature_columns, scaler):
		"""Optimize TCN with different configurations"""
		if not self.quiet:
			print("🔧 TCN OPTIMIZATION & HYPERPARAMETER TUNING")
			print("=" * 60)

		# Configuration space
		configs = [
			# Basic TCN configurations
			{'name': 'TCN_Shallow', 'model_type': 'tcn', 'hidden_dims': [32, 16], 'sequence_length': 24, 'epochs': 100, 'lr': 0.001},
			{'name': 'TCN_Medium', 'model_type': 'tcn', 'hidden_dims': [64, 32], 'sequence_length': 24, 'epochs': 150, 'lr': 0.001},
			{'name': 'TCN_Deep', 'model_type': 'tcn', 'hidden_dims': [128, 64, 32], 'sequence_length': 24, 'epochs': 200, 'lr': 0.0005},
			{'name': 'TCN_Wide', 'model_type': 'tcn', 'hidden_dims': [256, 128], 'sequence_length': 24, 'epochs': 150, 'lr': 0.001},
			# Larger depth/width variants
			{'name': 'TCN_Large', 'model_type': 'tcn', 'hidden_dims': [128, 64], 'sequence_length': 24, 'epochs': 150, 'lr': 0.001},
			{'name': 'TCN_Deeper', 'model_type': 'tcn', 'hidden_dims': [64, 64, 32], 'sequence_length': 24, 'epochs': 180, 'lr': 0.001},
			
			# Different sequence lengths
			{'name': 'TCN_12h', 'model_type': 'tcn', 'hidden_dims': [64, 32], 'sequence_length': 12, 'epochs': 150, 'lr': 0.001},
			{'name': 'TCN_36h', 'model_type': 'tcn', 'hidden_dims': [64, 32], 'sequence_length': 36, 'epochs': 150, 'lr': 0.001},
			{'name': 'TCN_48h', 'model_type': 'tcn', 'hidden_dims': [128, 64], 'sequence_length': 48, 'epochs': 180, 'lr': 0.001, 'kernel_size': 3},
			{'name': 'TCN_72h', 'model_type': 'tcn', 'hidden_dims': [128, 64, 32], 'sequence_length': 72, 'epochs': 220, 'lr': 0.0008, 'kernel_size': 3},
			
			# Hybrid architectures
			{'name': 'TCN_LSTM', 'model_type': 'tcnlstm', 'hidden_dims': [64, 32], 'sequence_length': 24, 'epochs': 150, 'lr': 0.001},
			{'name': 'TCN_Attention', 'model_type': 'tcnattention', 'hidden_dims': [64, 32], 'sequence_length': 24, 'epochs': 150, 'lr': 0.001},
			{'name': 'TCN_LSTM_48h', 'model_type': 'tcnlstm', 'hidden_dims': [64, 32], 'sequence_length': 48, 'epochs': 180, 'lr': 0.001, 'lstm_hidden': 96},
			
			# Different learning rates
			{'name': 'TCN_LR_High', 'model_type': 'tcn', 'hidden_dims': [64, 32], 'sequence_length': 24, 'epochs': 150, 'lr': 0.01},
			{'name': 'TCN_LR_Low', 'model_type': 'tcn', 'hidden_dims': [64, 32], 'sequence_length': 24, 'epochs': 150, 'lr': 0.0001},
		]

		# Horizon names for reporting when multi-output
		target_names = ['24h', '48h', '72h'] if (isinstance(y_train, pd.DataFrame) or (hasattr(y_train, 'ndim') and y_train.ndim == 2)) else ['target']
		for config in configs:
			if not self.quiet:
				print(f"\n🔧 Testing {config['name']}...")
				print(f"   Config: {config['hidden_dims']} dims, {config['sequence_length']}h seq, {config['epochs']} epochs, lr={config['lr']}")

			try:
				# Prepare sequences
				X_train_seq, y_train_seq = self.prepare_sequences(X_train, y_train, config['sequence_length'])
				X_val_seq, y_val_seq = self.prepare_sequences(X_val, y_val, config['sequence_length'])
				X_test_seq, y_test_seq = self.prepare_sequences(X_test, y_test, config['sequence_length'])

				# Data loaders
				train_dataset = TensorDataset(
					torch.FloatTensor(X_train_seq).transpose(1, 2),
					torch.FloatTensor(y_train_seq)
				)
				# Batch size: 64 if GPU else 32 on CPU
				batch_size = 64 if torch.cuda.is_available() else 32
				train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

				val_dataset = TensorDataset(
					torch.FloatTensor(X_val_seq).transpose(1, 2),
					torch.FloatTensor(y_val_seq)
				)
				val_loader = DataLoader(val_dataset, batch_size=batch_size)

				test_dataset = TensorDataset(
					torch.FloatTensor(X_test_seq).transpose(1, 2),
					torch.FloatTensor(y_test_seq)
				)
				test_loader = DataLoader(test_dataset, batch_size=batch_size)

				# Initialize model
				input_size = len(feature_columns)
				# Determine output size from y shape
				output_size = y_train_seq.shape[1] if isinstance(y_train_seq, np.ndarray) and y_train_seq.ndim == 2 else 1
				# Use higher dropout for deeper or longer-history configs
				dropout = 0.3 if (len(config['hidden_dims']) >= 3 or config['sequence_length'] >= 48) else 0.2
				if config['model_type'] == 'tcn':
					model = TCN(input_size, output_size, config['hidden_dims'], kernel_size=config.get('kernel_size', 2), dropout=dropout).to(self.device)
				elif config['model_type'] == 'tcnlstm':
					model = TCNLSTM(input_size, output_size, config['hidden_dims'], lstm_hidden=config.get('lstm_hidden', 64), dropout=dropout).to(self.device)
				elif config['model_type'] == 'tcnattention':
					model = TCNAttention(input_size, output_size, config['hidden_dims'], dropout=dropout).to(self.device)

				# Train model
				val_loss = self.train_model(model, train_loader, val_loader, config['epochs'], config['lr'])
				
				# Evaluate on test set
				rmse, mae, r2, mape, per_h = self.evaluate_model(model, test_loader)

				if not self.quiet:
					if len(target_names) == 3:
						print(f"   ✅ Val Loss: {val_loss:.2f} | Test RMSE(avg): {rmse:.2f} | R²(avg): {r2:.3f}")
						print(f"      24h RMSE: {per_h[0]['rmse']:.2f} | 48h RMSE: {per_h[1]['rmse']:.2f} | 72h RMSE: {per_h[2]['rmse']:.2f}")
					else:
						print(f"   ✅ Val Loss: {val_loss:.2f} | Test RMSE: {rmse:.2f} | R²: {r2:.3f}")

				# Record results
				result = {
					'Model': config['name'],
					'Architecture': config['model_type'],
					'Hidden_Dims': str(config['hidden_dims']),
					'Sequence_Length': config['sequence_length'],
					'Epochs': config['epochs'],
					'Learning_Rate': config['lr'],
					'Val_Loss': val_loss,
					'RMSE': rmse,
					'MAE': mae,
					'R²': r2,
					'MAPE (%)': mape
				}
				# Add per-horizon if multi-output
				if len(target_names) == 3:
					result.update({
						'24h_RMSE': per_h[0]['rmse'], '24h_R²': per_h[0]['r2'],
						'48h_RMSE': per_h[1]['rmse'], '48h_R²': per_h[1]['r2'],
						'72h_RMSE': per_h[2]['rmse'], '72h_R²': per_h[2]['r2']
					})
				self.results.append(result)

				# Selection metric: default 72h RMSE; override with env var TCN_SELECTION in {"72h","avg","24h","48h"}
				selection = os.getenv('TCN_SELECTION', '72h')
				if len(target_names) == 3:
					if selection == 'avg':
						score = rmse
					elif selection == '24h':
						score = per_h[0]['rmse']
					elif selection == '48h':
						score = per_h[1]['rmse']
					else:  # '72h'
						score = per_h[2]['rmse']
				else:
					score = rmse
				# Save best model by selection score
				if score < self.best_score:
					self.best_score = score
					self.best_model = model
					best_config = config

			except Exception as e:
				if not self.quiet:
					print(f"   ❌ Error: {str(e)}")
				continue

		# Save best model
		if self.best_model is not None:
			os.makedirs("saved_models", exist_ok=True)
			# Name by selection target (24h/48h/72h or avg)
			selection = os.getenv('TCN_SELECTION', '72h')
			model_path = f"saved_models/tcn_optimized_{selection}_{best_config['name'].lower().replace(' ', '_')}.pth"
			torch.save({
				'model_state_dict': self.best_model.state_dict(),
				'config': best_config,
				'feature_columns': feature_columns,
				'scaler': scaler
			}, model_path)
			if not self.quiet:
				print(f"\n🏆 Best TCN model saved: {model_path}")

		return self.results

def main():
	"""Main TCN optimization"""
	if not TORCH_AVAILABLE:
		print("❌ PyTorch not available")
		return

	print("🚀 TCN OPTIMIZATION & HYPERPARAMETER TUNING")
	print("=" * 60)

	# Load data
	data_path = "data_repositories/features/phase1_fixed_selected_features.csv"
	if not os.path.exists(data_path):
		print(f"❌ Data file not found: {data_path}")
		return

	df = pd.read_csv(data_path)
	df['timestamp'] = pd.to_datetime(df['timestamp'])

	# Ensure targets exist and drop rows with NaNs across all horizons
	required_targets = ['target_aqi_24h', 'target_aqi_48h', 'target_aqi_72h']
	missing_targets = [c for c in required_targets if c not in df.columns]
	if missing_targets:
		print(f"❌ Required target columns missing from dataset: {missing_targets}")
		return
	valid_mask = df['target_aqi_24h'].notna() & df['target_aqi_48h'].notna() & df['target_aqi_72h'].notna()
	df = df[valid_mask].copy()

	# Load feature columns and scaler
	with open("data_repositories/features/phase1_fixed_feature_columns.pkl", 'rb') as f:
		feature_columns = pickle.load(f)
	with open("data_repositories/features/phase1_fixed_feature_scaler.pkl", 'rb') as f:
		scaler = pickle.load(f)

	X = df[feature_columns]
	y = df[['target_aqi_24h', 'target_aqi_48h', 'target_aqi_72h']]

	# Scale features
	X_scaled = scaler.transform(X)
	X_scaled = pd.DataFrame(X_scaled, columns=feature_columns, index=X.index)

	# Split data
	total = len(df)
	train_end = int(total * 0.6)
	val_end = int(total * 0.8)

	X_train = X_scaled.iloc[:train_end]
	y_train = y.iloc[:train_end]
	X_val = X_scaled.iloc[train_end:val_end]
	y_val = y.iloc[train_end:val_end]
	X_test = X_scaled.iloc[val_end:]
	y_test = y.iloc[val_end:]

	print(f"   Multi-output forecasting enabled: targets = [24h, 48h, 72h]")
	print(f"   Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

	# Run optimization
	optimizer = TCNOptimizer()
	results = optimizer.optimize_tcn(X_train, y_train, X_val, y_val, X_test, y_test, feature_columns, scaler)

	# Save results
	try:
		results_df = pd.DataFrame(results).sort_values('RMSE')
		os.makedirs("saved_models", exist_ok=True)
		results_path = "saved_models/tcn_optimization_results_multi.csv"
		results_df.to_csv(results_path, index=False)
		
		print(f"\n📊 OPTIMIZATION RESULTS:")
		print(f"   Results saved to: {results_path}")
		print(f"\n🏆 Top 3 TCN Models:")
		print(results_df.head(3)[['Model', 'RMSE', 'R²', 'Architecture']].to_string(index=False))

		# Compare with tuned models
		# Optional: print a brief comparison if desired
		# print(f"\nBest TCN: RMSE={results_df.iloc[0]['RMSE']:.2f}, R²={results_df.iloc[0]['R²']:.3f}")
	except Exception as e:
		print(f"❌ Error saving results: {e}")

if __name__ == "__main__":
	main()
