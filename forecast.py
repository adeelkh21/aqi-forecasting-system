#!/usr/bin/env python3
"""
Forecast runner (24h/48h/72h) with per-horizon locked models and simple backtesting
===================================================================================

What this script does:
- Loads locked artifacts: feature columns, scalers, and best-per-horizon models
- Builds deterministic forecasts for 24h, 48h, 72h using:
  - 24h → CatBoost (single-step)
  - 48h/72h → TCN family (sequence models)
- Saves forecasts to CSV/JSON with trace info (base timestamp, snapshot hash)
- Provides a rolling-origin backtesting harness for sanity checks

Usage:
  python forecast.py                 # run one-shot forecast using latest data
  python forecast.py --backtest 5    # run 5-fold rolling-origin backtesting

Inputs (expected):
- data_repositories/features/phase1_fixed_selected_features.csv
- data_repositories/features/phase1_fixed_feature_columns.pkl
- saved_models/catboost_multi_horizon_tuned_model.txt
- saved_models/tcn_optimized_48h_*.pth
- saved_models/tcn_optimized_72h_*.pth

Outputs:
- saved_models/forecasts/forecast_<timestamp>.csv
- saved_models/forecasts/forecast_<timestamp>.json
- saved_models/reports/backtest_<timestamp>.csv (when --backtest)
"""

import os
import sys
import glob
import json
import hashlib
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# Optional ML libs (lazy guarded)
try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except Exception:
    CATBOOST_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


# -----------------------
# Minimal TCN definitions (only if PyTorch available)
# -----------------------
if TORCH_AVAILABLE:
    class Chomp1d(nn.Module):
        def __init__(self, chomp_size: int):
            super().__init__()
            self.chomp_size = chomp_size

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x[:, :, :-self.chomp_size] if self.chomp_size > 0 else x


    class TemporalBlock(nn.Module):
        def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int, stride: int, dilation: int, padding: int, dropout: float = 0.2):
            super().__init__()
            self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
            self.chomp1 = Chomp1d(padding)
            self.relu1 = nn.ReLU()
            self.dropout1 = nn.Dropout(dropout)
            self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
            self.chomp2 = Chomp1d(padding)
            self.relu2 = nn.ReLU()
            self.dropout2 = nn.Dropout(dropout)
            self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1, self.conv2, self.chomp2, self.relu2, self.dropout2)
            self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
            self.relu = nn.ReLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.net(x)
            res = x if self.downsample is None else self.downsample(x)
            return self.relu(out + res)


    class TCN(nn.Module):
        def __init__(self, input_size: int, output_size: int, num_channels: List[int], kernel_size: int = 2, dropout: float = 0.2):
            super().__init__()
            layers: List[nn.Module] = []
            for i in range(len(num_channels)):
                dilation = 2 ** i
                in_ch = input_size if i == 0 else num_channels[i - 1]
                layers.append(TemporalBlock(in_ch, num_channels[i], kernel_size, stride=1, dilation=dilation, padding=(kernel_size - 1) * dilation, dropout=dropout))
            self.network = nn.Sequential(*layers)
            self.final = nn.Linear(num_channels[-1], output_size)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.network(x)
            x = torch.mean(x, dim=2)
            return self.final(x)


    class TCNLSTM(nn.Module):
        def __init__(self, input_size: int, output_size: int, num_channels: List[int], kernel_size: int = 2, lstm_hidden: int = 64, dropout: float = 0.2):
            super().__init__()
            layers: List[nn.Module] = []
            for i in range(len(num_channels)):
                dilation = 2 ** i
                in_ch = input_size if i == 0 else num_channels[i - 1]
                layers.append(TemporalBlock(in_ch, num_channels[i], kernel_size, stride=1, dilation=dilation, padding=(kernel_size - 1) * dilation, dropout=dropout))
            self.tcn = nn.Sequential(*layers)
            self.lstm = nn.LSTM(num_channels[-1], lstm_hidden, batch_first=True, dropout=dropout)
            self.out = nn.Linear(lstm_hidden, output_size)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.tcn(x)
            x = x.transpose(1, 2)
            x, _ = self.lstm(x)
            x = x[:, -1, :]
            x = self.dropout(x)
            return self.out(x)


# -----------------------
# Utilities
# -----------------------
def _load_feature_columns() -> List[str]:
    import pickle
    with open('data_repositories/features/phase1_fixed_feature_columns.pkl', 'rb') as f:
        return pickle.load(f)


def _load_features_df() -> pd.DataFrame:
    path = 'data_repositories/features/phase1_fixed_selected_features.csv'
    if not os.path.exists(path):
        raise FileNotFoundError(f'Missing required features file: {path}')
    df = pd.read_csv(path)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def _align_features(df: pd.DataFrame, required_cols: List[str]) -> pd.DataFrame:
    aligned = df.copy()
    miss = [c for c in required_cols if c not in aligned.columns]
    for c in miss:
        aligned[c] = 0.0
    return aligned[required_cols]


def _snapshot_hash(values: np.ndarray) -> str:
    m = hashlib.md5()
    m.update(values.tobytes())
    return m.hexdigest()


def _latest(path_glob: str) -> str | None:
    matches = glob.glob(path_glob)
    if not matches:
        return None
    matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return matches[0]


# -----------------------
# Model loading
# -----------------------
def load_catboost_24h() -> CatBoostRegressor:
    if not CATBOOST_AVAILABLE:
        raise RuntimeError('CatBoost not available')
    mdl_path = _latest('saved_models/catboost_multi_horizon_tuned_model.*')
    if mdl_path is None:
        raise FileNotFoundError('CatBoost 24h model not found in saved_models/')
    model = CatBoostRegressor()
    model.load_model(mdl_path)
    return model


def build_tcn_from_config(input_size: int, output_size: int, cfg: Dict):
    if not TORCH_AVAILABLE:
        raise RuntimeError('PyTorch not available')
    name = (cfg.get('name') or cfg.get('model_type') or '').lower()
    channels = cfg.get('hidden_dims') or cfg.get('channels') or [64, 32]
    kernel = cfg.get('kernel_size', 2)
    dropout = cfg.get('dropout', 0.2)
    if 'lstm' in name:
        return TCNLSTM(input_size, output_size, channels, kernel_size=kernel, lstm_hidden=cfg.get('lstm_hidden', 64), dropout=dropout)
    return TCN(input_size, output_size, channels, kernel_size=kernel, dropout=dropout)


def load_tcn_checkpoint(horizon: str) -> Tuple[object, Dict, List[str], object, int]:
    assert horizon in {'48h', '72h'}
    # Try the fine-tuned checkpoints first, then fall back to optimized ones
    ckpt_path = _latest(f'saved_models/tcn_{horizon}_finetuned.pth')
    if ckpt_path is None:
        ckpt_path = _latest(f'saved_models/tcn_optimized_{horizon}_*.pth')
    if ckpt_path is None:
        raise FileNotFoundError(f'TCN checkpoint for {horizon} not found')
    device = torch.device('cuda' if (TORCH_AVAILABLE and torch.cuda.is_available()) else 'cpu')
    
    # Add sklearn scaler to safe globals for PyTorch 2.6+
    try:
        torch.serialization.add_safe_globals(['sklearn.preprocessing._data.StandardScaler'])
    except Exception:
        pass
    
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)
    cfg = state.get('config', {})
    feat_cols = state.get('feature_columns')
    scaler = state.get('scaler')
    seq_len = int(cfg.get('sequence_length') or cfg.get('seq_len') or (72 if horizon == '72h' else 48))
    model = build_tcn_from_config(input_size=len(feat_cols), output_size=3, cfg=cfg).to(device)
    model.load_state_dict(state['model_state_dict'], strict=False)
    model.eval()
    return model, cfg, feat_cols, scaler, seq_len


# -----------------------
# Forecasting
# -----------------------
def forecast_once() -> Tuple[pd.DataFrame, Dict[str, float]]:
    out_dir = 'saved_models/forecasts'
    os.makedirs(out_dir, exist_ok=True)

    feature_columns = _load_feature_columns()
    df = _load_features_df().copy()
    # Print dataset date range used for forecasting
    try:
        if 'timestamp' in df.columns:
            print(f"📅 Forecast dataset range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    except Exception:
        pass
    df = df.dropna(subset=['target_aqi_24h', 'target_aqi_48h', 'target_aqi_72h'])  # ensure clean rows

    # Build 24h forecast (CatBoost): use the latest feature row
    metrics: Dict[str, float] = {}
    if CATBOOST_AVAILABLE:
        cat = load_catboost_24h()
        x24 = df[feature_columns].iloc[[-1]].values  # last row
        y24_pred = float(cat.predict(x24)[0])
    else:
        y24_pred = float('nan')

    # Build 48h/72h forecast (TCN): use latest sequence window
    if not TORCH_AVAILABLE:
        # Fallback: persist 24h prediction to 48h/72h when TCN is unavailable
        y48_pred, y72_pred = y24_pred, y24_pred
    else:
        try:
            model48, cfg48, feat48, scaler48, seq48 = load_tcn_checkpoint('48h')
        except Exception as e:
            print(f"⚠️ Skipping 48h TCN forecast: {e}")
            model48 = None; cfg48 = {}; feat48 = []; scaler48 = None; seq48 = 48
        try:
            model72, cfg72, feat72, scaler72, seq72 = load_tcn_checkpoint('72h')
        except Exception as e:
            print(f"⚠️ Skipping 72h TCN forecast: {e}")
            model72 = None; cfg72 = {}; feat72 = []; scaler72 = None; seq72 = 72

        # Prepare windows with feature alignment
        X48_full = _align_features(df, feat48).values if feat48 else np.empty((0, 0))
        X72_full = _align_features(df, feat72).values if feat72 else np.empty((0, 0))
        if scaler48 is not None:
            try:
                X48_full = scaler48.transform(X48_full)
            except Exception:
                pass
        if scaler72 is not None:
            try:
                X72_full = scaler72.transform(X72_full)
            except Exception:
                pass

        if model48 is None or model72 is None:
            y48_pred, y72_pred = float('nan'), float('nan')
        elif len(X48_full) < seq48 or len(X72_full) < seq72:
            y48_pred, y72_pred = float('nan'), float('nan')
        else:
            win48 = torch.tensor(X48_full[-seq48:], dtype=torch.float32).unsqueeze(0).transpose(1, 2)
            win72 = torch.tensor(X72_full[-seq72:], dtype=torch.float32).unsqueeze(0).transpose(1, 2)
            with torch.no_grad():
                out48 = model48(win48)
                out72 = model72(win72)
                # horizons are [24h, 48h, 72h] indices 0,1,2
                y48_pred = float(out48.squeeze(0).cpu().numpy()[1])
                y72_pred = float(out72.squeeze(0).cpu().numpy()[2])

    # Base timestamp and target timestamps
    ts = df['timestamp'].iloc[-1] if 'timestamp' in df.columns else pd.Timestamp.now()
    out_rows = [
        {'base_timestamp': ts, 'target_timestamp': ts + pd.Timedelta(hours=24), 'horizon': '24h', 'prediction': y24_pred},
        {'base_timestamp': ts, 'target_timestamp': ts + pd.Timedelta(hours=48), 'horizon': '48h', 'prediction': y48_pred},
        {'base_timestamp': ts, 'target_timestamp': ts + pd.Timedelta(hours=72), 'horizon': '72h', 'prediction': y72_pred},
    ]
    out_df = pd.DataFrame(out_rows)

    # Snapshot hash for traceability (features last window)
    try:
        snap_arr = df[feature_columns].tail(max(1, max([1, 0]))).values
        snap_hash = _snapshot_hash(snap_arr)
    except Exception:
        snap_hash = ''

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(out_dir, f'forecast_{stamp}.csv')
    json_path = os.path.join(out_dir, f'forecast_{stamp}.json')
    out_df.to_csv(csv_path, index=False)
    with open(json_path, 'w') as f:
        json.dump({
            'generated_at': datetime.now().isoformat(),
            'snapshot_hash': snap_hash,
            'forecasts': out_rows,
        }, f, indent=2, default=str)

    return out_df, {'saved_csv': csv_path, 'saved_json': json_path}


# -----------------------
# Rolling-origin backtest (simple)
# -----------------------
def _rmse(y, p):
    return float(np.sqrt(((y - p) ** 2).mean()))


def _r2(y, p):
    ybar = y.mean()
    ss_res = ((y - p) ** 2).sum()
    ss_tot = ((y - ybar) ** 2).sum()
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float('nan')


def backtest(folds: int = 5) -> str:
    reports_dir = 'saved_models/reports'
    os.makedirs(reports_dir, exist_ok=True)

    feature_columns = _load_feature_columns()
    df = _load_features_df().dropna(subset=['target_aqi_24h', 'target_aqi_48h', 'target_aqi_72h']).reset_index(drop=True)

    # Load models
    cat = load_catboost_24h() if CATBOOST_AVAILABLE else None
    model48, cfg48, feat48, scaler48, seq48 = load_tcn_checkpoint('48h') if TORCH_AVAILABLE else (None, {}, [], None, 48)
    model72, cfg72, feat72, scaler72, seq72 = load_tcn_checkpoint('72h') if TORCH_AVAILABLE else (None, {}, [], None, 72)

    n = len(df)
    fold_size = n // (folds + 1)
    results = []

    for k in range(1, folds + 1):
        end_idx = fold_size * (k + 0)
        if end_idx <= max(seq48, seq72) + 1:
            continue
        sub = df.iloc[:end_idx].copy()

        # 24h pred vs truth at end-1 (predict for next 24h at that point)
        y24_true = float(sub['target_aqi_24h'].iloc[-1])
        p24 = float(cat.predict(sub[feature_columns].iloc[[-1]].values)[0]) if cat is not None else float('nan')

            # 48h sequence pred vs truth
        if TORCH_AVAILABLE and len(sub) >= seq48:
            # Use current dataset features, align with checkpoint features if possible
            try:
                # First try to use checkpoint features if they exist in current dataset
                available_feat48 = [f for f in feat48 if f in sub.columns]
                if len(available_feat48) == len(feat48):
                    X48 = sub[feat48].values
                else:
                    # Fallback: use current dataset features and pad with zeros if needed
                    current_features = [col for col in sub.columns if col not in ['timestamp', 'target_aqi_24h', 'target_aqi_48h', 'target_aqi_72h', 'primary_pollutant', 'target_timestamp']]
                    X48 = sub[current_features].values
                    # Pad to match expected input size if needed
                    if X48.shape[1] < len(feat48):
                        padding = np.zeros((X48.shape[0], len(feat48) - X48.shape[1]))
                        X48 = np.hstack([X48, padding])
                    elif X48.shape[1] > len(feat48):
                        X48 = X48[:, :len(feat48)]
            except Exception as e:
                print(f"Warning: Could not prepare 48h features: {e}")
                p48 = float('nan'); y48_true = float('nan')
                
            if scaler48 is not None:
                try:
                    X48 = scaler48.transform(X48)
                except Exception:
                    pass
            t48 = torch.tensor(X48[-seq48:], dtype=torch.float32).unsqueeze(0).transpose(1, 2)
            with torch.no_grad():
                out48 = model48(t48).squeeze(0).cpu().numpy()
            p48 = float(out48[1])
            y48_true = float(sub['target_aqi_48h'].iloc[-1])
        else:
            p48 = float('nan'); y48_true = float('nan')

        # 72h sequence pred vs truth
        if TORCH_AVAILABLE and len(sub) >= seq72:
            # Use current dataset features, align with checkpoint features if possible
            try:
                # First try to use checkpoint features if they exist in current dataset
                available_feat72 = [f for f in feat72 if f in sub.columns]
                if len(available_feat72) == len(feat72):
                    X72 = sub[feat72].values
                else:
                    # Fallback: use current dataset features and pad with zeros if needed
                    current_features = [col for col in sub.columns if col not in ['timestamp', 'target_aqi_24h', 'target_aqi_48h', 'target_aqi_72h', 'primary_pollutant', 'target_timestamp']]
                    X72 = sub[current_features].values
                    # Pad to match expected input size if needed
                    if X72.shape[1] < len(feat72):
                        padding = np.zeros((X72.shape[0], len(feat72) - X72.shape[1]))
                        X72 = np.hstack([X72, padding])
                    elif X72.shape[1] > len(feat72):
                        X72 = X72[:, :len(feat72)]
            except Exception as e:
                print(f"Warning: Could not prepare 72h features: {e}")
                p72 = float('nan'); y72_true = float('nan')
                
            if scaler72 is not None:
                try:
                    X72 = scaler72.transform(X72)
                except Exception:
                    pass
            t72 = torch.tensor(X72[-seq72:], dtype=torch.float32).unsqueeze(0).transpose(1, 2)
            with torch.no_grad():
                out72 = model72(t72).squeeze(0).cpu().numpy()
            p72 = float(out72[2])
            y72_true = float(sub['target_aqi_72h'].iloc[-1])
        else:
            p72 = float('nan'); y72_true = float('nan')

    results.append({
        'fold': k,
        '24h_true': y24_true, '24h_pred': p24,
        '48h_true': y48_true, '48h_pred': p48,
        '72h_true': y72_true, '72h_pred': p72,
    })

    res_df = pd.DataFrame(results)
    # Aggregate metrics
    metrics = {
        '24h_RMSE': _rmse(res_df['24h_true'].values, res_df['24h_pred'].values),
        '24h_R2': _r2(res_df['24h_true'].values, res_df['24h_pred'].values),
        '48h_RMSE': _rmse(res_df['48h_true'].values, res_df['48h_pred'].values),
        '48h_R2': _r2(res_df['48h_true'].values, res_df['48h_pred'].values),
        '72h_RMSE': _rmse(res_df['72h_true'].values, res_df['72h_pred'].values),
        '72h_R2': _r2(res_df['72h_true'].values, res_df['72h_pred'].values),
    }

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(reports_dir, f'backtest_{stamp}.csv')
    res_df.to_csv(out_path, index=False)

    # Print quick summary
    print('\nBacktest summary:')
    for k, v in metrics.items():
        print(f'  {k}: {v:.3f}')

    return out_path


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--backtest', type=int, default=0, help='Number of rolling folds (0 to skip)')
    args = parser.parse_args()

    if args.backtest and args.backtest > 0:
        path = backtest(args.backtest)
        print(f'\nSaved backtest to: {path}')
    else:
        out_df, paths = forecast_once()
        print('\nForecasts:')
        print(out_df.to_string(index=False))
        print(f"\nSaved: {paths['saved_csv']} and {paths['saved_json']}")


if __name__ == '__main__':
    main()





# -----------------------

# Rolling-origin backtest (simple)

# -----------------------

def _rmse(y, p):

    return float(np.sqrt(((y - p) ** 2).mean()))





def _r2(y, p):

    ybar = y.mean()

    ss_res = ((y - p) ** 2).sum()

    ss_tot = ((y - ybar) ** 2).sum()

    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float('nan')





def backtest(folds: int = 5) -> str:

    reports_dir = 'saved_models/reports'

    os.makedirs(reports_dir, exist_ok=True)



    feature_columns = _load_feature_columns()

    df = _load_features_df().dropna(subset=['target_aqi_24h', 'target_aqi_48h', 'target_aqi_72h']).reset_index(drop=True)



    # Load models

    cat = load_catboost_24h() if CATBOOST_AVAILABLE else None

    model48, cfg48, feat48, scaler48, seq48 = load_tcn_checkpoint('48h') if TORCH_AVAILABLE else (None, {}, [], None, 48)

    model72, cfg72, feat72, scaler72, seq72 = load_tcn_checkpoint('72h') if TORCH_AVAILABLE else (None, {}, [], None, 72)



    n = len(df)

    fold_size = n // (folds + 1)

    results = []



    for k in range(1, folds + 1):

        end_idx = fold_size * (k + 0)

        if end_idx <= max(seq48, seq72) + 1:

            continue

        sub = df.iloc[:end_idx].copy()



        # 24h pred vs truth at end-1 (predict for next 24h at that point)

        y24_true = float(sub['target_aqi_24h'].iloc[-1])

        p24 = float(cat.predict(sub[feature_columns].iloc[[-1]].values)[0]) if cat is not None else float('nan')



        # 48h sequence pred vs truth
        if TORCH_AVAILABLE and len(sub) >= seq48:
            # Use current dataset features, align with checkpoint features if possible
            try:
                # First try to use checkpoint features if they exist in current dataset
                available_feat48 = [f for f in feat48 if f in sub.columns]
                if len(available_feat48) == len(feat48):
                    X48 = sub[feat48].values
                else:
                    # Fallback: use current dataset features and pad with zeros if needed
                    current_features = [col for col in sub.columns if col not in ['timestamp', 'target_aqi_24h', 'target_aqi_48h', 'target_aqi_72h', 'primary_pollutant', 'target_timestamp']]
                    X48 = sub[current_features].values
                    # Pad to match expected input size if needed
                    if X48.shape[1] < len(feat48):
                        padding = np.zeros((X48.shape[0], len(feat48) - X48.shape[1]))
                        X48 = np.hstack([X48, padding])
                    elif X48.shape[1] > len(feat48):
                        X48 = X48[:, :len(feat48)]
            except Exception as e:
                print(f"Warning: Could not prepare 48h features: {e}")
                p48 = float('nan'); y48_true = float('nan')
                
            if scaler48 is not None:
                try:
                    X48 = scaler48.transform(X48)
                except Exception:
                    pass
            t48 = torch.tensor(X48[-seq48:], dtype=torch.float32).unsqueeze(0).transpose(1, 2)
            with torch.no_grad():
                out48 = model48(t48).squeeze(0).cpu().numpy()
            p48 = float(out48[1])
            y48_true = float(sub['target_aqi_48h'].iloc[-1])
        else:
            p48 = float('nan'); y48_true = float('nan')



        # 72h sequence pred vs truth
        if TORCH_AVAILABLE and len(sub) >= seq72:
            # Use current dataset features, align with checkpoint features if possible
            try:
                # First try to use checkpoint features if they exist in current dataset
                available_feat72 = [f for f in feat72 if f in sub.columns]
                if len(available_feat72) == len(feat72):
                    X72 = sub[feat72].values
                else:
                    # Fallback: use current dataset features and pad with zeros if needed
                    current_features = [col for col in sub.columns if col not in ['timestamp', 'target_aqi_24h', 'target_aqi_48h', 'target_aqi_72h', 'primary_pollutant', 'target_timestamp']]
                    X72 = sub[current_features].values
                    # Pad to match expected input size if needed
                    if X72.shape[1] < len(feat72):
                        padding = np.zeros((X72.shape[0], len(feat72) - X72.shape[1]))
                        X72 = np.hstack([X72, padding])
                    elif X72.shape[1] > len(feat72):
                        X72 = X72[:, :len(feat72)]
            except Exception as e:
                print(f"Warning: Could not prepare 72h features: {e}")
                p72 = float('nan'); y72_true = float('nan')
                
            if scaler72 is not None:
                try:
                    X72 = scaler72.transform(X72)
                except Exception:
                    pass
            t72 = torch.tensor(X72[-seq72:], dtype=torch.float32).unsqueeze(0).transpose(1, 2)
            with torch.no_grad():
                out72 = model72(t72).squeeze(0).cpu().numpy()
            p72 = float(out72[2])
            y72_true = float(sub['target_aqi_72h'].iloc[-1])
        else:
            p72 = float('nan'); y72_true = float('nan')



        results.append({

            'fold': k,

            '24h_true': y24_true, '24h_pred': p24,

            '48h_true': y48_true, '48h_pred': p48,

            '72h_true': y72_true, '72h_pred': p72,

        })



    res_df = pd.DataFrame(results)

    # Aggregate metrics

    metrics = {

        '24h_RMSE': _rmse(res_df['24h_true'].values, res_df['24h_pred'].values),

        '24h_R2': _r2(res_df['24h_true'].values, res_df['24h_pred'].values),

        '48h_RMSE': _rmse(res_df['48h_true'].values, res_df['48h_pred'].values),

        '48h_R2': _r2(res_df['48h_true'].values, res_df['48h_pred'].values),

        '72h_RMSE': _rmse(res_df['72h_true'].values, res_df['72h_pred'].values),

        '72h_R2': _r2(res_df['72h_true'].values, res_df['72h_pred'].values),

    }



    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    out_path = os.path.join(reports_dir, f'backtest_{stamp}.csv')

    res_df.to_csv(out_path, index=False)



    # Print quick summary

    print('\nBacktest summary:')

    for k, v in metrics.items():

        print(f'  {k}: {v:.3f}')



    return out_path





def main():

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--backtest', type=int, default=0, help='Number of rolling folds (0 to skip)')

    args = parser.parse_args()



    if args.backtest and args.backtest > 0:

        path = backtest(args.backtest)

        print(f'\nSaved backtest to: {path}')

    else:

        out_df, paths = forecast_once()

        print('\nForecasts:')

        print(out_df.to_string(index=False))

        print(f"\nSaved: {paths['saved_csv']} and {paths['saved_json']}")





if __name__ == '__main__':

    main()






