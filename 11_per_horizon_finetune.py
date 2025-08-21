#!/usr/bin/env python3
"""
Per-Horizon Fine-Tuning Pipeline (24h CatBoost, 48h/72h TCN)
============================================================

What this does:
- Loads the multi-horizon dataset (no leakage) and feature artifacts
- For 24h: fine-tunes CatBoost with small, stable search + early stopping
- For 48h and 72h: loads the best TCN checkpoints saved from 08 and fine-tunes
  with a low LR and horizon-focused loss weighting
- Saves improved checkpoints and a concise per-horizon metrics report

Inputs (required):
- data_repositories/features/phase1_fixed_selected_features.csv
- data_repositories/features/phase1_fixed_feature_columns.pkl
- data_repositories/features/phase1_fixed_feature_scaler.pkl

Outputs:
- saved_models/catboost_24h_finetuned.txt
- saved_models/tcn_48h_finetuned.pth
- saved_models/tcn_72h_finetuned.pth
- saved_models/per_horizon_finetune_results.csv
"""

import os
import glob
import json
import math
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except Exception:
    CATBOOST_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        'RMSE': rmse(y_true, y_pred),
        'MAE': float(mean_absolute_error(y_true, y_pred)),
        'R2': float(r2_score(y_true, y_pred)),
    }


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
            self._init()

        def _init(self):
            nn.init.normal_(self.conv1.weight, mean=0.0, std=0.01)
            nn.init.normal_(self.conv2.weight, mean=0.0, std=0.01)
            if self.downsample is not None:
                nn.init.normal_(self.downsample.weight, mean=0.0, std=0.01)

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


class PerHorizonFineTuner:
    def __init__(self) -> None:
        self.data_path = "data_repositories/features/phase1_fixed_selected_features.csv"
        self.feature_cols_path = "data_repositories/features/phase1_fixed_feature_columns.pkl"
        self.scaler_path = "data_repositories/features/phase1_fixed_feature_scaler.pkl"
        self.models_dir = "saved_models"
        os.makedirs(self.models_dir, exist_ok=True)

        self.df: pd.DataFrame = pd.DataFrame()
        self.feature_columns: List[str] = []
        self.scaler: StandardScaler = None  # type: ignore

        if TORCH_AVAILABLE:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = 'cpu'

    def _align_features(self, df: pd.DataFrame, required_cols: List[str]) -> pd.DataFrame:
        """Ensure all required columns exist; add missing with zeros and preserve order."""
        aligned = df.copy()
        missing = [c for c in required_cols if c not in aligned.columns]
        for c in missing:
            aligned[c] = 0.0
        # Reorder to required order
        return aligned[required_cols]

    def _safe_scale(self, Xtr: pd.DataFrame, Xv: pd.DataFrame, Xte: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Scale splits with existing scaler when shapes match; otherwise fit a new scaler on Xtr."""
        try:
            from sklearn.preprocessing import StandardScaler
        except Exception:
            # If sklearn unavailable, return as is
            return Xtr.values, Xv.values, Xte.values

        if isinstance(self.scaler, StandardScaler):
            try:
                # Only use if number of features matches
                if getattr(self.scaler, 'n_features_in_', None) == Xtr.shape[1]:
                    return (
                        self.scaler.transform(Xtr),
                        self.scaler.transform(Xv),
                        self.scaler.transform(Xte),
                    )
            except Exception:
                pass

        # Fallback: fit a temporary scaler on training split
        temp_scaler = StandardScaler()
        Xtr_s = temp_scaler.fit_transform(Xtr)
        Xv_s = temp_scaler.transform(Xv)
        Xte_s = temp_scaler.transform(Xte)
        return Xtr_s, Xv_s, Xte_s

    def load_artifacts(self) -> None:
        with open(self.feature_cols_path, 'rb') as f:
            import pickle
            self.feature_columns = pickle.load(f)
        with open(self.scaler_path, 'rb') as f:
            import pickle
            self.scaler = pickle.load(f)

    def load_data(self) -> None:
        df = pd.read_csv(self.data_path)
        required = ['target_aqi_24h', 'target_aqi_48h', 'target_aqi_72h']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise RuntimeError(f"Missing required target columns: {missing}")
        mask = df['target_aqi_24h'].notna() & df['target_aqi_48h'].notna() & df['target_aqi_72h'].notna()
        self.df = df.loc[mask].reset_index(drop=True)
        try:
            # Print dataset date range used for fine-tuning
            if 'timestamp' in df.columns:
                ts_min = pd.to_datetime(df['timestamp']).min()
                ts_max = pd.to_datetime(df['timestamp']).max()
                print(f"📅 Fine-tune dataset range: {ts_min} → {ts_max}")
        except Exception:
            pass

    def split(self, X: pd.DataFrame, y: pd.Series | pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series | pd.DataFrame, pd.Series | pd.DataFrame, pd.Series | pd.DataFrame]:
        n = len(X)
        n_tr = int(0.6 * n)
        n_va = int(0.2 * n)
        Xtr, Xv, Xte = X.iloc[:n_tr], X.iloc[n_tr:n_tr + n_va], X.iloc[n_tr + n_va:]
        ytr, yv, yte = y.iloc[:n_tr], y.iloc[n_tr:n_tr + n_va], y.iloc[n_tr + n_va:]
        return Xtr, Xv, Xte, ytr, yv, yte

    # ---------------- 24h: CatBoost ----------------
    def finetune_catboost_24h(self) -> Dict[str, float]:
        if not CATBOOST_AVAILABLE:
            print("❌ CatBoost not available; skipping 24h fine-tune")
            return {"RMSE": float('nan'), "MAE": float('nan'), "R2": float('nan')}

        X = self.df[self.feature_columns]
        y = self.df['target_aqi_24h']
        Xtr, Xv, Xte, ytr, yv, yte = self.split(X, y)
        Xtr_s = self.scaler.transform(Xtr)
        Xv_s = self.scaler.transform(Xv)
        Xte_s = self.scaler.transform(Xte)

        best_model = None
        best_val = float('inf')
        rng = np.random.default_rng(42)
        trials = 10
        for _ in range(trials):
            params = {
                'depth': int(rng.choice([5, 6, 7, 8])),
                'learning_rate': float(10 ** rng.uniform(-2.2, -1.1)),
                'l2_leaf_reg': float(rng.uniform(1.0, 6.0)),
                'iterations': int(rng.choice([400, 600, 800])),
                'subsample': float(rng.uniform(0.7, 1.0)),
            }
            model = CatBoostRegressor(
                loss_function='RMSE',
                eval_metric='RMSE',
                depth=params['depth'],
                learning_rate=params['learning_rate'],
                l2_leaf_reg=params['l2_leaf_reg'],
                iterations=params['iterations'],
                subsample=params['subsample'],
                random_state=42,
                verbose=False,
                thread_count=1,
            )
            model.fit(Xtr_s, ytr, eval_set=(Xv_s, yv), use_best_model=True, early_stopping_rounds=50, verbose=False)
            val_pred = model.predict(Xv_s)
            cur = rmse(yv.values, val_pred)
            if cur < best_val:
                best_val = cur
                best_model = model

        if best_model is None:
            return {"RMSE": float('nan'), "MAE": float('nan'), "R2": float('nan')}

        te_pred = best_model.predict(Xte_s)
        metrics = evaluate(yte.values, te_pred)
        best_model.save_model(os.path.join(self.models_dir, 'catboost_24h_finetuned.txt'))
        return metrics

    # ---------------- TCN utils ----------------
    def _latest_checkpoint(self, pattern: str) -> str | None:
        paths = glob.glob(pattern)
        if not paths:
            return None
        paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return paths[0]

    def _prepare_sequences(self, X: np.ndarray, y: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
        Xs, ys = [], []
        for i in range(len(X) - seq_len + 1):
            Xs.append(X[i:i + seq_len, :])
            ys.append(y[i + seq_len - 1])
        return np.array(Xs), np.array(ys)

    def _weighted_mse_loss(self, focus_idx: int, weights: Tuple[float, float, float] = (1.0, 1.0, 1.8)):
        # Increase weight for focused horizon (default heavier on index 2 → 72h)
        w = list(weights)
        w = [1.0, 1.0, 1.0]
        w[focus_idx] = 2.0
        w = torch.tensor(w, dtype=torch.float32, device=self.device)

        def loss_fn(pred, target):
            # pred/target: (batch, 3)
            diff = (pred - target) ** 2
            return torch.mean(diff * w)

        return loss_fn

    def _build_tcn_from_config(self, input_size: int, output_size: int, cfg: Dict):
        name = (cfg.get('name') or cfg.get('model_type') or '').lower()
        channels = cfg.get('hidden_dims') or cfg.get('channels') or [64, 32]
        dropout = cfg.get('dropout', 0.2)
        kernel_size = cfg.get('kernel_size', 2)
        if TORCH_AVAILABLE:
            if 'lstm' in name:
                model = TCNLSTM(input_size, output_size, channels, kernel_size=kernel_size, lstm_hidden=cfg.get('lstm_hidden', 64), dropout=dropout)
            else:
                model = TCN(input_size, output_size, channels, kernel_size=kernel_size, dropout=dropout)
            return model.to(self.device)
        raise RuntimeError("PyTorch not available")

    def _finetune_tcn_multi(self, horizon: str) -> Dict[str, float]:
        assert horizon in {'48h', '72h'}
        focus_idx = {'24h': 0, '48h': 1, '72h': 2}[horizon]
        ckpt = self._latest_checkpoint(os.path.join(self.models_dir, f"tcn_optimized_{horizon}_*.pth"))
        if ckpt is None:
            print(f"❌ No TCN checkpoint found for {horizon}")
            return {"RMSE": float('nan'), "MAE": float('nan'), "R2": float('nan')}

        # Load checkpoint
        # Robust checkpoint load: PyTorch 2.6 defaults to weights_only=True which breaks pickled objects (e.g., scalers)
        try:
            # Add sklearn scaler to safe globals for PyTorch 2.6+
            try:
                torch.serialization.add_safe_globals(['sklearn.preprocessing._data.StandardScaler'])
            except Exception:
                pass
            state = torch.load(ckpt, map_location=self.device, weights_only=False)
        except TypeError:
            # Older PyTorch without weights_only arg
            state = torch.load(ckpt, map_location=self.device)
        cfg = state.get('config', {})
        feat_cols = state.get('feature_columns') or self.feature_columns
        # Align to checkpoint feature columns to avoid KeyError on drift
        X = self._align_features(self.df, feat_cols)
        y = self.df[['target_aqi_24h', 'target_aqi_48h', 'target_aqi_72h']].values
        Xtr_df, Xv_df, Xte_df, ytr, yv, yte = self.split(X, pd.DataFrame(y, columns=['24h', '48h', '72h']))
        Xtr_s, Xv_s, Xte_s = self._safe_scale(Xtr_df, Xv_df, Xte_df)

        # sequences with longer history if present in cfg; else default 48 for long horizon
        seq_len = int(cfg.get('sequence_length') or cfg.get('seq_len') or (72 if horizon == '72h' else 48))
        Xtr_seq, ytr_seq = self._prepare_sequences(Xtr_s, ytr.values, seq_len)
        Xv_seq, yv_seq = self._prepare_sequences(Xv_s, yv.values, seq_len)
        Xte_seq, yte_seq = self._prepare_sequences(Xte_s, yte.values, seq_len)

        train_ds = TensorDataset(torch.tensor(Xtr_seq, dtype=torch.float32).transpose(1, 2), torch.tensor(ytr_seq, dtype=torch.float32))
        val_ds = TensorDataset(torch.tensor(Xv_seq, dtype=torch.float32).transpose(1, 2), torch.tensor(yv_seq, dtype=torch.float32))
        test_ds = TensorDataset(torch.tensor(Xte_seq, dtype=torch.float32).transpose(1, 2), torch.tensor(yte_seq, dtype=torch.float32))
        bs = 64 if torch.cuda.is_available() else 32
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=bs)
        test_loader = DataLoader(test_ds, batch_size=bs)

        input_size = Xtr_seq.shape[2]
        output_size = 3
        model = self._build_tcn_from_config(input_size, output_size, cfg)
        model.load_state_dict(state['model_state_dict'], strict=False)

        # Fine-tune: low LR, horizon-focused weighted MSE, early stopping
        base_lr = float(cfg.get('lr', 0.001))
        lr = max(1e-5, base_lr / 5.0)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5)
        criterion = self._weighted_mse_loss(focus_idx)

        best_val = float('inf')
        best_state = None
        patience, waited = 20, 0
        epochs = int(cfg.get('epochs', 150)) + 40

        for epoch in range(epochs):
            model.train()
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.7)
                optimizer.step()

            # validation
            model.eval()
            vlosses = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    out = model(xb)
                    vlosses.append(criterion(out, yb).item())
            vmean = float(np.mean(vlosses)) if vlosses else float('inf')
            scheduler.step(vmean)
            if vmean < best_val:
                best_val = vmean
                best_state = model.state_dict()
                waited = 0
            else:
                waited += 1
                if waited >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        # evaluate on test, report metrics for the focused horizon only
        preds, trues = [], []
        model.eval()
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(self.device)
                out = model(xb).cpu().numpy()
                preds.append(out[:, focus_idx])
                trues.append(yb.cpu().numpy()[:, focus_idx])
        y_pred = np.concatenate(preds)
        y_true = np.concatenate(trues)
        metrics = evaluate(y_true, y_pred)

        out_path = os.path.join(self.models_dir, f"tcn_{horizon}_finetuned.pth")
        torch.save({'model_state_dict': model.state_dict(), 'config': cfg, 'feature_columns': feat_cols, 'sequence_length': seq_len}, out_path)
        return metrics

    def run(self) -> None:
        print("\n🚀 PER-HORIZON FINE-TUNING (24h CatBoost, 48h/72h TCN)")
        print("=" * 70)
        self.load_artifacts()
        self.load_data()

        results: Dict[str, Dict[str, float]] = {}

        # 24h → CatBoost
        print("\n🎯 Fine-tuning 24h (CatBoost)...")
        results['catboost_24h_finetuned'] = self.finetune_catboost_24h()

        # 48h → TCN
        if TORCH_AVAILABLE:
            print("\n🎯 Fine-tuning 48h (TCN)...")
            results['tcn_48h_finetuned'] = self._finetune_tcn_multi('48h')

            print("\n🎯 Fine-tuning 72h (TCN)...")
            results['tcn_72h_finetuned'] = self._finetune_tcn_multi('72h')
        else:
            print("❌ PyTorch not available; skipping 48h/72h TCN fine-tuning")

        # Save report
        out_csv = os.path.join(self.models_dir, 'per_horizon_finetune_results.csv')
        pd.DataFrame([{ 'Model': k, **v } for k, v in results.items()]).to_csv(out_csv, index=False)
        with open(os.path.join(self.models_dir, 'per_horizon_finetune_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        print("\n📊 Fine-tuning summary (test):")
        for name, m in results.items():
            print(f"   {name}: RMSE={m['RMSE']:.2f}, MAE={m['MAE']:.2f}, R²={m['R2']:.3f}")
        print(f"\n💾 Results saved to: {out_csv}")


def main() -> None:
    PerHorizonFineTuner().run()


if __name__ == '__main__':
    main()


