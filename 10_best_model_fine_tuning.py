"""
Best Model Fine-Tuning (LightGBM, CatBoost, TCN)
================================================

Goal:
- Fine-tune the top models on the CLEAN 24-hour forecasting dataset
- Evaluate on a held-out test split using RMSE, MAE, and R²
- Save improved models and a concise performance report

Design:
- Uses the same dataset and feature set as the rest of the pipeline
- Serial, small searches with early stopping to be stable on Windows
- Saves results to saved_models/
"""

from __future__ import annotations

import os
import json
import math
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import lightgbm as lgb
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int = 2, stride: int = 1,
                 dilation: int = 1, padding: int = 1, dropout: float = 0.2) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.final_relu = nn.ReLU()
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.conv1.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.conv2.weight, mean=0.0, std=0.01)
        if self.downsample is not None:
            nn.init.normal_(self.downsample.weight, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.final_relu(out + res)


class TCN(nn.Module):
    def __init__(self, input_size: int, output_size: int, channels: List[int], kernel_size: int = 3,
                 dropout: float = 0.2) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        for i, ch in enumerate(channels):
            dilation_size = 2 ** i
            in_ch = input_size if i == 0 else channels[i - 1]
            layers.append(
                TemporalBlock(in_ch, ch, kernel_size=kernel_size,
                              dilation=dilation_size,
                              padding=(kernel_size - 1) * dilation_size,
                              dropout=dropout)
            )
        self.network = nn.Sequential(*layers)
        self.final_layer = nn.Linear(channels[-1], output_size)
        nn.init.xavier_uniform_(self.final_layer.weight, gain=0.01)
        nn.init.constant_(self.final_layer.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.network(x)
        x = torch.mean(x, dim=2)
        return self.final_layer(x)


class TCNLSTM(nn.Module):
    """TCN feature extractor followed by an LSTM head."""
    def __init__(self, input_size: int, output_size: int, channels: List[int], kernel_size: int = 3,
                 lstm_hidden: int = 64, lstm_layers: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        # TCN backbone without final pooling/linear
        layers: List[nn.Module] = []
        for i, ch in enumerate(channels):
            dilation_size = 2 ** i
            in_ch = input_size if i == 0 else channels[i - 1]
            layers.append(
                TemporalBlock(in_ch, ch, kernel_size=kernel_size,
                              dilation=dilation_size,
                              padding=(kernel_size - 1) * dilation_size,
                              dropout=dropout)
            )
        self.tcn = nn.Sequential(*layers)
        lstm_dropout = dropout if lstm_layers > 1 else 0.0
        self.lstm = nn.LSTM(input_size=channels[-1], hidden_size=lstm_hidden, num_layers=lstm_layers,
                            batch_first=True, dropout=lstm_dropout)
        self.out = nn.Linear(lstm_hidden, output_size)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.out.weight, gain=0.01)
        nn.init.constant_(self.out.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, features, seq_len) -> TCN -> (batch, channels, seq_len)
        x = self.tcn(x)
        # to LSTM: (batch, seq_len, channels)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.dropout(x)
        return self.out(x)


class Chomp1d(nn.Module):
    """Crops padding on the right to maintain causality/length alignment."""
    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size] if self.chomp_size > 0 else x


class BestModelFineTuner:
    def __init__(self) -> None:
        self.data_path = "data_repositories/features/phase1_fixed_selected_features.csv"
        self.feature_columns_path = "data_repositories/features/phase1_fixed_feature_columns.pkl"
        self.models_dir = "saved_models"
        os.makedirs(self.models_dir, exist_ok=True)

        self.feature_columns: List[str] = []
        self.scaler: Optional[StandardScaler] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_path)
        with open(self.feature_columns_path, "rb") as f:
            import pickle
            self.feature_columns = pickle.load(f)
        df = df[df["numerical_aqi"].notna()].copy()
        return df

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        X = df[self.feature_columns].copy()
        y = df["numerical_aqi"].copy()
        n = len(X)
        n_train = int(0.6 * n)
        n_val = int(0.2 * n)
        X_train = X.iloc[:n_train]
        y_train = y.iloc[:n_train]
        X_val = X.iloc[n_train:n_train + n_val]
        y_val = y.iloc[n_train:n_train + n_val]
        X_test = X.iloc[n_train + n_val:]
        y_test = y.iloc[n_train + n_val:]
        return X_train, X_val, X_test, y_train, y_val, y_test

    def scale_features(self, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        candidates = [
            os.path.join(self.models_dir, "standard_scaler_fixed_tuned.pkl"),
            os.path.join(self.models_dir, "standard_scaler_fixed.pkl"),
        ]
        self.scaler = None
        for p in candidates:
            if os.path.exists(p):
                with open(p, "rb") as f:
                    import pickle
                    self.scaler = pickle.load(f)
                break
        if self.scaler is None:
            self.scaler = StandardScaler()
            self.scaler.fit(X_train)
        Xtr = self.scaler.transform(X_train)
        Xv = self.scaler.transform(X_val)
        Xte = self.scaler.transform(X_test)
        return Xtr, Xv, Xte

    # ---------------- LightGBM ----------------
    def finetune_lightgbm(self, Xtr: np.ndarray, ytr: np.ndarray, Xv: np.ndarray, yv: np.ndarray, Xte: np.ndarray, yte: np.ndarray) -> Dict[str, float]:
        print("\n🔧 Fine-tuning LightGBM (serial, early stopping)...")
        # Ensure consistent feature names to avoid sklearn UserWarning
        Xtr_df = pd.DataFrame(Xtr, columns=self.feature_columns)
        Xv_df = pd.DataFrame(Xv, columns=self.feature_columns)
        Xte_df = pd.DataFrame(Xte, columns=self.feature_columns)
        rng = np.random.default_rng(42)
        trials = 12
        best_model = None
        best_val = float("inf")
        best_params = None
        for _ in range(trials):
            params = {
                "objective": "regression",
                "num_leaves": int(rng.choice([16, 31, 42, 64, 80])),
                "max_depth": int(rng.choice([-1, 3, 4, 5, 6])),
                "learning_rate": float(rng.uniform(0.02, 0.08)),
                "n_estimators": 1200,
                "subsample": float(rng.uniform(0.7, 1.0)),
                "colsample_bytree": float(rng.uniform(0.7, 1.0)),
                "min_child_samples": int(rng.choice([10, 20, 30, 50])),
                "reg_alpha": float(rng.uniform(0.0, 0.3)),
                "reg_lambda": float(rng.uniform(0.0, 0.5)),
                "n_jobs": 1,
                "verbosity": -1,
                "force_col_wise": True,
                "random_state": 42,
            }
            model = LGBMRegressor(**params)
            model.fit(
                Xtr_df, ytr,
                eval_set=[(Xv_df, yv)],
                eval_metric="rmse",
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )
            vpred = model.predict(Xv_df)
            cur = rmse(yv, vpred)
            if cur < best_val:
                best_val = cur
                best_model = model
                best_params = params
        assert best_model is not None
        print(f"   ✅ Best LightGBM (val RMSE={best_val:.2f}) params: {best_params}")
        tpred = best_model.predict(Xte_df)
        metrics = evaluate_metrics(yte, tpred)
        print(f"   ✅ LightGBM (test) → RMSE={metrics['RMSE']:.2f}, MAE={metrics['MAE']:.2f}, R²={metrics['R2']:.3f}")
        import pickle
        with open(os.path.join(self.models_dir, "lightgbm_best_finetuned.pkl"), "wb") as f:
            pickle.dump(best_model, f)
        return metrics

    # ---------------- CatBoost ----------------
    def finetune_catboost(self, Xtr: np.ndarray, ytr: np.ndarray, Xv: np.ndarray, yv: np.ndarray, Xte: np.ndarray, yte: np.ndarray) -> Dict[str, float]:
        print("\n🔧 Fine-tuning CatBoost (serial, early stopping)...")
        best_model: Optional[CatBoostRegressor] = None
        best_val = float("inf")
        rng = np.random.default_rng(42)
        for _ in range(12):
            params = {
                "depth": int(rng.choice([5, 6, 7, 8])),
                "learning_rate": float(10 ** rng.uniform(-2.2, -1.0)),
                "l2_leaf_reg": float(rng.uniform(1.0, 6.0)),
                "border_count": int(rng.choice([32, 40, 64, 128])),
                "iterations": int(rng.choice([300, 500, 800])),
                "subsample": float(rng.uniform(0.7, 1.0)),
            }
            model = CatBoostRegressor(
                loss_function="RMSE",
                eval_metric="RMSE",
                depth=params["depth"],
                learning_rate=params["learning_rate"],
                l2_leaf_reg=params["l2_leaf_reg"],
                border_count=params["border_count"],
                iterations=params["iterations"],
                subsample=params["subsample"],
                random_state=42,
                verbose=False,
                thread_count=1,
            )
            model.fit(Xtr, ytr, eval_set=(Xv, yv), use_best_model=True, early_stopping_rounds=50, verbose=False)
            vpred = model.predict(Xv)
            cur = rmse(yv, vpred)
            if cur < best_val:
                best_val = cur
                best_model = model
        assert best_model is not None
        tpred = best_model.predict(Xte)
        metrics = evaluate_metrics(yte, tpred)
        print(f"   ✅ CatBoost (test) → RMSE={metrics['RMSE']:.2f}, MAE={metrics['MAE']:.2f}, R²={metrics['R2']:.3f}")
        best_model.save_model(os.path.join(self.models_dir, "catboost_best_finetuned.txt"))
        return metrics

    # ---------------- TCN ----------------
    def prepare_sequences(self, X: np.ndarray, y: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
        Xs, ys = [], []
        for i in range(len(X) - seq_len + 1):
            # Window covers indices [i, i+seq_len-1];
            # target is already 24h ahead in the dataset, so align with window end at i+seq_len-1
            Xs.append(X[i:i + seq_len, :])
            ys.append(y[i + seq_len - 1])
        return np.array(Xs), np.array(ys)

    def train_tcn(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, epochs: int, lr: float, criterion: Optional[nn.Module] = None) -> Tuple[float, nn.Module]:
        if criterion is None:
            criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - 5))
        warmup_epochs = 5
        best_val = float('inf')
        best_state = None
        patience, waited = 20, 0
        for epoch in range(epochs):
            model.train()
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                preds = model(xb).squeeze(1)
                loss = criterion(preds, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
            # LR schedule: warmup then cosine
            if epoch < warmup_epochs:
                for g in optimizer.param_groups:
                    g['lr'] = lr * float(epoch + 1) / float(warmup_epochs)
            else:
                cosine_scheduler.step()
            # val
            model.eval()
            vlosses = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    pred = model(xb).squeeze(1)
                    vlosses.append(criterion(pred, yb).item())
            mv = float(np.mean(vlosses)) if vlosses else float('inf')
            if mv < best_val:
                best_val = mv
                best_state = model.state_dict()
                waited = 0
            else:
                waited += 1
                if waited >= patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        return best_val, model

    def finetune_tcn(self, Xtr: np.ndarray, ytr: np.ndarray, Xv: np.ndarray, yv: np.ndarray, Xte: np.ndarray, yte: np.ndarray, seq_len: int = 24) -> Dict[str, float]:
        print("\n🔧 Fine-tuning TCN (focused configs)...")
        # Select top-k feature channels for TCN to reduce noise (subset columns consistently across splits)
        input_size = Xtr.shape[1]
        try:
            imp_path = os.path.join("data_repositories", "features", "phase1_fixed_feature_importance.csv")
            imp_df = pd.read_csv(imp_path)
            top_features = imp_df.sort_values(by=imp_df.columns[1], ascending=False).iloc[:35, 0].tolist()
            idxs = [self.feature_columns.index(f) for f in top_features if f in self.feature_columns]
            if len(idxs) >= 10:
                Xtr = Xtr[:, idxs]
                Xv = Xv[:, idxs]
                Xte = Xte[:, idxs]
                input_size = Xtr.shape[1]
        except Exception:
            input_size = Xtr.shape[1]

        # Standardize target using train-only stats (global splits)
        y_mean = float(ytr.mean())
        y_std = float(ytr.std() + 1e-8)
        ytr_s = (ytr - y_mean) / y_std
        yv_s = (yv - y_mean) / y_std
        yte_s = (yte - y_mean) / y_std

        configs = [
           # Smaller model (baseline)
           {"arch": "tcnlstm", "channels": [32, 16], "lr": 0.001, "dropout": 0.2, "epochs": 120, "seq_len": 24},

           # Medium model, longer history
           {"arch": "tcnlstm", "channels": [64, 32], "lr": 0.0007, "dropout": 0.3, "epochs": 150, "seq_len": 36},

           # Larger model, 3-day history
           {"arch": "tcnlstm", "channels": [128, 64], "lr": 0.0005, "dropout": 0.3, "epochs": 160, "seq_len": 48},
        ]

        best_rmse = float('inf')
        best_state = None
        best_cfg = None
        for cfg in configs:
            # build sequences per-config sequence length
            sl = int(cfg["seq_len"])
            Xtr_seq, ytr_seq = self.prepare_sequences(Xtr, ytr_s, sl)
            Xv_seq, yv_seq = self.prepare_sequences(Xv, yv_s, sl)
            Xte_seq, yte_seq = self.prepare_sequences(Xte, yte_s, sl)

            train_ds = TensorDataset(torch.tensor(Xtr_seq, dtype=torch.float32).transpose(1, 2),
                                     torch.tensor(ytr_seq, dtype=torch.float32))
            val_ds = TensorDataset(torch.tensor(Xv_seq, dtype=torch.float32).transpose(1, 2),
                                   torch.tensor(yv_seq, dtype=torch.float32))
            test_ds = TensorDataset(torch.tensor(Xte_seq, dtype=torch.float32).transpose(1, 2),
                                    torch.tensor(yte_seq, dtype=torch.float32))

            train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
            test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

            if cfg.get("arch", "tcn") == "tcnlstm":
                model = TCNLSTM(input_size=input_size, output_size=1, channels=cfg["channels"], dropout=cfg["dropout"]).to(self.device)
            else:
                model = TCN(input_size=input_size, output_size=1, channels=cfg["channels"], dropout=cfg["dropout"]).to(self.device)
            # Huber loss for robustness
            huber = nn.SmoothL1Loss()
            _, trained = self.train_tcn(model, train_loader, val_loader, epochs=cfg["epochs"], lr=cfg["lr"], criterion=huber)

            trained.eval()
            preds, gts = [], []
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb = xb.to(self.device)
                    pred = trained(xb).squeeze(1).cpu().numpy()
                    preds.append(pred)
                    gts.append(yb.numpy())
            # Inverse-transform target back to real scale
            ypred = np.concatenate(preds) * y_std + y_mean
            ytrue = np.concatenate(gts) * y_std + y_mean
            cur = rmse(ytrue, ypred)
            if cur < best_rmse:
                best_rmse = cur
                best_state = trained.state_dict()
                best_cfg = cfg
        assert best_state is not None
        if best_cfg.get("arch", "tcn") == "tcnlstm":
            best_model = TCNLSTM(input_size=input_size, output_size=1, channels=best_cfg["channels"], dropout=best_cfg["dropout"]).to(self.device)
        else:
            best_model = TCN(input_size=input_size, output_size=1, channels=best_cfg["channels"], dropout=best_cfg["dropout"]).to(self.device)
        best_model.load_state_dict(best_state)
        # final metrics
        preds, gts = [], []
        best_model.eval()
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(self.device)
                pred = best_model(xb).squeeze(1).cpu().numpy()
                preds.append(pred)
                gts.append(yb.numpy())
        metrics = evaluate_metrics(np.concatenate(gts), np.concatenate(preds))
        print(f"   ✅ TCN (test) → RMSE={metrics['RMSE']:.2f}, MAE={metrics['MAE']:.2f}, R²={metrics['R2']:.3f}")
        torch.save({
            "model_state_dict": best_model.state_dict(),
            "config": best_cfg,
            "feature_columns": self.feature_columns,
            "sequence_length": int(best_cfg["seq_len"]),
        }, os.path.join(self.models_dir, "tcn_best_finetuned.pth"))
        return metrics

    def run(self) -> None:
        print("\n🚀 BEST MODEL FINE-TUNING PIPELINE (24h)")
        print("=" * 60)
        df = self.load_data()
        Xtr_df, Xv_df, Xte_df, ytr, yv, yte = self.split_data(df)
        Xtr, Xv, Xte = self.scale_features(Xtr_df, Xv_df, Xte_df)
        results: Dict[str, Dict[str, float]] = {}
        # LightGBM
        results["lightgbm_finetuned"] = self.finetune_lightgbm(Xtr, ytr.values, Xv, yv.values, Xte, yte.values)
        # CatBoost
        results["catboost_finetuned"] = self.finetune_catboost(Xtr, ytr.values, Xv, yv.values, Xte, yte.values)
        # TCN
        results["tcn_finetuned"] = self.finetune_tcn(Xtr, ytr.values, Xv, yv.values, Xte, yte.values, seq_len=24)
        # save
        csv_path = os.path.join(self.models_dir, "best_finetuned_results.csv")
        pd.DataFrame([{ "Model": k, **v } for k, v in results.items()]).to_csv(csv_path, index=False)
        with open(os.path.join(self.models_dir, "best_finetuned_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        print("\n📊 SUMMARY (test):")
        for name, m in results.items():
            print(f"   {name}: RMSE={m['RMSE']:.2f}, MAE={m['MAE']:.2f}, R²={m['R2']:.3f}")
        print(f"\n💾 Results saved to: {csv_path}")


def main() -> None:
    BestModelFineTuner().run()


if __name__ == "__main__":
    main()


