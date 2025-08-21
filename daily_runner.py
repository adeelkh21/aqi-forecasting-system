#!/usr/bin/env python3
"""
Daily Runner: data update → checks → fine-tune → gate → forecast
================================================================

What this does (idempotent, safe by default):
- Ensures required artifacts and directories exist
- Optionally refreshes features (calls preprocessing/feature selection if missing)
- Runs basic data quality checks
- Triggers per-horizon fine-tuning (24h CatBoost, 48h/72h TCN) via 11_per_horizon_finetune.py
- Applies a promotion gate (do-no-harm) using a simple registry in saved_models/registry/models.json
- Generates forecasts via forecast.py

Notes:
- On first run, a registry is created and no promotion happens (to be conservative).
- You can edit thresholds in the registry file after first run.
"""

import os
import json
import shutil
import subprocess
from datetime import datetime


ROOT = os.path.dirname(os.path.abspath(__file__))
SAVED_MODELS = os.path.join(ROOT, 'saved_models')
REGISTRY_DIR = os.path.join(SAVED_MODELS, 'registry')
FORECASTS_DIR = os.path.join(SAVED_MODELS, 'forecasts')
REPORTS_DIR = os.path.join(SAVED_MODELS, 'reports')
CHAMPIONS_DIR = os.path.join(SAVED_MODELS, 'champions')

FEATURES_CSV = os.path.join(ROOT, 'data_repositories', 'features', 'phase1_fixed_selected_features.csv')
FEATURE_COLS_PKL = os.path.join(ROOT, 'data_repositories', 'features', 'phase1_fixed_feature_columns.pkl')


def _venv_python() -> str:
    # Prefer venv under project root
    win_path = os.path.join(ROOT, 'venv', 'Scripts', 'python.exe')
    nix_path = os.path.join(ROOT, 'venv', 'bin', 'python')
    if os.path.exists(win_path):
        return win_path
    if os.path.exists(nix_path):
        return nix_path
    # Fallback to current interpreter
    return os.environ.get('PYTHON_EXECUTABLE') or 'python'


def run_cmd(args, cwd=ROOT) -> int:
    try:
        py = _venv_python()
        # Replace leading 'python'/'python3' with resolved interpreter
        if args and args[0] in ('python', 'python3'):
            args = [py] + args[1:]
        print(f"$ {' '.join(args)} (using {py})")
        return subprocess.call(args, cwd=cwd)
    except Exception as e:
        print(f"❌ Failed to run: {' '.join(args)} | {e}")
        return 1


def ensure_dirs():
    os.makedirs(SAVED_MODELS, exist_ok=True)
    os.makedirs(REGISTRY_DIR, exist_ok=True)
    os.makedirs(FORECASTS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(CHAMPIONS_DIR, exist_ok=True)


def features_available() -> bool:
    return os.path.exists(FEATURES_CSV) and os.path.exists(FEATURE_COLS_PKL)


def refresh_features_if_missing():
    if features_available():
        print("✅ Features present – skipping refresh")
        return
    print("⚠️ Features missing – attempting to regenerate via preprocessing")
    # Try to run preprocessing then feature selection, ignoring failures gracefully
    run_cmd(['python', '02_data_preprocessing.py'])
    run_cmd(['python', '03_feature_selection.py'])
    if features_available():
        print("✅ Features regenerated")
    else:
        print("❌ Could not regenerate features; proceeding without promotion")


def print_features_date_range(prefix: str = "") -> None:
    """Print min/max timestamp of the features CSV used by the pipeline."""
    try:
        import pandas as pd
        if os.path.exists(FEATURES_CSV):
            df = pd.read_csv(FEATURES_CSV, usecols=['timestamp'])
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                start, end = df['timestamp'].min(), df['timestamp'].max()
                rng = f"{start} → {end}"
            else:
                rng = "EMPTY"
        else:
            rng = "MISSING"
        label = prefix + " " if prefix else ""
        print(f"📅 {label}Features dataset range: {rng}")
    except Exception as e:
        print(f"⚠️ Could not read features date range: {e}")


def basic_data_checks() -> bool:
    try:
        import pandas as pd
        df = pd.read_csv(FEATURES_CSV)
        n = len(df)
        missing = int(df.isnull().sum().sum())
        print(f"📊 Data rows: {n}, total missing values: {missing}")
        if n < 500:
            print("⚠️ Very few rows – skipping promotion for safety")
            return False
        if missing > n * 10:  # crude threshold
            print("⚠️ Excessive missing values – skipping promotion")
            return False
        return True
    except Exception as e:
        print(f"⚠️ Data checks failed: {e}")
        return False


def registry_path() -> str:
    return os.path.join(REGISTRY_DIR, 'models.json')


def load_registry() -> dict:
    path = registry_path()
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        return json.load(f)


def save_registry(reg: dict) -> None:
    with open(registry_path(), 'w') as f:
        json.dump(reg, f, indent=2)


def init_registry_if_needed() -> dict:
    reg = load_registry()
    if reg:
        return reg
    # Initialize conservative defaults; no promotion on first run
    reg = {
        'created_at': datetime.now().isoformat(),
        'last_run_at': None,
        'thresholds': {
            'min_rmse_rel_improvement_daily': 0.02
        },
        'champions': {
            '24h': {'path': os.path.join(CHAMPIONS_DIR, 'catboost_24h.txt'), 'rmse': None, 'r2': None},
            '48h': {'path': os.path.join(CHAMPIONS_DIR, 'tcn_48h.pth'), 'rmse': None, 'r2': None},
            '72h': {'path': os.path.join(CHAMPIONS_DIR, 'tcn_72h.pth'), 'rmse': None, 'r2': None},
        }
    }
    # Seed champions if not present by copying latest models (best-effort)
    try:
        # CatBoost champion
        cb_latest = None
        for pat in ['saved_models/catboost_multi_horizon_tuned_model.txt', 'saved_models/catboost_fixed_tuned_model.txt', 'saved_models/catboost_fixed_model.txt']:
            if os.path.exists(os.path.join(ROOT, pat)):
                cb_latest = os.path.join(ROOT, pat)
                break
        if cb_latest:
            shutil.copy(cb_latest, reg['champions']['24h']['path'])
    except Exception:
        pass
    try:
        # TCN 48h
        import glob
        c48 = sorted(
            glob.glob(os.path.join(ROOT, 'saved_models', 'tcn_48h_finetuned.pth')) +
            glob.glob(os.path.join(ROOT, 'saved_models', 'tcn_optimized_48h_*.pth')),
            key=os.path.getmtime, reverse=True
        )
        if c48:
            shutil.copy(c48[0], reg['champions']['48h']['path'])
        # TCN 72h
        c72 = sorted(
            glob.glob(os.path.join(ROOT, 'saved_models', 'tcn_72h_finetuned.pth')) +
            glob.glob(os.path.join(ROOT, 'saved_models', 'tcn_optimized_72h_*.pth')),
            key=os.path.getmtime, reverse=True
        )
        if c72:
            shutil.copy(c72[0], reg['champions']['72h']['path'])
    except Exception:
        pass
    save_registry(reg)
    print("✅ Registry initialized (no promotion this run)")
    return reg


def run_finetune() -> str | None:
    code = run_cmd(['python', '11_per_horizon_finetune.py'])
    if code != 0:
        print("❌ Fine-tune step failed")
        return None
    results_csv = os.path.join(SAVED_MODELS, 'per_horizon_finetune_results.csv')
    if not os.path.exists(results_csv):
        print("⚠️ Fine-tune results not found")
        return None
    return results_csv


def parse_finetune_results(path: str) -> dict:
    import pandas as pd
    df = pd.read_csv(path)
    out = {}
    for _, row in df.iterrows():
        name = str(row['Model'])
        out[name] = {'RMSE': float(row['RMSE']), 'R2': float(row['R2'])}
    return out


def maybe_promote(reg: dict, ft_metrics: dict, allow_promotion: bool) -> dict:
    # Map fine-tune keys to horizons
    key_map = {
        'catboost_24h_finetuned': '24h',
        'tcn_48h_finetuned': '48h',
        'tcn_72h_finetuned': '72h',
    }
    thr = reg.get('thresholds', {}).get('min_rmse_rel_improvement_daily', 0.02)

    promoted = []
    for k, horizon in key_map.items():
        if k not in ft_metrics:
            continue
        cand_rmse = ft_metrics[k]['RMSE']
        champ = reg['champions'].get(horizon, {})
        champ_rmse = champ.get('rmse')
        champ_path = champ.get('path')
        # Only consider promotion if we have both metrics and allowed
        if not allow_promotion or champ_rmse is None or cand_rmse is None:
            continue
        rel_gain = (champ_rmse - cand_rmse) / max(1e-9, champ_rmse)
        if rel_gain >= thr:
            # Promote: replace champion file with latest fine-tuned file
            try:
                if horizon == '24h':
                    src = os.path.join(SAVED_MODELS, 'catboost_24h_finetuned.txt')
                elif horizon == '48h':
                    src = os.path.join(SAVED_MODELS, 'tcn_48h_finetuned.pth')
                else:
                    src = os.path.join(SAVED_MODELS, 'tcn_72h_finetuned.pth')
                if os.path.exists(src) and champ_path:
                    shutil.copy(src, champ_path)
                    reg['champions'][horizon]['rmse'] = cand_rmse
                    reg['champions'][horizon]['r2'] = ft_metrics[k]['R2']
                    promoted.append((horizon, rel_gain))
            except Exception as e:
                print(f"⚠️ Promotion failed for {horizon}: {e}")

    if promoted:
        print("✅ Promoted champions:")
        for h, g in promoted:
            print(f"  {h}: +{g*100:.1f}% RMSE improvement")
    else:
        print("ℹ️ No promotions (either no improvement or first-run/blocked)")
    return reg


def generate_forecasts():
    code = run_cmd(['python', 'forecast.py'])
    if code != 0:
        print("⚠️ Forecast generation failed")


def main():
    ensure_dirs()
    refresh_features_if_missing()
    print_features_date_range(prefix="(pre-finetune)")

    # Initialize or load registry
    reg = init_registry_if_needed()
    allow_promotion = features_available() and basic_data_checks() and (reg.get('last_run_at') is not None)

    # Fine-tune and try promotion
    ft_csv = run_finetune()
    if ft_csv:
        ft_metrics = parse_finetune_results(ft_csv)
        reg = maybe_promote(reg, ft_metrics, allow_promotion)

    # Forecast
    print_features_date_range(prefix="(pre-forecast)")
    generate_forecasts()

    reg['last_run_at'] = datetime.now().isoformat()
    save_registry(reg)
    print("\n✅ Daily run completed.")


if __name__ == '__main__':
    main()


