"""
Backfill 150 days of historical weather and pollution data ending on 2025-07-31.
This uses the same storage structure as the main pipeline and saves under
data_repositories/historical_data. Running the regular 01 script later will
append and deduplicate into the same master merged dataset automatically.
"""

import sys
import os
import importlib.util
from datetime import datetime, timedelta


def _load_data_collector():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    module_path = os.path.join(current_dir, "01_data_collection.py")
    spec = importlib.util.spec_from_file_location("data_collection_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load 01_data_collection.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "DataCollector")


def main() -> None:
    # End at 2025-07-31 inclusive; set end to last second of the day to capture full day
    end_dt = datetime(2025, 7, 31, 23, 59, 59)
    start_dt = end_dt - timedelta(days=150)

    print("🚀 Starting 150-day backfill (historical) ...")
    print(f"⏳ Start: {start_dt}  →  End: {end_dt}")

    DataCollector = _load_data_collector()
    collector = DataCollector(start_date=start_dt, end_date=end_dt, mode="historical")
    ok = collector.run_pipeline()

    if ok:
        print("\n🎉 Backfill completed successfully.")
        sys.exit(0)
    else:
        print("\n❌ Backfill failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()


