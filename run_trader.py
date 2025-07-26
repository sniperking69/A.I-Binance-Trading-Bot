import subprocess
import time

while True:
    print("\n🔁 Starting periodic inference training (stonevision.py)...", flush=True)
    train_result = subprocess.run(
        ["python3", "-u", "stonevision.py"]
    )

    if train_result.returncode != 0:
        print("❌ Training script failed. Skipping trading this cycle.", flush=True)
    else:
        print("✅ Training complete. Proceeding to trading...", flush=True)
        trade_result = subprocess.run(
            ["python3", "-u", "stone_trader.py"]
        )
        if trade_result.returncode != 0:
            print("❌ Trading script failed.", flush=True)
        else:
            print("✅ Trading cycle complete.", flush=True)

    print("🕒 Sleeping for 4 hours...\n", flush=True)
    time.sleep(4 * 60 * 60)
