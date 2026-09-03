"""Master coordinator script to execute both Action Transformer and Anomaly Autoencoder training."""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    python_exe = sys.executable

    print("\n=======================================================")
    print("STEP 1: Training Spatial-Temporal Action Transformer")
    print("=======================================================\n")
    cmd_action = [python_exe, str(root / "training" / "train_action.py"), "--epochs", "50"]
    subprocess.run(cmd_action, check=True)

    print("\n=======================================================")
    print("STEP 2: Training Unsupervised Pose Sequence Autoencoder")
    print("=======================================================\n")
    cmd_anomaly = [python_exe, str(root / "training" / "train_anomaly.py"), "--epochs", "40"]
    subprocess.run(cmd_anomaly, check=True)

    print("\n=======================================================")
    print("STEP 3: Running Model Evaluation")
    print("=======================================================\n")
    cmd_eval = [python_exe, str(root / "training" / "evaluate.py")]
    subprocess.run(cmd_eval, check=True)

    print("\n[Success] All temporal models trained and evaluated successfully!")


if __name__ == "__main__":
    main()
