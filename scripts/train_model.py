from pathlib import Path
import sys
import argparse
import json

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from carebridge.risk_engine import (
    train_addresso_bundle,
    train_and_save_model,
    train_eeg_bundle,
    train_imaging_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train CareBridge models for one of several dataset modalities."
    )
    parser.add_argument(
        "modality",
        nargs="?",
        default="tabular",
        choices=["tabular", "addresso", "eeg", "imaging"],
        help="Which training pipeline to run.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Path to CSV or dataset root. Leave blank to use default configured paths for non-tabular modalities.",
    )
    args = parser.parse_args()

    if args.modality == "tabular":
        if not args.path:
            raise SystemExit("For tabular training, provide a CSV path.")
        metadata = train_and_save_model(args.path)
    elif args.modality == "addresso":
        metadata = train_addresso_bundle(args.path) if args.path else train_addresso_bundle()
    elif args.modality == "eeg":
        metadata = train_eeg_bundle(args.path) if args.path else train_eeg_bundle()
    elif args.modality == "imaging":
        metadata = train_imaging_bundle(args.path) if args.path else train_imaging_bundle()
    else:
        raise SystemExit("Unsupported modality.")

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
