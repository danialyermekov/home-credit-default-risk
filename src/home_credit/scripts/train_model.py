import argparse
import logging
from pathlib import Path

from home_credit.core.model_config import FINAL_CB_PARAMS
from home_credit.services.training import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CatBoost model")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing the raw data files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/models/catboost_cli.cbm"),
        help="Path to save the trained model",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(filename)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    trainer = Trainer(
        data_dir=args.data_dir,
        model_params=FINAL_CB_PARAMS,
        model_save_path=args.output,
    )

    trainer.train_model()


if __name__ == "__main__":
    main()
