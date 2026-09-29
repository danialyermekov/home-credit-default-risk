from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import home_credit.services.training as training_module
from home_credit.services.training import Trainer

FEATURES = ("f_num", "f_cat")
CAT_FEATURES = ("f_cat",)


def fake_read_csv(path: Path, *args: Any, **kwargs: Any) -> pd.DataFrame:
    if Path(path).name == "application_train.csv":
        return pd.DataFrame(
            {
                "SK_ID_CURR": [20, 10],
                "TARGET": [1, 0],
                "AMT_INCOME_TOTAL": [200_000, 100_000],
            }
        )

    return pd.DataFrame()


def test_get_train_dataset_aligns_target_by_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(training_module.pd, "read_csv", fake_read_csv)

    def fake_build_feature_dataset(**kwargs: Any) -> pd.DataFrame:
        # Намеренно меняем порядок ID.
        return pd.DataFrame(
            {
                "SK_ID_CURR": [10, 20],
                "f_num": [1.0, 2.0],
                "f_cat": pd.Series(["A", None], dtype="category"),
            }
        )

    monkeypatch.setattr(
        training_module,
        "build_feature_dataset",
        fake_build_feature_dataset,
    )

    trainer = Trainer(
        data_dir=tmp_path,
        model_params={},
        features=FEATURES,
        cat_features=CAT_FEATURES,
    )

    X, y = trainer._get_train_dataset()

    assert list(X.columns) == ["f_num", "f_cat"]

    # ID 10 имеет TARGET=0, ID 20 имеет TARGET=1.
    assert y.tolist() == [0, 1]

    assert X["f_cat"].tolist() == ["A", "__MISSING__"]


def test_get_train_dataset_rejects_missing_applicant(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(training_module.pd, "read_csv", fake_read_csv)

    def fake_build_feature_dataset(**kwargs: Any) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "SK_ID_CURR": [10],
                "f_num": [1.0],
                "f_cat": ["A"],
            }
        )

    monkeypatch.setattr(
        training_module,
        "build_feature_dataset",
        fake_build_feature_dataset,
    )

    trainer = Trainer(
        data_dir=tmp_path,
        model_params={},
        features=FEATURES,
        cat_features=CAT_FEATURES,
    )

    with pytest.raises(
        ValueError,
        match="Some training applicants were lost",
    ):
        trainer._get_train_dataset()


def test_train_model_fits_and_saves_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    X = pd.DataFrame(
        {
            "f_num": [1.0, 2.0],
            "f_cat": ["A", "__MISSING__"],
        }
    )
    y = pd.Series([0, 1], name="TARGET")

    output_path = tmp_path / "models" / "model.cbm"

    trainer = Trainer(
        data_dir=tmp_path,
        model_params={"iterations": 10},
        model_save_path=output_path,
        features=FEATURES,
        cat_features=CAT_FEATURES,
    )

    monkeypatch.setattr(
        trainer,
        "_get_train_dataset",
        lambda: (X, y),
    )

    class FakeCatBoostClassifier:
        def __init__(self, **params: Any) -> None:
            self.params = params
            self.cat_features: list[str] | None = None

        def fit(
            self,
            features: pd.DataFrame,
            target: pd.Series,
            cat_features: list[str],
        ) -> None:
            self.cat_features = cat_features

        def save_model(self, path: str | Path) -> None:
            Path(path).write_text("fake model", encoding="utf-8")

    monkeypatch.setattr(
        training_module,
        "CatBoostClassifier",
        FakeCatBoostClassifier,
    )

    model = trainer.train_model()

    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8") == "fake model"

    assert model.params["iterations"] == 10
    assert model.params["allow_writing_files"] is False
    assert model.cat_features == ["f_cat"]
