from home_credit.schema import ACCEPTED_CATEGORICAL_FEATURES, ACCEPTED_FINAL_FEATURES


def test_schema_feature_count() -> None:
    assert (
        len(ACCEPTED_FINAL_FEATURES) == 166 and len(ACCEPTED_CATEGORICAL_FEATURES) == 15
    ), "Expected 166 features"


def test_schema_cat_feature_count() -> None:
    assert len(ACCEPTED_CATEGORICAL_FEATURES) == 15, "Expected 15 categorical features"


def test_schema_no_duplicates_in_features() -> None:
    assert len(ACCEPTED_FINAL_FEATURES) == len(set(ACCEPTED_FINAL_FEATURES)), (
        "Duplicate features found in ACCEPTED_FINAL_FEATURES"
    )


def test_schema_no_duplicates_in_categorical_features() -> None:
    assert len(ACCEPTED_CATEGORICAL_FEATURES) == len(
        set(ACCEPTED_CATEGORICAL_FEATURES)
    ), "Duplicate categorical features found in ACCEPTED_CATEGORICAL_FEATURES"


def test_categorical_features_subset() -> None:
    assert set(ACCEPTED_CATEGORICAL_FEATURES).issubset(set(ACCEPTED_FINAL_FEATURES)), (
        "Categorical features are not a subset of final features"
    )


def test_target_is_not_in_features() -> None:
    assert "TARGET" not in ACCEPTED_FINAL_FEATURES, (
        "TARGET should not be in ACCEPTED_FINAL_FEATURES"
    )


def test_target_is_not_in_categorical_features() -> None:
    assert "TARGET" not in ACCEPTED_CATEGORICAL_FEATURES, (
        "TARGET should not be in ACCEPTED_CATEGORICAL_FEATURES"
    )


def test_id_is_not_in_features() -> None:
    assert "SK_ID_CURR" not in ACCEPTED_FINAL_FEATURES, (
        "SK_ID_CURR should not be in ACCEPTED_FINAL_FEATURES"
    )


def test_id_is_not_in_categorical_features() -> None:
    assert "SK_ID_CURR" not in ACCEPTED_CATEGORICAL_FEATURES, (
        "SK_ID_CURR should not be in ACCEPTED_CATEGORICAL_FEATURES"
    )


def test_no_ip_in_features() -> None:
    for column in ACCEPTED_FINAL_FEATURES:
        assert not column.startswith("IPX_"), (
            f"Feature {column} should not start with 'IPX_'"
        )


def test_no_ip_in_categorical_features() -> None:
    for column in ACCEPTED_CATEGORICAL_FEATURES:
        assert not column.startswith("IPX_"), (
            f"Categorical feature {column} should not start with 'IPX_'"
        )
