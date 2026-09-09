from home_credit.schema import ACCEPTED_FINAL_FEATURES, ACCEPTED_CATEGORICAL_FEATURES

def test_schema_feature_count():
    assert (len(set(ACCEPTED_FINAL_FEATURES)) == 166
            and len(set(ACCEPTED_CATEGORICAL_FEATURES)) == 15), "Feature counts do not match expected values"

def test_schema_no_duplicates():
    assert len(ACCEPTED_FINAL_FEATURES) == len(set(ACCEPTED_FINAL_FEATURES)), "Duplicate features found in ACCEPTED_FINAL_FEATURES"
    assert len(ACCEPTED_CATEGORICAL_FEATURES) == len(set(ACCEPTED_CATEGORICAL_FEATURES)), "Duplicate categorical features found in ACCEPTED_CATEGORICAL_FEATURES"

def test_categorical_features_subset():
    assert set(ACCEPTED_CATEGORICAL_FEATURES).issubset(set(ACCEPTED_FINAL_FEATURES)), "Categorical features are not a subset of final features"

def test_target_is_not_in_features():
    assert 'TARGET' not in ACCEPTED_FINAL_FEATURES, "TARGET should not be in ACCEPTED_FINAL_FEATURES"
    assert 'TARGET' not in ACCEPTED_CATEGORICAL_FEATURES, "TARGET should not be in ACCEPTED_CATEGORICAL_FEATURES"

def test_id_is_not_in_features():
    assert 'SK_ID_CURR' not in ACCEPTED_FINAL_FEATURES, "SK_ID_CURR should not be in ACCEPTED_FINAL_FEATURES"
    assert 'SK_ID_CURR' not in ACCEPTED_CATEGORICAL_FEATURES, "SK_ID_CURR should not be in ACCEPTED_CATEGORICAL_FEATURES"

def test_no_ipx():
    for column in ACCEPTED_FINAL_FEATURES:
        assert not column.startswith("IPX_"), f"Feature {column} should not start with 'IPX_'"
    for column in ACCEPTED_CATEGORICAL_FEATURES:
        assert not column.startswith("IPX_"), f"Feature {column} should not start with 'IPX_'"
