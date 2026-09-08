"""Unit and integration tests for feature engineering modules.

Verifies:
1. Exact feature counts and column names for each historical module.
2. Interface contracts and types.
3. Feature parity and alignment with ACCEPTED_FINAL_FEATURES.
4. Correct assembly and categorical handling.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from assemble_features import (
    ACCEPTED_CATEGORICAL_FEATURES,
    assemble_features,
    build_application_features,
    load_accepted_final_features,
)
from bureau_balance_features import (
    BBX_ACCEPTED_FEATURES,
    build_bureau_balance_features,
)
from bureau_features import (
    BUREAU_ACCEPTED_FEATURES,
    build_bureau_features,
)
from credit_card_features import (
    CREDIT_CARD_ACCEPTED_FEATURES,
    build_credit_card_features,
)
from installments_features import (
    INSTALLMENTS_ACCEPTED_FEATURES,
    build_installments_features,
)
from pos_cash_features import (
    POS_ACCEPTED_FEATURES,
    build_pos_cash_features,
)
from previous_application_features import (
    PREVIOUS_APPLICATION_ACCEPTED_FEATURES,
    build_previous_application_features,
)


class TestFeatureModules(unittest.TestCase):
    """Test suite for individual table feature modules."""

    def test_accepted_feature_counts(self):
        """Verify the number of accepted features per historical table."""
        self.assertEqual(len(BUREAU_ACCEPTED_FEATURES), 16)
        self.assertEqual(len(BBX_ACCEPTED_FEATURES), 9)
        self.assertEqual(len(PREVIOUS_APPLICATION_ACCEPTED_FEATURES), 34)
        self.assertEqual(len(CREDIT_CARD_ACCEPTED_FEATURES), 12)
        self.assertEqual(len(INSTALLMENTS_ACCEPTED_FEATURES), 19)
        self.assertEqual(len(POS_ACCEPTED_FEATURES), 9)

        total_historical = (
            len(BUREAU_ACCEPTED_FEATURES)
            + len(BBX_ACCEPTED_FEATURES)
            + len(PREVIOUS_APPLICATION_ACCEPTED_FEATURES)
            + len(CREDIT_CARD_ACCEPTED_FEATURES)
            + len(INSTALLMENTS_ACCEPTED_FEATURES)
            + len(POS_ACCEPTED_FEATURES)
        )
        self.assertEqual(total_historical, 99)

    def test_load_accepted_final_features(self):
        """Verify that load_accepted_final_features loads 166 features matching model schema."""
        final_features = load_accepted_final_features()
        self.assertEqual(len(final_features), 166)
        self.assertEqual(len(set(final_features)), 166)

        # Check all historical features are included in final_features
        all_hist = (
            BUREAU_ACCEPTED_FEATURES
            + BBX_ACCEPTED_FEATURES
            + PREVIOUS_APPLICATION_ACCEPTED_FEATURES
            + CREDIT_CARD_ACCEPTED_FEATURES
            + INSTALLMENTS_ACCEPTED_FEATURES
            + POS_ACCEPTED_FEATURES
        )
        for feat in all_hist:
            self.assertIn(feat, final_features)

    def test_bureau_features_mock(self):
        """Verify build_bureau_features with synthetic data."""
        bureau_mock = pd.DataFrame({
            "SK_ID_CURR": [100001, 100001, 100002],
            "SK_ID_BUREAU": [1, 2, 3],
            "CREDIT_ACTIVE": ["Active", "Closed", "Active"],
            "DAYS_CREDIT": [-100, -500, -200],
            "DAYS_CREDIT_UPDATE": [-10, -50, 0],
            "AMT_CREDIT_SUM": [100000.0, 50000.0, 200000.0],
            "AMT_CREDIT_SUM_LIMIT": [0.0, 0.0, 5000.0],
            "AMT_CREDIT_SUM_DEBT": [50000.0, 0.0, 100000.0],
            "AMT_CREDIT_MAX_OVERDUE": [0.0, 1000.0, 0.0],
        })

        out = build_bureau_features(bureau_mock)
        self.assertIn("SK_ID_CURR", out.columns)
        for col in BUREAU_ACCEPTED_FEATURES:
            self.assertIn(col, out.columns)
        self.assertEqual(len(out), 2)
        row1 = out.loc[out["SK_ID_CURR"] == 100001].iloc[0]
        self.assertEqual(row1["BUREAU_CREDIT_COUNT"], 2)
        self.assertEqual(row1["BUREAU_ACTIVE_COUNT"], 1)
        self.assertEqual(row1["BUREAU_CLOSED_COUNT"], 1)
        self.assertAlmostEqual(row1["BUREAU_ACTIVE_SHARE"], 0.5)

    def test_bureau_balance_features_mock(self):
        """Verify build_bureau_balance_features with synthetic data."""
        bb_mock = pd.DataFrame({
            "SK_ID_BUREAU": [1, 1, 2],
            "MONTHS_BALANCE": [0, -1, 0],
            "STATUS": ["0", "1", "C"],
        })
        bureau_link = pd.DataFrame({
            "SK_ID_BUREAU": [1, 2],
            "SK_ID_CURR": [100001, 100002],
        })

        out = build_bureau_balance_features(bb_mock, bureau=bureau_link)
        self.assertIn("SK_ID_CURR", out.columns)
        for col in BBX_ACCEPTED_FEATURES:
            self.assertIn(col, out.columns)
        self.assertEqual(len(out), 2)

    def test_previous_application_features_mock(self):
        """Verify build_previous_application_features with synthetic data."""
        prev_mock = pd.DataFrame({
            "SK_ID_CURR": [100001, 100001],
            "SK_ID_PREV": [101, 102],
            "AMT_APPLICATION": [50000.0, 100000.0],
            "AMT_CREDIT": [50000.0, 90000.0],
            "AMT_GOODS_PRICE": [50000.0, 100000.0],
            "AMT_ANNUITY": [5000.0, 9000.0],
            "AMT_DOWN_PAYMENT": [0.0, 10000.0],
            "CNT_PAYMENT": [12, 24],
            "DAYS_DECISION": [-100, -300],
            "DAYS_FIRST_DRAWING": [365243, -290],
            "DAYS_FIRST_DUE": [-90, -290],
            "DAYS_LAST_DUE_1ST_VERSION": [270, 430],
            "DAYS_LAST_DUE": [365243, -50],
            "DAYS_TERMINATION": [365243, -40],
        })

        out = build_previous_application_features(prev_mock)
        self.assertIn("SK_ID_CURR", out.columns)
        for col in PREVIOUS_APPLICATION_ACCEPTED_FEATURES:
            self.assertIn(col, out.columns)
        self.assertEqual(len(out), 1)
        row = out.iloc[0]
        self.assertEqual(row["TOTAL_AMT_PREV_APPLICATION"], 150000.0)

    def test_credit_card_features_mock(self):
        """Verify build_credit_card_features with synthetic data."""
        cc_mock = pd.DataFrame({
            "SK_ID_CURR": [100001, 100001],
            "SK_ID_PREV": [201, 201],
            "MONTHS_BALANCE": [-2, -1],
            "AMT_BALANCE": [10000.0, 20000.0],
            "AMT_CREDIT_LIMIT_ACTUAL": [50000.0, 50000.0],
            "AMT_DRAWINGS_CURRENT": [5000.0, 10000.0],
        })

        out = build_credit_card_features(cc_mock)
        self.assertIn("SK_ID_CURR", out.columns)
        for col in CREDIT_CARD_ACCEPTED_FEATURES:
            self.assertIn(col, out.columns)
        self.assertEqual(len(out), 1)
        row = out.iloc[0]
        self.assertEqual(row["CC_MONTHS_OBSERVED"], 2)
        self.assertEqual(row["CC_LATEST_BALANCE"], 20000.0)

    def test_installments_features_mock(self):
        """Verify build_installments_features with synthetic data."""
        ip_mock = pd.DataFrame({
            "SK_ID_CURR": [100001, 100001],
            "SK_ID_PREV": [301, 301],
            "NUM_INSTALMENT_VERSION": [1, 1],
            "NUM_INSTALMENT_NUMBER": [1, 2],
            "DAYS_INSTALMENT": [-60, -30],
            "DAYS_ENTRY_PAYMENT": [-62, -25],
            "AMT_INSTALMENT": [5000.0, 5000.0],
            "AMT_PAYMENT": [5000.0, 4500.0],
        })

        out = build_installments_features(ip_mock)
        self.assertIn("SK_ID_CURR", out.columns)
        for col in INSTALLMENTS_ACCEPTED_FEATURES:
            self.assertIn(col, out.columns)
        self.assertEqual(len(out), 1)
        row = out.iloc[0]
        self.assertEqual(row["IP_CONTRACT_COUNT"], 1)
        self.assertEqual(row["IP_PAYMENT_RECORD_COUNT"], 2)

    def test_pos_cash_features_mock(self):
        """Verify build_pos_cash_features with synthetic data."""
        pos_mock = pd.DataFrame({
            "SK_ID_CURR": [100001, 100001],
            "SK_ID_PREV": [401, 401],
            "MONTHS_BALANCE": [-2, -1],
            "CNT_INSTALMENT": [12, 12],
            "CNT_INSTALMENT_FUTURE": [10, 9],
            "SK_DPD": [0, 0],
            "SK_DPD_DEF": [0, 0],
        })

        out = build_pos_cash_features(pos_mock)
        self.assertIn("SK_ID_CURR", out.columns)
        for col in POS_ACCEPTED_FEATURES:
            self.assertIn(col, out.columns)
        self.assertEqual(len(out), 1)

    def test_assemble_features_alignment(self):
        """Verify that assemble_features produces exactly the 166 model features."""
        final_features = load_accepted_final_features()

        # Build dummy application table with all necessary columns
        app_dict = {
            "SK_ID_CURR": [100001],
            "DAYS_BIRTH": [-15000],
            "DAYS_EMPLOYED": [-1000],
            "AMT_INCOME_TOTAL": [200000.0],
            "AMT_CREDIT": [500000.0],
            "AMT_ANNUITY": [25000.0],
        }
        for c in final_features:
            if c not in app_dict:
                if c in ACCEPTED_CATEGORICAL_FEATURES:
                    app_dict[c] = ["TestVal"]
                else:
                    app_dict[c] = [0.0]
        raw_app = pd.DataFrame(app_dict)

        assembled = assemble_features(
            application=raw_app,
            include_id=False,
        )

        self.assertEqual(list(assembled.columns), final_features)
        self.assertEqual(assembled.index.name, "SK_ID_CURR")
        self.assertEqual(assembled.shape, (1, 166))

        for cat_col in ACCEPTED_CATEGORICAL_FEATURES:
            self.assertTrue(
                isinstance(assembled[cat_col].dtype, pd.CategoricalDtype),
                f"{cat_col} is not category dtype",
            )


if __name__ == "__main__":
    unittest.main()
