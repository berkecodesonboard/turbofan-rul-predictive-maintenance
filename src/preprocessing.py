from typing import Optional

import pandas as pd
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = (
    ["setting_1", "setting_2", "setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def add_train_rul(
    train_df: pd.DataFrame,
    cap: Optional[int] = None,
) -> pd.DataFrame:
    """
    Adds RUL label to train data.

    Train data reaches failure, so:
    RUL = max_cycle_per_engine - current_cycle
    """
    df = train_df.copy()

    df["max_cycle"] = df.groupby("unit_id")["cycle"].transform("max")
    df["RUL"] = df["max_cycle"] - df["cycle"]

    if cap is not None:
        df["RUL"] = df["RUL"].clip(upper=cap)

    return df


def add_test_rul(
    test_df: pd.DataFrame,
    test_rul_df: pd.DataFrame,
    cap: Optional[int] = None,
) -> pd.DataFrame:
    """
    Adds RUL label to test data.

    Test data does not reach failure.
    RUL_FD001 gives remaining cycles after the last observed cycle.

    failure_cycle = max_test_cycle + final_RUL
    RUL = failure_cycle - current_cycle
    """
    df = test_df.copy()

    max_test_cycles = (
        df.groupby("unit_id")["cycle"]
        .max()
        .reset_index()
        .rename(columns={"cycle": "max_test_cycle"})
    )

    df = df.merge(max_test_cycles, on="unit_id", how="left")
    df = df.merge(test_rul_df, on="unit_id", how="left")

    df["failure_cycle"] = df["max_test_cycle"] + df["final_RUL"]
    df["RUL"] = df["failure_cycle"] - df["cycle"]

    if cap is not None:
        df["RUL"] = df["RUL"].clip(upper=cap)

    return df


def split_features_target(df: pd.DataFrame):
    """
    Splits dataframe into input features X and target y.
    """
    X = df[FEATURE_COLUMNS]
    y = df["RUL"]

    return X, y


def scale_train_test(X_train, X_test):
    """
    Fits scaler only on train data to prevent data leakage.
    """
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler