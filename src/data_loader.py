from pathlib import Path

import pandas as pd


INDEX_COLUMNS = ["unit_id", "cycle"]

SETTING_COLUMNS = [
    "setting_1",
    "setting_2",
    "setting_3",
]

SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]

ALL_COLUMNS = INDEX_COLUMNS + SETTING_COLUMNS + SENSOR_COLUMNS


def load_train_data(data_dir: str | Path, dataset_id: str = "FD001") -> pd.DataFrame:
    """
    Loads C-MAPSS train data.

    Train data contains full run-to-failure trajectories.
    Therefore, RUL can be generated from max cycle per engine.
    """
    data_dir = Path(data_dir)
    file_path = data_dir / f"train_{dataset_id}.txt"

    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=ALL_COLUMNS,
    )

    return df


def load_test_data(data_dir: str | Path, dataset_id: str = "FD001") -> pd.DataFrame:
    """
    Loads C-MAPSS test data.

    Test data is truncated before failure.
    True RUL values for the last cycle of each engine are stored separately.
    """
    data_dir = Path(data_dir)
    file_path = data_dir / f"test_{dataset_id}.txt"

    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=ALL_COLUMNS,
    )

    return df


def load_test_rul(data_dir: str | Path, dataset_id: str = "FD001") -> pd.DataFrame:
    """
    Loads true RUL values for test engines.

    Each row corresponds to the RUL of one test engine at its last observed cycle.
    """
    data_dir = Path(data_dir)
    file_path = data_dir / f"RUL_{dataset_id}.txt"

    rul_df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=["final_RUL"],
    )

    rul_df["unit_id"] = range(1, len(rul_df) + 1)

    return rul_df[["unit_id", "final_RUL"]]