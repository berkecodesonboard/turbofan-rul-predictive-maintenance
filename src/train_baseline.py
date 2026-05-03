from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

from data_loader import load_test_data, load_test_rul, load_train_data
from evaluate import regression_metrics
from plots import (
    plot_actual_vs_predicted,
    plot_error_histogram,
    plot_model_comparison,
)
from preprocessing import (
    add_test_rul,
    add_train_rul,
    scale_train_test,
    split_features_target,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "raw"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"

RUL_CAP = 125


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    train_raw = load_train_data(DATA_DIR, "FD001")
    test_raw = load_test_data(DATA_DIR, "FD001")
    test_rul = load_test_rul(DATA_DIR, "FD001")

    print("Generating RUL labels...")
    train_df = add_train_rul(train_raw, cap=RUL_CAP)
    test_df = add_test_rul(test_raw, test_rul, cap=RUL_CAP)

    # All-cycle test data:
    # Test setindeki tüm cycle satırları üzerinden değerlendirme yapılır.
    X_train, y_train = split_features_target(train_df)
    X_test, y_test = split_features_target(test_df)

    # Last-cycle test data:
    # Her test motorunun sadece son gözlenen cycle'ı alınır.
    # Bu, C-MAPSS benchmark mantığına daha yakındır.
    last_cycle_test_df = test_df.loc[
        test_df.groupby("unit_id")["cycle"].idxmax()
    ].copy()

    X_test_last, y_test_last = split_features_target(last_cycle_test_df)

    print("Scaling features...")

    # Leakage önlemek için scaler sadece train veriye fit edilir.
    X_train_scaled, X_test_scaled, scaler = scale_train_test(X_train, X_test)

    # Last-cycle test de aynı scaler ile transform edilir.
    X_test_last_scaled = scaler.transform(X_test_last)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=50,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),
    }

    all_cycle_results = []
    last_cycle_results = []

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")

        model.fit(X_train_scaled, y_train)

        # 1) All-cycle predictions
        y_pred_all = model.predict(X_test_scaled)

        all_metrics = regression_metrics(y_test, y_pred_all)
        all_metrics["model"] = model_name
        all_metrics["evaluation"] = "all_cycles"

        all_cycle_results.append(all_metrics)

        # 2) Last-cycle predictions
        y_pred_last = model.predict(X_test_last_scaled)

        last_metrics = regression_metrics(y_test_last, y_pred_last)
        last_metrics["model"] = model_name
        last_metrics["evaluation"] = "last_cycle"

        last_cycle_results.append(last_metrics)

        safe_name = model_name.lower().replace(" ", "_")

        # All-cycle plots
        plot_actual_vs_predicted(
            y_test,
            y_pred_all,
            f"{model_name} - All Cycles",
            FIGURES_DIR / f"actual_vs_predicted_{safe_name}_all_cycles.png",
        )

        plot_error_histogram(
            y_test,
            y_pred_all,
            f"{model_name} - All Cycles",
            FIGURES_DIR / f"error_histogram_{safe_name}_all_cycles.png",
        )

        # Last-cycle plots
        plot_actual_vs_predicted(
            y_test_last,
            y_pred_last,
            f"{model_name} - Last Cycle",
            FIGURES_DIR / f"actual_vs_predicted_{safe_name}_last_cycle.png",
        )

        plot_error_histogram(
            y_test_last,
            y_pred_last,
            f"{model_name} - Last Cycle",
            FIGURES_DIR / f"error_histogram_{safe_name}_last_cycle.png",
        )

        print(f"{model_name} all-cycle metrics:")
        print(all_metrics)

        print(f"{model_name} last-cycle metrics:")
        print(last_metrics)

    all_cycle_metrics_df = pd.DataFrame(all_cycle_results)
    all_cycle_metrics_df = all_cycle_metrics_df[
        ["model", "evaluation", "MAE", "RMSE", "R2"]
    ]

    last_cycle_metrics_df = pd.DataFrame(last_cycle_results)
    last_cycle_metrics_df = last_cycle_metrics_df[
        ["model", "evaluation", "MAE", "RMSE", "R2"]
    ]

    all_cycle_metrics_path = METRICS_DIR / "baseline_metrics_all_cycles.csv"
    last_cycle_metrics_path = METRICS_DIR / "baseline_metrics_last_cycle.csv"

    all_cycle_metrics_df.to_csv(all_cycle_metrics_path, index=False)
    last_cycle_metrics_df.to_csv(last_cycle_metrics_path, index=False)

    plot_model_comparison(
        all_cycle_metrics_df,
        "RMSE",
        FIGURES_DIR / "model_comparison_rmse_all_cycles.png",
    )

    plot_model_comparison(
        last_cycle_metrics_df,
        "RMSE",
        FIGURES_DIR / "model_comparison_rmse_last_cycle.png",
    )

    print("\nBaseline training completed.")

    print("\nAll-cycle evaluation:")
    print(all_cycle_metrics_df)

    print("\nLast-cycle evaluation:")
    print(last_cycle_metrics_df)


if __name__ == "__main__":
    main()