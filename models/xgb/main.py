#!/usr/bin/env python
import argparse
from pathlib import Path

import csv
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from tqdm import tqdm
from xgboost import XGBRegressor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="XGBoost feature selection and evaluation for MOF bandgaps"
    )
    parser.add_argument(
        "--matminer-features",
        type=Path,
        default=Path("data/matminer_features.csv"),
        help="Path to matminer features CSV",
    )
    parser.add_argument(
        "--mof-features",
        type=Path,
        default=Path("data/mof_features_combined.csv"),
        help="Path to MOF combined features CSV",
    )
    parser.add_argument(
        "--mofdscribe-results",
        type=Path,
        default=Path("data/mofdscribe_results.csv"),
        help="Path to mofdscribe results CSV",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results/xgb"),
        help="Directory to save outputs",
    )
    return parser.parse_args()


def build_merged_dataframe(
    df: pd.DataFrame, df_2: pd.DataFrame, df_3: pd.DataFrame
) -> pd.DataFrame:
    """
    Build the merged feature dataframe and return it with the final columns:
    [qmof_ids, bandgaps, all feature groups...]
    """

    # Drop columns from df_3 as in your original code
    drop_cols = [col for col in ["uc_volume", "density"] if col in df_3.columns]
    df_3 = df_3.drop(columns=drop_cols)

    # Merge on qmof_ids / qmof_id keys
    merged_df = pd.merge(df, df_2, left_on="qmof_ids", right_on="qmof_id", how="inner")
    merged_df = pd.merge(merged_df, df_3, left_on="qmof_ids", right_on="qmof_id", how="inner")

    # Define feature groups (same as your script)
    bond_length_columns = [
        "mean absolute deviation in relative bond length",
        "max relative bond length",
        "min relative bond length",
    ]

    neighbor_distance_columns = [
        "minimum neighbor distance variation",
        "maximum neighbor distance variation",
        "range neighbor distance variation",
        "mean neighbor distance variation",
        "avg_dev neighbor distance variation",
    ]

    structural_info = [
        "mean absolute deviation in relative cell size",
        "max packing efficiency",
        "structural complexity per atom",
        "structural complexity per cell",
    ]

    structural_metrics = [
        "density",
        "vpa",
        "packing fraction",
    ]

    ordering_parameters = [
        "mean ordering parameter shell 1",
        "mean ordering parameter shell 2",
        "mean ordering parameter shell 3",
    ]

    rdf_columns = [col for col in df.columns if col.startswith("rdf [")]

    spacegroup_columns = [
        "spacegroup_num",
        "crystal_system",
        "crystal_system_int",
        "is_centrosymmetric",
        "n_symmetry_ops",
        "dimensionality",
    ]

    pore_features = [
        "asa_a2",
        "asa_m2cm3",
        "asa_m2g",
        "nasa_a2_0.1",
        "nasa_m2cm3_0.1",
        "nasa_m2g_0.1",
        "lis",
        "lifs",
        "lifsp",
        "av_volume_fraction_0.1",
        "av_cm3g_0.1",
        "nav_a3_0.1",
        "nav_volume_fraction_0.1",
        "nav_cm3g_0.1",
    ]

    # Individual dataframes for safety/clarity
    bond_length_df = merged_df[bond_length_columns]
    neighbor_distance_df = merged_df[neighbor_distance_columns]
    structural_info_df = merged_df[structural_info]
    structural_metrics_df = merged_df[structural_metrics]
    ordering_parameters_df = merged_df[ordering_parameters]

    rdf_df = merged_df[rdf_columns].copy()
    rdf_df.columns = [col.replace("[", "_").replace("]", "_") for col in rdf_df.columns]

    crystal_system_mapping = {
        "monoclinic": 1,
        "triclinic": 2,
        "hexagonal": 3,
        "orthorhombic": 4,
        "tetragonal": 5,
        "cubic": 6,
        "rhombohedral": 7,
    }
    spacegroup_df = merged_df[spacegroup_columns].copy()
    spacegroup_df["is_centrosymmetric"] = spacegroup_df["is_centrosymmetric"].astype(int)
    spacegroup_df["crystal_system"] = spacegroup_df["crystal_system"].map(crystal_system_mapping)

    pore_features_df = merged_df[pore_features]

    # Final merged dataframe with only the desired columns
    merged_df_final = pd.concat(
        [
            merged_df[["qmof_ids", "bandgaps"]],
            bond_length_df,
            neighbor_distance_df,
            structural_info_df,
            structural_metrics_df,
            ordering_parameters_df,
            rdf_df,
            spacegroup_df,
            pore_features_df,
        ],
        axis=1,
    )

    return merged_df_final


def run_feature_importance(
    merged_df: pd.DataFrame, param_grid: dict, outdir: Path
) -> pd.Series:
    """
    Step 1: Iterate random_state 0-9 on full features to get average XGB feature importances.
    Saves avg_feature_importances.csv and returns the importance series.
    """
    warnings.filterwarnings("ignore")

    full_features = merged_df.drop(columns=["qmof_ids", "bandgaps"]).columns.tolist()
    X_full = merged_df.drop(columns=["qmof_ids", "bandgaps"])
    X_full = X_full.fillna(X_full.mean())
    y_full = merged_df["bandgaps"]

    feature_importances_list = []
    print("Starting feature importance estimation for random states 0-9:")

    for rs in tqdm(range(10), desc="Feature Importance"):
        X_train, X_test, y_train, y_test = train_test_split(
            X_full, y_full, test_size=0.1, random_state=rs
        )
        xgb = XGBRegressor(objective="reg:squarederror", random_state=42)
        grid_search = GridSearchCV(
            estimator=xgb,
            param_grid=param_grid,
            cv=5,
            scoring="r2",
            n_jobs=-1,
            verbose=0,
        )
        grid_search.fit(X_train, y_train)
        fi_series = pd.Series(
            grid_search.best_estimator_.feature_importances_, index=X_full.columns
        )
        feature_importances_list.append(fi_series)

    avg_feature_importances = (
        pd.concat(feature_importances_list, axis=1)
        .mean(axis=1)
        .sort_values(ascending=False)
    )

    out_path = outdir / "avg_feature_importances.csv"
    avg_feature_importances.to_csv(out_path, header=["average_importance"])
    print(f"Saved average feature importances to {out_path}")

    return avg_feature_importances


def run_top50_experiment(
    merged_df: pd.DataFrame,
    avg_feature_importances: pd.Series,
    param_grid: dict,
    outdir: Path,
) -> pd.DataFrame:
    """
    Step 2: Rerun grid search using only top 50 features (rs = 10).
    Saves model_performance_top50.csv and returns its dataframe.
    """
    X_full = merged_df.drop(columns=["qmof_ids", "bandgaps"])
    X_full = X_full.fillna(X_full.mean())
    y_full = merged_df["bandgaps"]

    top_50_features = avg_feature_importances.head(50).index.tolist()
    X = X_full[top_50_features]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_full, test_size=0.1, random_state=10
    )

    xgb = XGBRegressor(objective="reg:squarederror", random_state=42)
    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    performance_top50 = {
        "random_state": 10,
        "BestParams": grid_search.best_params_,
        "MAE": mae,
        "R2": r2,
    }

    df_perf_top50 = pd.DataFrame([performance_top50])
    out_path = outdir / "model_performance_top50.csv"
    df_perf_top50.to_csv(out_path, index=False)
    print(f"Saved top-50 model performance to {out_path}")

    return df_perf_top50


def run_topk_sweep(
    merged_df: pd.DataFrame,
    avg_feature_importances: pd.Series,
    param_grid: dict,
    outdir: Path,
) -> (pd.DataFrame, list):
    """
    Step 3: Iterate over top_k features from 50 down to 5 (step -5), for rs=0..9.
    Saves model_performance_by_top_features.csv and performance_vs_features.png.

    Returns:
        df_perf: performance dataframe
        all_run_records: list with per-run predictions (for Step 4)
    """
    X_full = merged_df.drop(columns=["qmof_ids", "bandgaps"])
    X_full = X_full.fillna(X_full.mean())
    y_full = merged_df["bandgaps"]

    performance_records = []
    all_run_records = []

    print("Starting iterations over top_k features (from 50 down to 5):")
    for topk in tqdm(range(50, 0, -5), desc="Top_k Features"):
        topk = max(topk, 5)  # ensure at least 5 features
        current_features = avg_feature_importances.head(topk).index.tolist()

        for rs in range(10):
            X = X_full[current_features]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_full, test_size=0.1, random_state=rs
            )
            xgb = XGBRegressor(objective="reg:squarederror", random_state=42)
            grid_search = GridSearchCV(
                estimator=xgb,
                param_grid=param_grid,
                cv=5,
                scoring="r2",
                n_jobs=-1,
                verbose=0,
            )
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            performance_records.append(
                {
                    "top_features": topk,
                    "random_state": rs,
                    "MAE": mae,
                    "R2": r2,
                    "BestParams": grid_search.best_params_,
                }
            )

            run_record = {
                "top_features": topk,
                "random_state": rs,
                "y_test": y_test.reset_index(drop=True),
                "y_pred": pd.Series(y_pred).reset_index(drop=True),
            }
            all_run_records.append(run_record)

        print(f"Completed top_k = {topk}")

    df_perf = pd.DataFrame(performance_records)
    perf_path = outdir / "model_performance_by_top_features.csv"
    df_perf.to_csv(perf_path, index=False)
    print(f"Saved performance over top_k features to {perf_path}")

    # Average performance vs k
    avg_perf = df_perf.groupby("top_features").mean()[["MAE", "R2"]].reset_index()

    fig, ax1 = plt.subplots(figsize=(8, 6))
    color = "tab:blue"
    ax1.set_xlabel("Number of Top Features Used")
    ax1.set_ylabel("MAE", color=color)
    ax1.plot(avg_perf.top_features, avg_perf.MAE, color=color, marker="o", label="MAE")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("R2", color=color)
    ax2.plot(avg_perf.top_features, avg_perf.R2, color=color, marker="s", label="R2")
    ax2.tick_params(axis="y", labelcolor=color)

    fig.tight_layout()
    plt.title("Performance vs. Number of Top Features")
    fig_path = outdir / "performance_vs_features.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved performance plot to {fig_path}")

    return df_perf, all_run_records


def evaluate_best_model(
    df_perf: pd.DataFrame, all_run_records: list, outdir: Path
) -> None:
    """
    Step 4: Identify best model (highest R2 among all runs) and plot predicted vs. actual.
    Saves best_model_predictions.csv and best_model_pred_vs_actual.png.
    """
    df_best = df_perf.loc[df_perf["R2"].idxmax()]
    best_top_features = int(df_best.top_features)
    best_rs = int(df_best.random_state)

    best_run = None
    for rec in all_run_records:
        if rec["top_features"] == best_top_features and rec["random_state"] == best_rs:
            best_run = rec
            break

    if best_run is None:
        raise RuntimeError("Could not find best run record.")

    df_best_pred = pd.DataFrame(
        {
            "Actual": best_run["y_test"],
            "Predicted": best_run["y_pred"],
        }
    )
    best_pred_path = outdir / "best_model_predictions.csv"
    df_best_pred.to_csv(best_pred_path, index=False)
    print(f"Saved best model predictions to {best_pred_path}")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_best_pred, x="Actual", y="Predicted", color="purple")
    plt.xlabel("Actual Bandgap")
    plt.ylabel("Predicted Bandgap")
    plt.title(
        f"Best Model Prediction vs Actual (Top {best_top_features} Features, RS={best_rs})"
    )
    min_val = min(df_best_pred.Actual.min(), df_best_pred.Predicted.min())
    max_val = max(df_best_pred.Actual.max(), df_best_pred.Predicted.max())
    plt.plot([min_val, max_val], [min_val, max_val], "k--", lw=2)
    best_plot_path = outdir / "best_model_pred_vs_actual.png"
    plt.savefig(best_plot_path)
    plt.close()
    print(f"Saved best model scatter plot to {best_plot_path}")


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    print("Reading input data...")
    df = pd.read_csv(args.matminer_features)
    df_2 = pd.read_csv(args.mof_features)
    df_3 = pd.read_csv(args.mofdscribe_results)

    print("Building merged feature dataframe...")
    merged_df = build_merged_dataframe(df, df_2, df_3)

    # Hyperparameter grid (used in all grid searches)
    param_grid = {
        "n_estimators": [800, 1000],
        "max_depth": [7, 9],
        "learning_rate": [0.01, 0.05],
    }

    # Step 1: Feature importance
    avg_feature_importances = run_feature_importance(merged_df, param_grid, args.outdir)

    # Step 2: Top 50 experiment
    run_top50_experiment(merged_df, avg_feature_importances, param_grid, args.outdir)

    # Step 3: Sweep across top_k features
    df_perf, all_run_records = run_topk_sweep(
        merged_df, avg_feature_importances, param_grid, args.outdir
    )

    # Step 4: Best model analysis
    evaluate_best_model(df_perf, all_run_records, args.outdir)

    print("All XGB feature selection experiments completed.")


if __name__ == "__main__":
    main()