#!/usr/bin/env python
import argparse
from pathlib import Path
import warnings

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBRegressor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SHAP analysis for XGBoost MOF bandgap model using top-N features."
    )

    # Data paths
    parser.add_argument(
        "--matminer-features",
        type=Path,
        default=Path("data/raw/matminer_features.csv"),
        help="Path to matminer features CSV.",
    )
    parser.add_argument(
        "--mof-features",
        type=Path,
        default=Path("data/raw/mof_features_combined.csv"),
        help="Path to MOF combined features CSV.",
    )
    parser.add_argument(
        "--mofdscribe-results",
        type=Path,
        default=Path("data/raw/mofdscribe_results.csv"),
        help="Path to mofdscribe results CSV.",
    )
    parser.add_argument(
        "--avg-importances",
        type=Path,
        default=Path("results/xgb/avg_feature_importances.csv"),
        help="Path to avg_feature_importances.csv.",
    )

    # SHAP / model options
    parser.add_argument(
        "--top-n",
        type=int,
        default=30,
        help="Number of top features to use for SHAP (default: 30).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.1,
        help="Test set fraction (default: 0.1).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for train/test split (default: 42).",
    )

    # Output
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results/xgb/shap"),
        help="Directory to save SHAP plots and metrics.",
    )

    return parser.parse_args()


def build_merged_df(df: pd.DataFrame, df_2: pd.DataFrame, df_3: pd.DataFrame) -> pd.DataFrame:
    """Rebuild merged_df with all feature groups, as in your original script."""

    # Drop columns from df_3 if present
    drop_cols = [c for c in ["uc_volume", "density"] if c in df_3.columns]
    if drop_cols:
        df_3 = df_3.drop(columns=drop_cols)

    # Merge on qmof_ids / qmof_id
    merged_df = pd.merge(df, df_2, left_on="qmof_ids", right_on="qmof_id", how="inner")
    merged_df = pd.merge(merged_df, df_3, left_on="qmof_ids", right_on="qmof_id", how="inner")

    # --- feature groups (same as your code) ---
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

    # Build sub-dataframes (for clarity/sanity)
    bond_length_df = merged_df[bond_length_columns]
    neighbor_distance_df = merged_df[neighbor_distance_columns]
    structural_info_df = merged_df[structural_info]
    structural_metrics_df = merged_df[structural_metrics]
    ordering_parameters_df = merged_df[ordering_parameters]

    # RDF columns: rename [] to _
    rdf_df = merged_df[rdf_columns].copy()
    rdf_df.columns = [col.replace("[", "_").replace("]", "_") for col in rdf_df.columns]

    # Spacegroup
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


def load_top_features(avg_importances_path: Path, top_n: int) -> list[str]:
    """Load top-N feature names from avg_feature_importances.csv.

    Supports both:
    - index = feature name, column 'average_importance'
    - column 'feature' + 'average_importance'
    """

    df_imp = pd.read_csv(avg_importances_path)

    if "feature" in df_imp.columns:
        # Explicit feature column
        features = df_imp["feature"]
    else:
        # Assume first column is feature name (e.g., 'Unnamed: 0')
        features = df_imp.iloc[:, 0]

    top_n = min(top_n, len(features))
    top_features = features.head(top_n).tolist()
    return top_features


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore")

    print("Reading input data...")
    df = pd.read_csv(args.matminer_features)
    df_2 = pd.read_csv(args.mof_features)
    df_3 = pd.read_csv(args.mofdscribe_results)

    print("Building merged feature dataframe...")
    merged_df = build_merged_df(df, df_2, df_3)

    print(f"Loading top-{args.top_n} features from {args.avg_importances}...")
    top_features = load_top_features(args.avg_importances, args.top_n)

    # Prepare X and y
    X_full = merged_df[top_features].copy()
    X_full = X_full.fillna(X_full.mean())
    y_full = merged_df["bandgaps"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_full,
        y_full,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    # Hyperparameter grid (same as your previous setup)
    param_grid = {
        "n_estimators": [800, 1000],
        "max_depth": [7, 9],
        "learning_rate": [0.01, 0.05],
    }

    print("Running XGBoost GridSearchCV...")
    xgb_base = XGBRegressor(objective="reg:squarederror", random_state=args.random_state)
    grid_search = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)

    best_xgb_model = grid_search.best_estimator_
    y_pred = best_xgb_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Best XGBoost Parameters: {grid_search.best_params_}")
    print(f"MAE (test): {mae:.4f}")
    print(f"R²  (test): {r2:.4f}")

    # SHAP analysis
    print("Computing SHAP values...")
    shap.initjs()

    explainer = shap.TreeExplainer(best_xgb_model)
    shap_values = explainer.shap_values(X_test)

    # Custom colormap (two colors as given)
    custom_colors = ["#FF513D", "#3D91FF"]  # e.g. low → red, high → blue
    custom_cmap = mcolors.LinearSegmentedColormap.from_list(
        "my_custom_cmap", custom_colors, N=256
    )

    # Plot SHAP summary
    plt.rcParams.update({"font.size": 6})
    plt.figure(figsize=(10, 10))
    shap.summary_plot(shap_values, X_test, show=False, cmap=custom_cmap)
    plt.title(
        f"SHAP Summary Plot (Top {len(top_features)} Features)",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()

    out_path = args.outdir / f"shap_summary_top{len(top_features)}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"SHAP summary plot saved to: {out_path}")


if __name__ == "__main__":
    main()