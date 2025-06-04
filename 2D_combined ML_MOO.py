#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
combined_rf_optimization.py
===========================

Key Steps:
1. Load data from multiple Excel files.
2. Identify target (objective) columns and categorical input features.
3. Perform one-hot encoding for categorical features globally.
4. Train Random Forest models for each target:
   - Scale input features using a global StandardScaler.
   - Tune 'max_depth' hyperparameter.
   - Generate diagnostic plots.
5. Train XGBoost models for each target (for comparison and diagnostics):
   - Use the same scaled input features.
   - Tune 'max_depth' hyperparameter.
   - Generate diagnostic plots.
6. Generate combined feature importance plots (RF vs XGB).
7. Define a Pymoo optimization problem using the trained RF models as surrogates.
8. Run NSGA-II to find Pareto-optimal solutions.
9. Plot optimization results (RadViz, 2D Pareto fronts) and save solutions to CSV.
10. Generate combined plot of all target distributions.
"""

import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.api import types as ptypes
from scipy.optimize import curve_fit
import itertools
import os
import re

from sklearn.model_selection import train_test_split, validation_curve, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.optimize import minimize

# --- Configuration ---
FILE_CONFIGS = [
    {
        'name': '2D_thermal_lidless.xlsx',
        'categorical_cols': ['Bulk silicon thickness', 'TIM1 k']
    },
    {
        'name': '2D_SJR_lidless.xlsx',
        'categorical_cols': ['Bulk silicon thickness', 'Bump solder', 'BGA solder']
    },
    {
        'name': '2D_assembly_lidless.xlsx',
        'categorical_cols': ['Bulk silicon thickness', 'Bump solder', 'BGA solder']
    }
]

TARGET_KEYWORDS = ["Theta", "DeltaW", "stress", "Warpage"]
RF_MAX_DEPTH_RANGE = [3, 5, 8, 10, 12, 15]
XGB_MAX_DEPTH_RANGE = [2, 3, 4, 5, 6, 8]  # From RF_XGBoost_v7
N_ESTIMATORS_DEFAULT = 100
TOP_N_FEATURES = 10  # From RF_XGBoost_v7

PLOTS_DIR = "optimization_plots_combined"
os.makedirs(PLOTS_DIR, exist_ok=True)
print(f"Plots will be saved to '{PLOTS_DIR}/' directory.")


# --- Helper Functions ---

def sanitize_filename_component(name):
    name = str(name)
    name = re.sub(r'[<>:"/\\|?*]+', '_', name)
    name = name.replace(' ', '_')
    return name


def build_column_defaults(dfs, cols_for_defaults):
    defaults = {}
    for c in cols_for_defaults:
        vals = []
        for df_item in dfs:
            if c in df_item.columns:
                numeric_col = pd.to_numeric(df_item[c], errors='coerce')
                valid_numeric_vals = numeric_col.dropna()
                if not valid_numeric_vals.empty:
                    vals.extend(valid_numeric_vals.tolist())
        defaults[c] = np.median(vals) if vals else 0.0
    return defaults


def get_best_param_from_r2_vc(estimator_class, base_params, X_train, y_train,
                              param_name, param_range, cv=5, n_jobs=-1, current_target_name_for_log=""):
    log_prefix = f"Target '{current_target_name_for_log}': " if current_target_name_for_log else ""
    model_name_for_log = estimator_class.__name__
    print(
        f"{log_prefix}Determining best '{param_name}' for {model_name_for_log} using R² validation curve (range: {param_range})...")

    current_tuning_params = base_params.copy()
    if param_name in current_tuning_params:
        del current_tuning_params[param_name]

    estimator = estimator_class(**current_tuning_params)
    try:
        X_train_np = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        y_train_np = y_train.values if isinstance(y_train, pd.Series) else y_train

        if np.any(np.isnan(X_train_np)) or np.any(np.isinf(X_train_np)):
            print(f"{log_prefix}Warning: NaNs or Infs found in X_train for {param_name} VC of {model_name_for_log}.")
        if np.any(np.isnan(y_train_np)) or np.any(np.isinf(y_train_np)):
            print(f"{log_prefix}Warning: NaNs or Infs found in y_train for {param_name} VC of {model_name_for_log}.")

        _, test_scores = validation_curve(
            estimator, X_train_np, y_train_np, param_name=param_name, param_range=param_range,
            cv=cv, scoring="r2", n_jobs=n_jobs, error_score=np.nan
        )
        raw_test_scores_mean = np.nanmean(test_scores, axis=1)

        if np.all(np.isnan(raw_test_scores_mean)):
            print(
                f"{log_prefix}Warning: All CV R² scores are NaN for {param_name} ({model_name_for_log}). Defaulting to first param: {param_range[0] if param_range else 'N/A'}.")
            return param_range[0] if param_range and len(param_range) > 0 else None

        best_param_idx = np.nanargmax(raw_test_scores_mean)
        param_range_array = np.array(param_range)
        best_param_value = param_range_array[best_param_idx]
        best_score = raw_test_scores_mean[best_param_idx]

        print(
            f"{log_prefix}Best {param_name} for {model_name_for_log} from R² VC: {best_param_value} (CV R² score: {best_score:.4f})")
        return best_param_value
    except Exception as e:
        print(f"{log_prefix}ERROR during R² validation_curve for {param_name} ({model_name_for_log}): {e}")
        if isinstance(X_train, pd.DataFrame): print(f"X_train dtypes:\n{X_train.dtypes}")
        return param_range[0] if param_range and len(param_range) > 0 else None


# --- Plotting Helper Functions (from RF_XGBoost_v7.py and bayes_v12.py) ---

def plot_target_distribution(target_series, full_target_name_for_title, ax_hist, ax_box):
    """Plots histogram and box plot of the target variable on given axes."""
    try:
        data_for_hist = target_series.dropna()
        if data_for_hist.empty: raise ValueError("Target series empty for hist")
        ax_hist.hist(data_for_hist, bins='auto', edgecolor='k', alpha=0.7)
        ax_hist.set_title(f'Histogram - {full_target_name_for_title}')
        ax_hist.set_xlabel('Target Value');
        ax_hist.set_ylabel('Frequency')
        ax_hist.grid(True, linestyle='--', alpha=0.7)
    except Exception as e:
        ax_hist.text(0.5, 0.5, f'Error: {e}', ha='center', va='center', transform=ax_hist.transAxes)
        ax_hist.set_title(f'Histogram - {full_target_name_for_title}\nError')

    try:
        data_for_box = target_series.dropna()
        if data_for_box.empty: raise ValueError("Target series empty for boxplot")
        ax_box.boxplot(data_for_box, vert=False, widths=0.7, patch_artist=True, medianprops={'color': 'black'})
        ax_box.set_title(f'Box Plot - {full_target_name_for_title}')
        ax_box.set_yticklabels([]);
        ax_box.set_xlabel('Target Value')
        ax_box.grid(True, linestyle='--', alpha=0.7)
    except Exception as e:
        ax_box.text(0.5, 0.5, f'Error: {e}', ha='center', va='center', transform=ax_box.transAxes)
        ax_box.set_title(f'Box Plot - {full_target_name_for_title}\nError')


def _plot_single_validation_curve(ax, estimator_instance, X, y, param_name, param_range, scoring, title_suffix, cv=5,
                                  n_jobs=-1):
    """Helper to plot a single validation curve on a given Axes object."""
    try:
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        y_np = y.values if isinstance(y, pd.Series) else y

        train_scores, test_scores = validation_curve(
            estimator_instance, X_np, y_np, param_name=param_name, param_range=param_range,
            cv=cv, scoring=scoring, n_jobs=n_jobs, error_score=np.nan
        )
    except Exception as e:
        ax.text(0.5, 0.5, f'VC Error:\n{e}', ha='center', va='center', transform=ax.transAxes, fontsize=8)
        ax.set_title(f"Validation Curve {title_suffix}\nError");
        return

    raw_train_scores_mean = np.nanmean(train_scores, axis=1)
    raw_test_scores_mean = np.nanmean(test_scores, axis=1)
    train_scores_std = np.nanstd(train_scores, axis=1)
    test_scores_std = np.nanstd(test_scores, axis=1)

    plot_train_scores_mean, plot_test_scores_mean = raw_train_scores_mean, raw_test_scores_mean
    ylabel = f"Score ({scoring})"
    if scoring == "neg_mean_squared_error":
        plot_train_scores_mean, plot_test_scores_mean = -raw_train_scores_mean, -raw_test_scores_mean
        ylabel = "Mean Squared Error (MSE)"
        max_val = 0  # Initialize max_val
        if not np.all(np.isnan(plot_train_scores_mean)): max_val = np.nanmax(
            plot_train_scores_mean[~np.isnan(plot_train_scores_mean)])
        if not np.all(np.isnan(plot_test_scores_mean)): max_val = max(max_val, np.nanmax(
            plot_test_scores_mean[~np.isnan(plot_test_scores_mean)]))
        ax.set_ylim(bottom=0, top=max_val * 1.1 if max_val > 0 else 1.0)
    elif scoring == "r2":
        ylabel = "R² Score";
        ax.set_ylim(-0.1, 1.1)

    param_range_array = np.array(param_range)
    plot_param_range = np.arange(len(param_range_array)) if not all(
        isinstance(pr, (int, float)) for pr in param_range_array) else param_range_array.astype(float)
    tick_labels = [str(pr) for pr in param_range_array]

    ax.plot(plot_param_range, plot_train_scores_mean, label="Training score", color="darkorange", marker='o', lw=2)
    ax.fill_between(plot_param_range, plot_train_scores_mean - train_scores_std,
                    plot_train_scores_mean + train_scores_std, alpha=0.2, color="darkorange")
    ax.plot(plot_param_range, plot_test_scores_mean, label="Cross-validation score", color="navy", marker='o', lw=2)
    ax.fill_between(plot_param_range, plot_test_scores_mean - test_scores_std, plot_test_scores_mean + test_scores_std,
                    alpha=0.2, color="navy")

    ax.set_xticks(plot_param_range);
    ax.set_xticklabels(tick_labels, rotation=30, ha='right')
    ax.set_title(f"Validation Curve {title_suffix} ({param_name})")
    ax.set_xlabel(str(param_name));
    ax.set_ylabel(ylabel)
    ax.legend(loc="best");
    ax.grid(True)


def _plot_single_learning_curve(ax, estimator_instance, X, y, scoring, title_suffix, cv=5, n_jobs=-1,
                                train_sizes=np.linspace(.1, 1.0, 5)):
    """Helper to plot a single learning curve on a given Axes object."""
    try:
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        y_np = y.values if isinstance(y, pd.Series) else y
        train_sizes_abs, train_scores, test_scores = learning_curve(
            estimator_instance, X_np, y_np, cv=cv, scoring=scoring, n_jobs=n_jobs, train_sizes=train_sizes,
            random_state=42, error_score=np.nan
        )
    except Exception as e:
        ax.text(0.5, 0.5, f'LC Error:\n{e}', ha='center', va='center', transform=ax.transAxes, fontsize=8)
        ax.set_title(f"Learning Curve {title_suffix}\nError");
        return

    raw_train_scores_mean = np.nanmean(train_scores, axis=1)
    raw_test_scores_mean = np.nanmean(test_scores, axis=1)
    train_scores_std = np.nanstd(train_scores, axis=1)
    test_scores_std = np.nanstd(test_scores, axis=1)

    plot_train_scores_mean, plot_test_scores_mean = raw_train_scores_mean, raw_test_scores_mean
    ylabel = f"Score ({scoring})"
    if scoring == "neg_mean_squared_error":
        plot_train_scores_mean, plot_test_scores_mean = -raw_train_scores_mean, -raw_test_scores_mean
        ylabel = "Mean Squared Error (MSE)"
        max_val = 0  # Initialize max_val
        if not np.all(np.isnan(plot_train_scores_mean)): max_val = np.nanmax(
            plot_train_scores_mean[~np.isnan(plot_train_scores_mean)])
        if not np.all(np.isnan(plot_test_scores_mean)): max_val = max(max_val, np.nanmax(
            plot_test_scores_mean[~np.isnan(plot_test_scores_mean)]))
        ax.set_ylim(bottom=0, top=max_val * 1.1 if max_val > 0 else 1.0)
    elif scoring == "r2":
        ylabel = "R² Score";
        ax.set_ylim(-0.1, 1.1)

    ax.plot(train_sizes_abs, plot_train_scores_mean, 'o-', color="darkorange", label="Training score", lw=2)
    ax.fill_between(train_sizes_abs, plot_train_scores_mean - train_scores_std,
                    plot_train_scores_mean + train_scores_std, alpha=0.1, color="darkorange")
    ax.plot(train_sizes_abs, plot_test_scores_mean, 'o-', color="navy", label="Cross-validation score", lw=2)
    ax.fill_between(train_sizes_abs, plot_test_scores_mean - test_scores_std, plot_test_scores_mean + test_scores_std,
                    alpha=0.1, color="navy")

    ax.set_title(f"Learning Curve {title_suffix}")
    ax.set_xlabel("Training examples");
    ax.set_ylabel(ylabel)
    ax.legend(loc="best");
    ax.grid(True)
    ax.set_xlim(left=0, right=train_sizes_abs.max() * 1.05 if train_sizes_abs.size > 0 else 1)


def _plot_single_actual_vs_predicted(ax, y_true, y_pred, dataset_name_str):
    """Helper to plot a single Actual vs. Predicted scatter plot."""
    if y_true is None or y_pred is None:
        ax.text(0.5, 0.5, 'Data N/A', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f"Actual vs. Predicted - {dataset_name_str}\nData N/A");
        return

    y_true_clean = y_true[~np.isnan(y_true) & ~np.isnan(y_pred)]
    y_pred_clean = y_pred[~np.isnan(y_true) & ~np.isnan(y_pred)]
    if len(y_true_clean) == 0:
        ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f"Actual vs. Predicted - {dataset_name_str}\nNo Valid Data");
        return

    ax.scatter(y_true_clean, y_pred_clean, alpha=0.6, edgecolors='k', s=50, label="Data points")
    all_vals = np.concatenate([y_true_clean, y_pred_clean])
    min_val, max_val = np.min(all_vals), np.max(all_vals)
    margin = (max_val - min_val) * 0.05 if (max_val - min_val) > 0 else 0.1
    plot_min, plot_max = min_val - margin, max_val + margin
    if plot_min == plot_max: plot_min -= 0.5; plot_max += 0.5

    ax.plot([plot_min, plot_max], [plot_min, plot_max], 'r--', lw=2, label="Ideal (y=x)")
    ax.set_xlabel("Actual Values");
    ax.set_ylabel("Predicted Values")
    ax.set_title(f"Actual vs. Predicted - {dataset_name_str}")
    ax.legend(loc="best");
    ax.grid(True)
    ax.set_xlim(plot_min, plot_max);
    ax.set_ylim(plot_min, plot_max)
    ax.set_aspect('equal', adjustable='box')


def generate_diagnostic_subplots(estimator_class, base_params,
                                 X_train, y_train, X_test, y_test,  # These should be scaled
                                 vc_param_name, vc_param_range,
                                 best_param_value_for_lc,
                                 main_plot_title, output_plot_filename_base):
    """Generates a 2x3 subplot figure with VCs, LCs, and Actual vs. Predicted plots."""
    print(f"  Generating diagnostic plot: {main_plot_title}")
    fig, axs = plt.subplots(2, 3, figsize=(24, 14))

    # Validation Curves
    vc_estimator_params_for_plot = base_params.copy()
    if vc_param_name in vc_estimator_params_for_plot: del vc_estimator_params_for_plot[vc_param_name]
    vc_estimator = estimator_class(**vc_estimator_params_for_plot)
    _plot_single_validation_curve(axs[0, 0], vc_estimator, X_train, y_train, vc_param_name, vc_param_range, "r2",
                                  "(R²)")
    _plot_single_validation_curve(axs[0, 1], vc_estimator, X_train, y_train, vc_param_name, vc_param_range,
                                  "neg_mean_squared_error", "(MSE)")

    model_params_for_lc_pred = base_params.copy()
    initial_model_fit_error = False
    if best_param_value_for_lc is not None:
        model_params_for_lc_pred[vc_param_name] = best_param_value_for_lc
    else:
        print(f"    Warning: best_param_value_for_lc for {vc_param_name} is None. Using first from range for LC/Pred.")
        if vc_param_range and len(vc_param_range) > 0:
            model_params_for_lc_pred[vc_param_name] = vc_param_range[0]
        else:
            initial_model_fit_error = True; print(
                f"    Error: vc_param_range empty for {vc_param_name}. Cannot create model.")

    if 'n_estimators' not in model_params_for_lc_pred:
        model_params_for_lc_pred['n_estimators'] = base_params.get('n_estimators', N_ESTIMATORS_DEFAULT)

    model_for_lc_pred = None
    if not initial_model_fit_error:
        try:
            model_for_lc_pred = estimator_class(**model_params_for_lc_pred)
        except Exception as e_inst:
            print(f"    Error instantiating model for LC/Pred plots: {e_inst}")
            initial_model_fit_error = True

    y_train_pred, y_test_pred = None, None
    predictions_available = False
    actual_fitting_error = False

    if model_for_lc_pred is not None:  # Changed from 'if model_for_lc_pred:' to 'if model_for_lc_pred is not None:'
        try:
            model_for_lc_pred.fit(X_train, y_train)
            y_train_pred = model_for_lc_pred.predict(X_train)
            y_test_pred = model_for_lc_pred.predict(X_test)
            predictions_available = True
        except Exception as e_fit:
            print(f"    Error fitting model for LC/Pred plots: {e_fit}");
            actual_fitting_error = True
    else:
        actual_fitting_error = True

    if predictions_available:
        _plot_single_actual_vs_predicted(axs[0, 2], y_train, y_train_pred, "Training Data")
    else:
        err_msg = 'Model fit/instantiation failed' if (initial_model_fit_error or actual_fitting_error) else 'Pred. N/A'
        axs[0, 2].text(0.5, 0.5, err_msg, ha='center', va='center', transform=axs[0, 2].transAxes)
        axs[0, 2].set_title(f"Actual vs. Predicted - Training\n{err_msg}")

    if model_for_lc_pred is not None and not actual_fitting_error:
        lc_title_suffix_r2 = f"(R², {vc_param_name}={model_params_for_lc_pred.get(vc_param_name, 'N/A')}, N_est={model_params_for_lc_pred.get('n_estimators', 'N/A')})"
        lc_title_suffix_mse = f"(MSE, {vc_param_name}={model_params_for_lc_pred.get(vc_param_name, 'N/A')}, N_est={model_params_for_lc_pred.get('n_estimators', 'N/A')})"
        _plot_single_learning_curve(axs[1, 0], model_for_lc_pred, X_train, y_train, "r2", lc_title_suffix_r2)
        _plot_single_learning_curve(axs[1, 1], model_for_lc_pred, X_train, y_train, "neg_mean_squared_error",
                                    lc_title_suffix_mse)
    else:
        err_msg = 'LC not plotted (model error)'
        axs[1, 0].text(0.5, 0.5, err_msg, ha='center', va='center');
        axs[1, 0].set_title(f"Learning Curve (R²)\n{err_msg}")
        axs[1, 1].text(0.5, 0.5, err_msg, ha='center', va='center');
        axs[1, 1].set_title(f"Learning Curve (MSE)\n{err_msg}")

    if predictions_available:
        _plot_single_actual_vs_predicted(axs[1, 2], y_test, y_test_pred, "Test Data")
    else:
        err_msg = 'Model fit/instantiation failed' if (initial_model_fit_error or actual_fitting_error) else 'Pred. N/A'
        axs[1, 2].text(0.5, 0.5, err_msg, ha='center', va='center', transform=axs[1, 2].transAxes)
        axs[1, 2].set_title(f"Actual vs. Predicted - Test\n{err_msg}")

    fig.suptitle(main_plot_title, fontsize=18)
    try:
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    except UserWarning as e:
        print(f"    UserWarning during tight_layout: {e}")

    output_filename = os.path.join(PLOTS_DIR, f"{output_plot_filename_base}_diagnostic_plots.png")
    plt.savefig(output_filename);
    print(f"  Saved diagnostic plots to {output_filename}")
    plt.close(fig)


def plot_combined_top_n_feature_importances(model1, model2, model1_name, model2_name,
                                            feature_names, top_n, plot_title_suffix, ax):
    """Plots combined feature importances from two models on a given ax."""
    if not feature_names:
        ax.text(0.5, 0.5, "Feat. names missing", ha='center', va='center');
        ax.set_title(f"FI - {plot_title_suffix}\nError");
        return

    m1_fi = model1.feature_importances_ if model1 and hasattr(model1, 'feature_importances_') else None
    m2_fi = model2.feature_importances_ if model2 and hasattr(model2, 'feature_importances_') else None

    if m1_fi is None and m2_fi is None:
        ax.text(0.5, 0.5, f'{model1_name} & {model2_name}\nNo importances', ha='center', va='center');
        ax.set_title(f"FI - {plot_title_suffix}\nError");
        return

    len_features = len(feature_names)
    imp_data = {'feature': list(feature_names)}
    m1_valid, m2_valid = False, False

    if m1_fi is not None and len(m1_fi) == len_features:
        imp_data[model1_name] = m1_fi; m1_valid = True
    else:
        imp_data[model1_name] = np.zeros(len_features);

    if m2_fi is not None and len(m2_fi) == len_features:
        imp_data[model2_name] = m2_fi; m2_valid = True
    else:
        imp_data[model2_name] = np.zeros(len_features);

    if not m1_valid and not m2_valid: ax.text(0.5, 0.5, 'No valid FI data', ha='center', va='center'); ax.set_title(
        f"FI - {plot_title_suffix}\nError"); return

    df_imp = pd.DataFrame(imp_data)
    df_imp['combined_score'] = df_imp[model1_name] + df_imp[model2_name] if m1_valid and m2_valid else (
        df_imp[model1_name] if m1_valid else df_imp[model2_name])
    top_features_df = df_imp.sort_values(by='combined_score', ascending=False).head(top_n)

    if top_features_df.empty:
        ax.text(0.5, 0.5, 'No features to display', ha='center', va='center');
        ax.set_title(f"Top {top_n} FI - {plot_title_suffix}\nNo data");
        return

    index = np.arange(len(top_features_df))
    bar_width = 0.35

    if m1_valid: ax.barh(index - bar_width / 2 if m2_valid else index, top_features_df[model1_name],
                         bar_width if m2_valid else bar_width * 1.5, label=model1_name, color='skyblue')
    if m2_valid: ax.barh(index + bar_width / 2 if m1_valid else index, top_features_df[model2_name],
                         bar_width if m1_valid else bar_width * 1.5, label=model2_name, color='lightcoral')

    ax.set_xlabel('Feature Importance');
    ax.set_ylabel('Features')
    ax.set_title(f"Top {top_n} Feature Importances - {plot_title_suffix}")
    ax.set_yticks(index);
    ax.set_yticklabels(top_features_df['feature'])
    ax.legend();
    ax.invert_yaxis()


def train_evaluate_final_model(model_class, model_params, X_train, y_train, X_test, y_test, model_name_with_target):
    """Trains and evaluates the final model, returns the trained model."""
    print(f"  Training and evaluating final {model_name_with_target}...")
    model = model_class(**model_params)
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        print(f"    ERROR during final model fitting for {model_name_with_target}: {e}")
        return None

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    print(f"  --- {model_name_with_target} Final Evaluation ---")
    print(f"    Training -> MSE: {train_mse:.4f}, R2: {train_r2:.4f}")
    print(f"    Test     -> MSE: {test_mse:.4f}, R2: {test_r2:.4f}")
    return model


def radviz_projection(objs):
    if objs is None or objs.shape[0] == 0: return np.empty((0, 2))
    m = objs.shape[1]
    if m == 0: return np.empty((objs.shape[0], 2))
    obj_min = objs.min(axis=0);
    obj_range = np.ptp(objs, axis=0)
    obj_range_safe = np.where(obj_range == 0, 1e-9, obj_range)
    norm = (objs - obj_min) / obj_range_safe;
    norm = 1.0 - norm
    angles = 2 * np.pi * np.arange(m) / m
    S_matrix = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    numerator = norm @ S_matrix
    denominator = norm.sum(axis=1, keepdims=True)
    denominator_safe = np.where(denominator == 0, 1e-12, denominator)
    return numerator / denominator_safe


def exp_func_type1(x, a, b, c):
    x_safe = np.clip(x * b, -700, 700);
    return a * np.exp(x_safe) + c


# --- Main Script Logic ---

def run_combined_workflow():
    print("--- Starting Combined RF Training and Optimization Workflow ---")
    all_target_series_for_global_plot = []
    all_feature_importance_data = []

    # 1. Load Excel data & Define Feature Sets
    print("\n1. Loading data and defining global feature sets...")
    raw_dfs = []
    for config in FILE_CONFIGS:
        fname = config['name']
        try:
            df = pd.read_excel(fname)
            raw_dfs.append(df);
            print(f"Successfully loaded {fname} with {len(df.columns)} columns.")
        except Exception as e:
            print(f"Error loading {fname}: {e}. Proceeding without it.");
            raw_dfs.append(pd.DataFrame())
    original_dfs_for_cat_analysis = [df.copy() for df in raw_dfs]

    initial_objective_names = []
    target_info_map = {}
    for idx, df_orig in enumerate(original_dfs_for_cat_analysis):
        if df_orig.empty: continue

        file_name_prefix_for_target = "unknownfile"
        if idx < len(FILE_CONFIGS):
            original_file_name = FILE_CONFIGS[idx]['name']
            file_name_prefix_for_target = sanitize_filename_component(os.path.splitext(original_file_name)[0])

        for col_name in df_orig.columns:
            is_target = any(keyword.lower() in col_name.lower() for keyword in TARGET_KEYWORDS)
            if is_target:
                if col_name not in initial_objective_names:
                    initial_objective_names.append(col_name)
                    target_info_map[col_name] = {'df_index': idx, 'file_name_prefix': file_name_prefix_for_target}
    initial_objective_names = sorted(list(set(initial_objective_names)))
    print(f"Identified potential objectives ({len(initial_objective_names)}): {initial_objective_names}")

    globally_identified_cat_cols_set = set()
    for config_idx, config in enumerate(FILE_CONFIGS):
        if config_idx < len(original_dfs_for_cat_analysis) and not original_dfs_for_cat_analysis[config_idx].empty:
            current_df = original_dfs_for_cat_analysis[config_idx]
            for cat_col in config.get('categorical_cols', []):
                if cat_col in current_df.columns and cat_col not in initial_objective_names:
                    globally_identified_cat_cols_set.add(cat_col)
            for col_name in current_df.columns:
                if col_name not in initial_objective_names and col_name not in globally_identified_cat_cols_set:
                    if "discrete" in col_name.lower() or ptypes.is_object_dtype(current_df[col_name]):
                        globally_identified_cat_cols_set.add(col_name)
    final_cat_cols_to_process = sorted(list(globally_identified_cat_cols_set))
    print(f"Final identified categorical/discrete input columns for OHE: {final_cat_cols_to_process}")

    CAT_COL_DETAILS = {}
    globally_generated_ohe_col_names = []
    for col_name in final_cat_cols_to_process:
        all_vals = []
        for df_orig in original_dfs_for_cat_analysis:
            if col_name in df_orig.columns: all_vals.extend(df_orig[col_name].dropna().astype(str).tolist())
        unique_cats = sorted(list(set(all_vals)))
        if not unique_cats: print(f"Warn: No unique cats for '{col_name}'. Skipping OHE."); continue
        ohe_names = [f"{sanitize_filename_component(col_name)}_{sanitize_filename_component(s_val)}" for s_val in
                     unique_cats]
        CAT_COL_DETAILS[col_name] = {'categories': unique_cats, 'one_hot_names': ohe_names}
        globally_generated_ohe_col_names.extend(ohe_names)
    globally_generated_ohe_col_names = sorted(list(set(globally_generated_ohe_col_names)))
    print(f"Total unique OHE column names generated globally: {len(globally_generated_ohe_col_names)}")

    processed_dfs_after_ohe = []
    target_to_ohe_df_map = {}
    for df_idx, df_orig in enumerate(original_dfs_for_cat_analysis):
        if df_orig.empty: processed_dfs_after_ohe.append(df_orig.copy()); continue
        df_updated = df_orig.copy()
        cats_in_this_df = [c for c in final_cat_cols_to_process if c in df_updated.columns]
        non_cat_part = df_updated.drop(columns=cats_in_this_df,
                                       errors='ignore') if cats_in_this_df else df_updated.copy()
        ohe_gen_part = pd.DataFrame(index=non_cat_part.index)
        if cats_in_this_df:
            cat_part_to_encode = df_updated[cats_in_this_df].astype(str)
            s_prefixes = {col: sanitize_filename_component(col) for col in cats_in_this_df}
            ohe_gen_part = pd.get_dummies(cat_part_to_encode, columns=cats_in_this_df, prefix=s_prefixes,
                                          prefix_sep='_', dummy_na=False)

        ohe_reindexed = ohe_gen_part.reindex(columns=globally_generated_ohe_col_names, fill_value=0)
        df_full_updated = pd.concat([non_cat_part, ohe_reindexed], axis=1)
        processed_dfs_after_ohe.append(df_full_updated)

    for target_name, info in target_info_map.items():
        target_to_ohe_df_map[target_name] = processed_dfs_after_ohe[info['df_index']]
    print("One-hot encoding and DataFrame updates complete.")

    temp_rf_feat_set = set()
    for df_ohe in processed_dfs_after_ohe:
        if not df_ohe.empty:
            for col in df_ohe.columns:
                if col not in initial_objective_names: temp_rf_feat_set.add(col)
    rf_input_features = sorted(list(temp_rf_feat_set))
    print(f"Total global input features for models (rf_input_features): {len(rf_input_features)}")
    if not rf_input_features: raise SystemExit("CRITICAL: No input features for models. Exiting.")

    pymoo_opt_vars_continuous = sorted([col for col in rf_input_features if
                                        col not in globally_generated_ohe_col_names and col not in final_cat_cols_to_process])
    pymoo_opt_vars_categorical_original = sorted(final_cat_cols_to_process)
    pymoo_opt_vars_all = pymoo_opt_vars_continuous + pymoo_opt_vars_categorical_original
    print(
        f"Continuous variables for Pymoo ({len(pymoo_opt_vars_continuous)}): {pymoo_opt_vars_continuous[:min(5, len(pymoo_opt_vars_continuous))]}...")
    print(
        f"Original categorical variables for Pymoo ({len(pymoo_opt_vars_categorical_original)}): {pymoo_opt_vars_categorical_original}")
    for cat_name, details in CAT_COL_DETAILS.items():
        if cat_name in pymoo_opt_vars_all: details['pymoo_index'] = pymoo_opt_vars_all.index(cat_name)

    defaults_for_rf = build_column_defaults(processed_dfs_after_ohe, rf_input_features)
    print(f"Computed defaults for {len(defaults_for_rf)} model input features.")

    # 2. Fit RF and XGBoost models for each objective
    print("\n2. Fitting RF and XGBoost surrogates with hyperparameter tuning...")
    fitted_rf_models_dict = {}
    global_problem_scaler = None

    for target_idx, target_name in enumerate(initial_objective_names):
        print(f"\n  --- Processing Target: {target_name} ({target_idx + 1}/{len(initial_objective_names)}) ---")
        if target_name not in target_to_ohe_df_map: print(f"  Warn: No OHE DF for '{target_name}'. Skip."); continue
        df_for_target = target_to_ohe_df_map[target_name]
        if df_for_target.empty or target_name not in df_for_target.columns: print(
            f"  Skipping '{target_name}': DF empty/target missing."); continue

        current_input_cols = [col for col in df_for_target.columns if col in rf_input_features]
        X_intermediate = df_for_target[current_input_cols]
        X_reindexed = X_intermediate.reindex(columns=rf_input_features, fill_value=np.nan)
        X_filled = X_reindexed.fillna(defaults_for_rf)
        y = df_for_target[target_name].fillna(df_for_target[target_name].median())

        if X_filled.empty or y.empty: print(f"  Skipping '{target_name}': X or y empty."); continue
        if X_filled.isnull().values.any(): X_filled = X_filled.fillna(0); print(
            f"  Warn: X for {target_name} had NaNs after reindex/default fill. Filled with 0.")

        X_train_orig, X_test_orig, y_train, y_test = train_test_split(X_filled, y, test_size=0.2, random_state=42)
        if X_train_orig.empty: print(f"  Skipping '{target_name}': X_train empty."); continue

        if global_problem_scaler is None:
            print(f"    Fitting global_problem_scaler on X_train_orig of first target '{target_name}'.")
            global_problem_scaler = StandardScaler().fit(X_train_orig)

        Xs_train = global_problem_scaler.transform(X_train_orig)
        Xs_test = global_problem_scaler.transform(X_test_orig)

        file_prefix = "file"
        if target_name in target_info_map and 'file_name_prefix' in target_info_map[target_name]:
            file_prefix = target_info_map[target_name]['file_name_prefix']
        plot_target_name_for_title = f"{file_prefix}_{sanitize_filename_component(target_name)}"
        all_target_series_for_global_plot.append((df_for_target[target_name], plot_target_name_for_title))

        # --- Random Forest Model ---
        print(f"\n    ----- Random Forest for Target: {target_name} -----")
        rf_base_params = {'random_state': 42, 'n_estimators': N_ESTIMATORS_DEFAULT, 'n_jobs': -1}
        best_rf_depth = get_best_param_from_r2_vc(RandomForestRegressor, rf_base_params, Xs_train, y_train, "max_depth",
                                                  RF_MAX_DEPTH_RANGE, current_target_name_for_log=target_name)

        trained_rf_model = None
        if best_rf_depth is not None:
            final_rf_params = {**rf_base_params, 'max_depth': best_rf_depth}
            generate_diagnostic_subplots(RandomForestRegressor, final_rf_params, Xs_train, y_train, Xs_test, y_test,
                                         "max_depth", RF_MAX_DEPTH_RANGE, best_rf_depth,
                                         f"RF Diagnostics - {target_name}", f"{plot_target_name_for_title}_RF")
            trained_rf_model = train_evaluate_final_model(RandomForestRegressor, final_rf_params, Xs_train, y_train,
                                                          Xs_test, y_test, f"Final RF ({target_name})")
            if trained_rf_model: fitted_rf_models_dict[target_name] = trained_rf_model
        else:
            print(f"    Skipping RF training for '{target_name}' due to depth determination error.")

        # --- XGBoost Model ---
        print(f"\n    ----- XGBoost for Target: {target_name} -----")
        xgb_base_params = {'random_state': 42, 'n_estimators': N_ESTIMATORS_DEFAULT, 'n_jobs': -1,
                           'objective': 'reg:squarederror', 'eval_metric': 'rmse'}
        best_xgb_depth = get_best_param_from_r2_vc(XGBRegressor, xgb_base_params, Xs_train, y_train, "max_depth",
                                                   XGB_MAX_DEPTH_RANGE, current_target_name_for_log=target_name)

        trained_xgb_model = None
        if best_xgb_depth is not None:
            final_xgb_params = {**xgb_base_params, 'max_depth': best_xgb_depth}
            generate_diagnostic_subplots(XGBRegressor, final_xgb_params, Xs_train, y_train, Xs_test, y_test,
                                         "max_depth", XGB_MAX_DEPTH_RANGE, best_xgb_depth,
                                         f"XGBoost Diagnostics - {target_name}", f"{plot_target_name_for_title}_XGB")
            trained_xgb_model = train_evaluate_final_model(XGBRegressor, final_xgb_params, Xs_train, y_train, Xs_test,
                                                           y_test, f"Final XGBoost ({target_name})")
        else:
            print(f"    Skipping XGBoost training for '{target_name}' due to depth determination error.")

        all_feature_importance_data.append(
            (trained_rf_model, trained_xgb_model, list(X_filled.columns), plot_target_name_for_title))

    active_objective_names_for_pymoo = [name for name in initial_objective_names if name in fitted_rf_models_dict]
    active_rf_models_for_pymoo = [fitted_rf_models_dict[name] for name in active_objective_names_for_pymoo]

    if not active_objective_names_for_pymoo or global_problem_scaler is None:
        raise SystemExit("CRITICAL: No RF models for Pymoo or global_scaler missing. Exiting.")
    print(
        f"\nRF models for Pymoo ready. Active objectives ({len(active_objective_names_for_pymoo)}): {active_objective_names_for_pymoo}")

    # 3. Define the multi-objective problem
    print("\n3. Defining MOO problem...")
    var_bounds_list = []
    all_orig_dfs_for_bounds = [df.copy() for df in original_dfs_for_cat_analysis if not df.empty]
    for var_name in pymoo_opt_vars_all:
        if var_name in pymoo_opt_vars_continuous:
            min_v, max_v = np.inf, -np.inf;
            found = False
            for df_src in all_orig_dfs_for_bounds:
                if var_name in df_src.columns and pd.api.types.is_numeric_dtype(df_src[var_name]):
                    num_data = pd.to_numeric(df_src[var_name], errors='coerce').dropna()
                    if not num_data.empty: min_v = min(min_v, num_data.min()); max_v = max(max_v,
                                                                                           num_data.max()); found = True
            if not found or min_v == np.inf:
                min_v, max_v = 0.0, 1.0; print(f"Warn: Bounds for '{var_name}' not found. Default [0,1].")
            elif min_v >= max_v:
                dlt = max(0.01, abs(max_v * 0.01)); min_v, max_v = min(min_v, max_v) - dlt, max(min_v, max_v) + dlt
            var_bounds_list.append([min_v, max_v])
        elif var_name in CAT_COL_DETAILS:
            var_bounds_list.append([0, len(CAT_COL_DETAILS[var_name]['categories']) - 1])
        else:
            var_bounds_list.append([0.0, 1.0]); print(f"Err: Var '{var_name}' bounds undefined. Default [0,1].")
    bounds_np_array = np.array(var_bounds_list, dtype=float)

    class SurrogateProblem(Problem):
        def __init__(self):
            super().__init__(n_var=len(pymoo_opt_vars_all), n_obj=len(active_objective_names_for_pymoo),
                             n_constr=0, xl=bounds_np_array[:, 0], xu=bounds_np_array[:, 1])

        def _evaluate(self, X_batch, out, *args, **kwargs):
            n_sols = X_batch.shape[0]
            X_rf = pd.DataFrame(columns=rf_input_features, index=range(n_sols))
            for cont_idx, cont_name in enumerate(pymoo_opt_vars_continuous):
                if cont_name in X_rf.columns: X_rf[cont_name] = X_batch[:, pymoo_opt_vars_all.index(cont_name)]
            for cat_name, details in CAT_COL_DETAILS.items():
                if 'pymoo_index' not in details:
                    for ohe_n in details['one_hot_names']:
                        if ohe_n in X_rf.columns: X_rf[ohe_n] = defaults_for_rf.get(ohe_n, 0.0)
                    continue
                int_choices = np.clip(np.round(X_batch[:, details['pymoo_index']]).astype(int), 0,
                                      len(details['categories']) - 1)
                for ohe_n in details['one_hot_names']:
                    if ohe_n in X_rf.columns: X_rf[ohe_n] = 0.0
                for sol_i, choice_i in enumerate(int_choices):
                    chosen_ohe = details['one_hot_names'][choice_i]
                    if chosen_ohe in X_rf.columns: X_rf.loc[sol_i, chosen_ohe] = 1.0
            for col_fill in rf_input_features:
                if col_fill not in X_rf.columns or X_rf[col_fill].isnull().all():
                    X_rf[col_fill] = defaults_for_rf.get(col_fill, 0.0)
                elif X_rf[col_fill].isnull().any():
                    X_rf[col_fill].fillna(defaults_for_rf.get(col_fill, 0.0), inplace=True)
            X_rf_ordered = X_rf[rf_input_features]
            if global_problem_scaler is None: print("Err: Scaler missing in _evaluate."); out["F"] = np.full(
                (n_sols, len(active_objective_names_for_pymoo)), np.nan); return
            try:
                X_scaled = global_problem_scaler.transform(X_rf_ordered)
            except Exception as e:
                print(f"Err scaling in _evaluate: {e}"); out["F"] = np.full(
                    (n_sols, len(active_objective_names_for_pymoo)), np.nan); return
            out["F"] = np.column_stack([model.predict(X_scaled) for model in active_rf_models_for_pymoo])

    problem = SurrogateProblem()
    print("MOO problem defined.")

    # 4. Run NSGA-II
    print(f"\n4. Running NSGA-II for {len(active_objective_names_for_pymoo)} objectives...")
    if not active_objective_names_for_pymoo:
        print("Skipping NSGA-II: No objectives."); res = None
    else:
        algo = NSGA2(pop_size=100, eliminate_duplicates=True)
        res = minimize(problem, algo, termination=get_termination("n_gen", 100), seed=42, verbose=True)
        print("NSGA-II optimization finished.")

    pareto_F = res.F if res and hasattr(res, 'F') else None
    pareto_X = res.X if res and hasattr(res, 'X') else None
    if pareto_F is None or len(pareto_F) == 0: print("Pareto front empty. Exiting early."); return

    # 5. Plotting and CSV Output for Optimization
    XY_radviz = radviz_projection(pareto_F) if pareto_F is not None else None
    print("\n5a. Generating RadViz plots (Optimization)...")
    if XY_radviz is not None and XY_radviz.shape[0] > 0 and pareto_F is not None and pareto_F.shape[0] > 0:
        if XY_radviz.ndim < 2 or XY_radviz.shape[1] < 2:
            print("Warn: RadViz projection insufficient dims.")
        else:
            num_obj_plot = pareto_F.shape[1]
            if num_obj_plot < 2:
                print("Skip Radviz: Need >= 2 objectives.")
            else:
                angles = 2 * np.pi * np.arange(num_obj_plot) / num_obj_plot
                anchors = np.stack([np.cos(angles), np.sin(angles)], axis=1)
                for i, name in enumerate(active_objective_names_for_pymoo):
                    fig, ax = plt.subplots(figsize=(9, 9))
                    sc = ax.scatter(XY_radviz[:, 0], XY_radviz[:, 1], c=pareto_F[:, i], cmap="viridis", s=60, ec="k",
                                    alpha=0.7, label="Pareto Designs")
                    poly_x = np.append(anchors[:, 0], anchors[0, 0]);
                    poly_y = np.append(anchors[:, 1], anchors[0, 1])
                    ax.plot(poly_x, poly_y, "--", c="r", lw=1.5, label="Objective Anchors")
                    ax.scatter(anchors[:, 0], anchors[:, 1], marker="^", c="r", s=120)
                    for il, lab in enumerate(active_objective_names_for_pymoo):
                        ax.text(anchors[il, 0] * 1.08, anchors[il, 1] * 1.08, lab, c="r", ha="center", va="center",
                                fontsize=8)  # Adjusted offset
                    ax.set_title(f"RadViz (Color by {name})");
                    ax.set_xlabel("RadViz X");
                    ax.set_ylabel("RadViz Y");
                    ax.axhline(0, c='k', lw=0.5, ls='--');
                    ax.axvline(0, c='k', lw=0.5, ls='--');
                    ax.set_aspect('equal');
                    plt.colorbar(sc, ax=ax, label=name);
                    ax.legend(loc="upper right")
                    plt.tight_layout()
                    sfname = sanitize_filename_component(name);
                    plt.savefig(os.path.join(PLOTS_DIR, f"radviz_opt_{sfname}.png"));
                    plt.close(fig)
                    print(f"Saved RadViz opt plot: radviz_opt_{sfname}.png")
    else:
        print("Skipping RadViz (Opt) plots.")

    print("\n5b. Generating 2D scatter plots (Optimization)...")
    if pareto_F is not None and len(pareto_F) > 0 and pareto_F.shape[1] >= 2:
        for i1, i2 in itertools.combinations(range(pareto_F.shape[1]), 2):
            obj1_name_s, obj2_name_s = active_objective_names_for_pymoo[i1], active_objective_names_for_pymoo[i2]
            plot_title_s = f"{obj1_name_s} vs {obj2_name_s}"

            fig_scatter_s, ax_scatter_s = plt.subplots(figsize=(9, 8))
            ax_scatter_s.scatter(pareto_F[:, i1], pareto_F[:, i2], color='tab:blue',
                                 s=60, edgecolor='black', label='Pareto Front Points', zorder=5)

            if len(pareto_F) > 1:
                df_for_interp = pd.DataFrame({'x': pareto_F[:, i1], 'y': pareto_F[:, i2]})
                unique_pf_points = df_for_interp.sort_values(by=['x', 'y']).drop_duplicates(subset=['x'], keep='first')
                interp_x_final = unique_pf_points['x'].values
                interp_y_final = unique_pf_points['y'].values

                if len(interp_x_final) >= 3:
                    try:
                        initial_guess_exp1 = [np.max(interp_y_final), -0.1, np.min(interp_y_final)]
                        bounds_exp1 = ([0, -np.inf, -np.inf], [np.inf, 0, np.inf])
                        params_exp1, _ = curve_fit(exp_func_type1, interp_x_final, interp_y_final,
                                                   p0=initial_guess_exp1, bounds=bounds_exp1, maxfev=5000)
                        x_smooth_exp1 = np.linspace(interp_x_final.min(), interp_x_final.max(), 300)
                        y_smooth_exp1 = exp_func_type1(x_smooth_exp1, *params_exp1)
                        ax_scatter_s.plot(x_smooth_exp1, y_smooth_exp1, linestyle='--', color='red',
                                          lw=2, label='Pareto Curve (Exp: ae^(bx)+c)', zorder=3)
                    except (RuntimeError, ValueError) as e_exp_fit:
                        print(
                            f"Exponential fit failed for plot '{plot_title_s}': {e_exp_fit}. Plotting linear fallback.")
                        if len(interp_x_final) >= 2:
                            ax_scatter_s.plot(interp_x_final, interp_y_final, '-', color='darkcyan', lw=1.5,
                                              label='Pareto Front (Line Fallback)', zorder=1)
                elif len(interp_x_final) >= 2:
                    ax_scatter_s.plot(interp_x_final, interp_y_final, '-', color='darkcyan', lw=1.5,
                                      label='Pareto Front (Line)', zorder=1)

            ax_scatter_s.set_xlabel(f"{obj1_name_s}");
            ax_scatter_s.set_ylabel(f"{obj2_name_s}")
            ax_scatter_s.set_title(plot_title_s);
            ax_scatter_s.legend(fontsize='small', loc='best')
            ax_scatter_s.grid(True, linestyle='--', alpha=0.5);
            plt.tight_layout()

            stitle = sanitize_filename_component(plot_title_s).replace('vs', 'VS');
            plt.savefig(os.path.join(PLOTS_DIR, f"scatter_opt_{stitle}.png"));
            plt.close(fig_scatter_s)
            print(f"Saved Scatter opt plot: scatter_opt_{stitle}.png")
    else:
        print("Skipping 2D Scatter (Opt) plots.")

    print("\n5c. Mapping Pareto Solutions & CSV Output (Optimization)...")
    if pareto_F is not None and pareto_X is not None and len(pareto_F) == len(pareto_X):
        X_df_out = pd.DataFrame(pareto_X, columns=pymoo_opt_vars_all).copy()
        for cat_name, details in CAT_COL_DETAILS.items():
            if cat_name in X_df_out.columns:
                int_choices = np.clip(np.round(X_df_out[cat_name].values).astype(int), 0,
                                      len(details['categories']) - 1)
                X_df_out[cat_name] = [details['categories'][c] for c in int_choices]
        obj_cols = [f"Obj_{sanitize_filename_component(n)}" for n in active_objective_names_for_pymoo]
        F_df = pd.DataFrame(pareto_F, columns=obj_cols)
        radviz_df = pd.DataFrame(XY_radviz, columns=["RadViz_X", "RadViz_Y"]) if XY_radviz is not None and len(
            XY_radviz) == len(F_df) else pd.DataFrame(index=F_df.index)

        final_df_list = []
        if not radviz_df.empty: final_df_list.append(radviz_df)
        final_df_list.append(F_df)
        final_df_list.append(X_df_out)

        results_df = pd.concat(final_df_list, axis=1)
        csv_p = os.path.join(PLOTS_DIR, "pareto_solutions_designs_opt.csv")
        results_df.to_csv(csv_p, index_label="Sol_Index");
        print(f"Saved Pareto solutions to {csv_p}")
    else:
        print("Skipping CSV output (Opt).")

    # 6. Generate Combined Diagnostic Plots (at the end)
    num_all_targets = len(all_target_series_for_global_plot)
    if num_all_targets > 0:
        fig_all_dist, axs_all_dist = plt.subplots(num_all_targets, 2, figsize=(14, 5 * num_all_targets), squeeze=False)
        fig_all_dist.suptitle("All Target Variable Distributions", fontsize=16, y=0.99)
        for i, (series, title) in enumerate(all_target_series_for_global_plot):
            plot_target_distribution(series, title, axs_all_dist[i, 0], axs_all_dist[i, 1])
        try:
            fig_all_dist.tight_layout(rect=[0, 0, 1, 0.98])
        except UserWarning as e:
            print(f"Warn tight_layout (all_dist): {e}")
        dist_path = os.path.join(PLOTS_DIR, "all_targets_distributions.png")
        fig_all_dist.savefig(dist_path);
        print(f"\nSaved combined target distributions to {dist_path}");
        plt.close(fig_all_dist)

    num_fi_plots = len(all_feature_importance_data)
    if num_fi_plots > 0:
        fig_fi, axs_fi = plt.subplots(num_fi_plots, 1, figsize=(12, 7 * num_fi_plots), squeeze=False)
        fig_fi.suptitle("Combined Feature Importances (RF vs XGB) per Target", fontsize=16, y=0.99)
        for i, (rf_model, xgb_model, f_names, suffix) in enumerate(all_feature_importance_data):
            plot_combined_top_n_feature_importances(rf_model, xgb_model, "RandomForest", "XGBoost", f_names,
                                                    TOP_N_FEATURES, suffix, axs_fi[i, 0])
        try:
            fig_fi.tight_layout(rect=[0, 0, 1, 0.98])
        except UserWarning as e:
            print(f"Warn tight_layout (all_fi): {e}")
        fi_path = os.path.join(PLOTS_DIR, "all_targets_feature_importances.png")
        fig_fi.savefig(fi_path);
        print(f"Saved combined feature importances to {fi_path}");
        plt.close(fig_fi)

    print("\n--- Script finished. ---")


if __name__ == "__main__":
    run_combined_workflow()
