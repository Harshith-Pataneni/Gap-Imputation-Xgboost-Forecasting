import importlib
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_absolute_error, mean_squared_error

plt.rcParams.update({
    'text.color':           'black',
    'font.family':          'DejaVu Sans',
    'font.weight':          'bold',
    'font.size':            17,
    'axes.labelcolor':      'black',
    'axes.labelweight':     'bold',
    'axes.labelsize':       18,
    'axes.titlesize':       20,
    'axes.titleweight':     'bold',
    'axes.edgecolor':       'black',
    'axes.linewidth':       2.0,
    'xtick.color':          'black',
    'ytick.color':          'black',
    'xtick.labelsize':      15,
    'ytick.labelsize':      15,
    'xtick.major.width':    2.0,
    'ytick.major.width':    2.0,
    'xtick.major.size':     6,
    'ytick.major.size':     6,
    'legend.fontsize':      15,
    'legend.title_fontsize':15,
    'legend.edgecolor':     'black',
    'legend.framealpha':    1.0,
    'figure.titlesize':     22,
    'figure.titleweight':   'bold',
    'savefig.facecolor':    'white',
    'savefig.edgecolor':    'black',
})

import xgboost_model
import interpolation_methods as im
importlib.reload(xgboost_model)
importlib.reload(im)

GAP   = xgboost_model.FOLDER_GAP_PLOTS
RES   = xgboost_model.FOLDER_RESULTS
COMP  = xgboost_model.FOLDER_COMPARISON
EXTRA = xgboost_model.FOLDER_EXTRA
ROOT  = xgboost_model.OUTPUT_ROOT
TEST_START = xgboost_model.TEST_START

COND_DIR = os.path.join(ROOT, "interpolation_comparison")
os.makedirs(COND_DIR, exist_ok=True)

METHOD_COLORS = {
    'Linear':               '#E53935',
    'Spline':               '#FF9800',
    'Diagonal Climatology': '#00C853',
    'LOCF':                 '#AA00FF',
}
METHODS = ['Linear', 'Spline', 'Diagonal Climatology', 'LOCF']
# PCT_ORDER is set dynamically after the main loop

# LOAD DATA
df_clean = xgboost_model.load_and_clean('Proj data.csv')

# ACF/PACF + CROSS-CORRELATION
print("\n=== ACF/PACF + Cross-Correlation Lag Feature Justification ===")
xgboost_model.plot_acf_pacf_lag_selection(df_clean)

# HYPERPARAMETER TUNING — chronological 80/10/10
print("\n=== Hyperparameter Tuning (Optuna, Chronological 80/10/10) ===")
xgboost_model.tune_hyperparameters(df_clean, n_trials=25)

# BASE MODEL
print("\n=== Base Model (clean data, chronological, lag features) ===")
base_results = xgboost_model.train_and_evaluate(df_clean)
xgboost_model.plot_results(base_results,
    title="Base Model — Chronological 80/10/10 (Lag Features)",
    out_path=os.path.join(RES, "base_model.png"))
xgboost_model.plot_residuals(base_results, "Base Model", "base")
xgboost_model.plot_feature_importance(base_results, "Base Model", "base")
xgboost_model.plot_error_by_hour(base_results, "Base Model", "base")
xgboost_model.plot_boosting_curve(base_results, "Base Model", "base")

# FULL DATASET SPLIT PLOT
df_indexed       = df_clean.copy()
df_indexed.index = pd.to_datetime(df_indexed["TIMESTAMP"])
df_indexed       = df_indexed.drop(columns=["TIMESTAMP"])
xgboost_model.plot_full_dataset_split(base_results, df_indexed, tag='base')

# FORECAST CONFIDENCE INTERVALS
print("\n=== Forecast Confidence Intervals ===")
xgboost_model.forecast_confidence_intervals(
    df_clean, n_bootstrap=30, show_steps=48, tag='base')

# INDEXED DATA
df_train_only = df_indexed[df_indexed.index <  TEST_START]
df_test_only  = df_indexed[df_indexed.index >= TEST_START]

# STORAGE
model_results     = []
recovery_records  = []
analysis_datasets = []
all_gap_row_sets  = {}
all_method_dfs    = {}

SEEDS = [42, 7, 13, 21, 99]

# CONDITION THRESHOLDS
_TEMP_VERY_HIGH_THRESH = df_train_only['AirTC_18m'].quantile(0.90)
_TEMP_HIGH_THRESH      = df_train_only['AirTC_18m'].quantile(0.85)
_TEMP_LOW_THRESH       = df_train_only['AirTC_18m'].quantile(0.15)
_WIND_LOW_THRESH       = df_train_only['WS_ms_18m_Avg'].quantile(0.20)
_TEMP_STD              = df_train_only['AirTC_18m'].std()

print(f"\n  Condition thresholds:")
print(f"    Extreme high temp : AirTC >= {_TEMP_VERY_HIGH_THRESH:.1f} C")
print(f"    High temperature  : AirTC >= {_TEMP_HIGH_THRESH:.1f} C")
print(f"    Low temperature   : AirTC <= {_TEMP_LOW_THRESH:.1f} C")
print(f"    Stagnant wind     : WS <= {_WIND_LOW_THRESH:.2f} m/s")


# GAP INTRODUCTION
# Only AirTC_18m is set to NaN — other variables
# retain their true observed values (practical scenario:
# temperature sensor fails, other sensors still working)
def make_corrupted(count):
    temp        = df_train_only["AirTC_18m"].ffill()
    peak_idx,   _ = find_peaks(temp.values, distance=6)
    trough_idx, _ = find_peaks(-temp.values, distance=6)
    slope         = np.abs(np.gradient(temp.values))
    trans_idx     = np.where(slope > np.percentile(slope, 85))[0]

    pool = {}
    for i in peak_idx:   pool[i] = 'peak'
    for i in trough_idx: pool[i] = 'trough'
    for i in trans_idx:
        if i not in pool: pool[i] = 'transition'

    pool_arr = np.array(list(pool.keys()))
    total    = len(df_train_only)
    target   = int(0.05 * count * total)

    np.random.seed(42)
    np.random.shuffle(pool_arr)

    df_train_corr = df_train_only.copy()
    used_rows     = set()
    gap_row_sets  = []

    for center in pool_arr:
        if len(used_rows) >= target: break
        gap_type = np.random.choice(['short', 'medium', 'long'],
                                     p=[0.5, 0.35, 0.15])
        if gap_type == 'short':
            w = np.random.randint(1, 7)
        elif gap_type == 'medium':
            w = np.random.randint(7, 25)
        else:
            w = np.random.randint(25, 49)

        block = []
        for i in range(max(0, center - w // 2),
                       min(total, center + w // 2 + 1)):
            if i not in used_rows:
                used_rows.add(i); block.append(i)
            if len(used_rows) >= target: break
        if block:
            gap_row_sets.append(sorted(block))

    for i in pool_arr:
        if len(used_rows) >= target: break
        if i not in used_rows:
            used_rows.add(i); gap_row_sets.append([i])

    all_rows     = sorted(used_rows)[:target]
    all_rows_set = set(all_rows)
    gap_row_sets = [
        [r for r in block if r in all_rows_set]
        for block in gap_row_sets
        if any(r in all_rows_set for r in block)
    ]

    # Only temperature goes NaN — RH, WS, WindDir remain observed
    df_train_corr.iloc[
        all_rows,
        df_train_corr.columns.get_loc('AirTC_18m')
    ] = np.nan

    # Count NaN rows in AirTC only (not all columns)
    actual_pct = df_train_corr['AirTC_18m'].isna().sum() / total * 100
    return df_train_corr, actual_pct, gap_row_sets


# CATEGORISATION HELPERS
def classify_condition(row_indices):
    rows = df_train_only.iloc[row_indices]
    temp = rows['AirTC_18m'].mean()
    ws   = rows['WS_ms_18m_Avg'].mean()
    if temp >= _TEMP_VERY_HIGH_THRESH: return 'extreme_high_temp'
    if temp >= _TEMP_HIGH_THRESH:      return 'high_temp'
    if temp <= _TEMP_LOW_THRESH:       return 'low_temp'
    if ws   <= _WIND_LOW_THRESH:       return 'stagnant_wind'
    return 'normal'


def classify_boundary(row_indices):
    total     = len(df_train_only)
    first_row = min(row_indices); last_row = max(row_indices)
    pre_val  = df_train_only.iloc[first_row - 1]['AirTC_18m'] \
               if first_row > 0        else np.nan
    post_val = df_train_only.iloc[last_row  + 1]['AirTC_18m'] \
               if last_row < total - 1 else np.nan
    if np.isnan(pre_val) or np.isnan(post_val): return 'stable'
    pre_ext  = (pre_val  >= _TEMP_HIGH_THRESH) or (pre_val  <= _TEMP_LOW_THRESH)
    post_ext = (post_val >= _TEMP_HIGH_THRESH) or (post_val <= _TEMP_LOW_THRESH)
    if pre_ext != post_ext: return 'crossing'
    if abs(post_val - pre_val) < 0.5 * _TEMP_STD: return 'stable'
    return 'rising' if post_val > pre_val else 'falling'


def classify_length(row_indices):
    n = len(row_indices)
    if n <= 6:  return 'short'    # up to 3 hours
    if n <= 24: return 'medium'   # 3 to 12 hours
    return 'long'                  # over 12 hours


# RECOVERY EVALUATION
def evaluate_recovery(df_interpolated, gap_row_sets, method_name, pct_label):
    records = []
    for block in gap_row_sets:
        if not block: continue
        actual = df_train_only.iloc[block]['AirTC_18m'].values
        filled = df_interpolated.iloc[block]['AirTC_18m'].values
        valid  = ~(np.isnan(actual) | np.isnan(filled))
        if valid.sum() == 0: continue
        a, f = actual[valid], filled[valid]
        ss_res = np.sum((a - f) ** 2)
        ss_tot = np.sum((a - np.mean(a)) ** 2)
        r2_val = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
        records.append({
            'Method':    method_name,
            'Missing %': pct_label,
            'Condition': classify_condition(block),
            'Boundary':  classify_boundary(block),
            'Length':    classify_length(block),
            'Gap_size':  len(block),
            'MAE':       round(mean_absolute_error(a, f), 3),
            'RMSE':      round(float(np.sqrt(mean_squared_error(a, f))), 3),
            'R2':        round(r2_val, 4) if not np.isnan(r2_val) else np.nan,
            'Bias':      round(float(np.mean(f - a)), 3),
        })
    return records


# RUN METHOD
def run_method(method_name, df_train_interp, pct_label, pct_tag):
    df_full = pd.concat([df_train_interp, df_test_only])
    df_full = df_full.reset_index().rename(columns={"index": "TIMESTAMP"})
    tag     = f"{method_name.lower().replace(' ', '_')}_{pct_tag}"

    seed_maes, seed_rmses, seed_r2s, seed_biases = [], [], [], []
    last_r = None
    for seed in SEEDS:
        r = xgboost_model.train_and_evaluate(df_full, block_seed=seed)
        seed_maes.append(r["mae"]); seed_rmses.append(r["rmse"])
        seed_r2s.append(r["r2"]);   seed_biases.append(r["bias"])
        last_r = r

    mean_mae  = float(np.mean(seed_maes));  std_mae  = float(np.std(seed_maes))
    mean_rmse = float(np.mean(seed_rmses)); std_rmse = float(np.std(seed_rmses))
    mean_r2   = float(np.mean(seed_r2s));   std_r2   = float(np.std(seed_r2s))
    mean_bias = float(np.mean(seed_biases));std_bias = float(np.std(seed_biases))

    last_r.update({
        "mae": mean_mae, "rmse": mean_rmse, "r2": mean_r2, "bias": mean_bias,
        "mae_std": std_mae, "rmse_std": std_rmse,
        "r2_std": std_r2, "bias_std": std_bias,
    })

    xgboost_model.plot_results(last_r,
        title=f"{method_name} — {pct_label} missing ({len(SEEDS)} seeds)",
        out_path=os.path.join(RES, f"{tag}_results.png"))
    xgboost_model.plot_residuals(last_r, f"{method_name} {pct_label}", tag)
    xgboost_model.plot_boosting_curve(last_r, f"{method_name} {pct_label}", tag)

    model_results.append({
        "Missing %": pct_label, "Method": method_name,
        "MAE":      round(mean_mae,  3), "MAE_std":  round(std_mae,  3),
        "RMSE":     round(mean_rmse, 3), "RMSE_std": round(std_rmse, 3),
        "R2":       round(mean_r2,   4), "R2_std":   round(std_r2,   4),
        "Bias":     round(mean_bias, 3), "Bias_std": round(std_bias, 3),
    })
    print(f"    {method_name}: RMSE={mean_rmse:.3f}+/-{std_rmse:.3f} "
          f"over {len(SEEDS)} seeds")


# GAP LENGTH ANALYSIS
def plot_gap_length_analysis(all_gap_row_sets_local):
    colors_pct = ['#1565C0', '#E53935', '#2E7D32', '#F57C00']
    pct_labels = list(all_gap_row_sets_local.keys())
    all_lengths = {p: [len(b) for b in all_gap_row_sets_local[p] if len(b) > 0]
                   for p in pct_labels}

    # Histogram per pct level
    fig, axes = plt.subplots(2, 2, figsize=(20, 14), constrained_layout=True)
    axes = axes.flatten()
    fig.suptitle('Gap Length Distribution by Missing %',
                 fontsize=18, fontweight='bold')
    for ax, (pct, lengths) in zip(axes, all_lengths.items()):
        if not lengths: continue
        arr  = np.array(lengths)
        bins = np.arange(1, min(arr.max() + 2, 55))
        idx  = pct_labels.index(pct)
        ax.hist(arr, bins=bins, color=colors_pct[idx], alpha=0.75,
                edgecolor='white', lw=1.5)
        ax.axvline(np.mean(arr), color='black', lw=2.0, ls='--',
                   label=f'Mean={np.mean(arr):.1f}')
        ax.axvline(np.median(arr), color='red', lw=2.0, ls='-.',
                   label=f'Median={np.median(arr):.1f}')
        ax.set_xlabel('Gap Length (steps, 1 step = 30 min)', fontsize=19)
        ax.set_ylabel('Count', fontsize=19)
        ax.set_title(f'{pct} missing — {len(lengths)} gaps  |  '
                     f'Short={sum(1 for l in lengths if l<=6)}  '
                     f'Medium={sum(1 for l in lengths if 6<l<=24)}  '
                     f'Long={sum(1 for l in lengths if l>24)}',
                     fontsize=15, fontweight='bold')
        ax.legend(fontsize=18)
        ax.grid(True, alpha=0.25)
    plt.tight_layout(pad=1.5)
    plt.savefig(os.path.join(GAP, 'gap_length_histogram.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: gap_length_histogram.png")

    # Probability distribution (empirical PMF)
    fig, ax = plt.subplots(figsize=(16, 8))
    for i, (pct, lengths) in enumerate(all_lengths.items()):
        if not lengths: continue
        arr          = np.array(lengths, dtype=float)
        unique, cnts = np.unique(arr, return_counts=True)
        ax.plot(unique, cnts / cnts.sum(), color=colors_pct[i], lw=2.0,
                marker='o', ms=4, alpha=0.8, label=f'{pct} missing')
    ax.set_xlabel('Gap Length (steps)', fontsize=17, fontweight='bold')
    ax.set_ylabel('Probability', fontsize=17, fontweight='bold', color='black')
    ax.set_title('Gap Length — Empirical PMF',
                 fontsize=18, fontweight='bold', color='black')
    ax.legend(fontsize=19)
    ax.grid(True, alpha=0.25)
    for spine in ax.spines.values():
        spine.set_edgecolor('black'); spine.set_linewidth(2.0)
    plt.tight_layout(pad=1.5)
    plt.savefig(os.path.join(GAP, 'gap_length_probability.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: gap_length_probability.png")

    # CDF
    fig, ax = plt.subplots(figsize=(16, 8))
    for i, (pct, lengths) in enumerate(all_lengths.items()):
        if not lengths: continue
        arr = np.sort(np.array(lengths, dtype=float))
        ax.plot(arr, np.arange(1, len(arr)+1) / len(arr),
                color=colors_pct[i], lw=2.5, label=f'{pct} missing')
    ax.axvline(6,  color='grey', lw=1.5, ls='--', alpha=0.7,
               label='Short/Medium boundary (6 steps = 3h)')
    ax.axvline(24, color='grey', lw=1.5, ls=':',  alpha=0.7,
               label='Medium/Long boundary (24 steps = 12h)')
    ax.set_xlabel('Gap Length (steps)', fontsize=17, fontweight='bold')
    ax.set_ylabel('Cumulative Probability', fontsize=17, fontweight='bold', color='black')
    ax.set_title('Gap Length CDF',
                 fontsize=17, fontweight='bold', color='black')
    ax.legend(fontsize=15); ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)
    for spine in ax.spines.values():
        spine.set_edgecolor('black'); spine.set_linewidth(2.0)
    plt.tight_layout(pad=1.5)
    plt.savefig(os.path.join(GAP, 'gap_length_cdf.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: gap_length_cdf.png")

    # Stacked bar
    shorts  = [sum(1 for l in all_lengths[p] if l<=6)    for p in pct_labels]
    mediums = [sum(1 for l in all_lengths[p] if 6<l<=24) for p in pct_labels]
    longs   = [sum(1 for l in all_lengths[p] if l>24)    for p in pct_labels]
    x = np.arange(len(pct_labels))
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.bar(x, shorts,  color='#2E7D32', alpha=0.85,
           label='Short (<=6 steps, <=3h)')
    ax.bar(x, mediums, bottom=shorts,
           color='#F57C00', alpha=0.85, label='Medium (7-24 steps, 3-12h)')
    ax.bar(x, longs,
           bottom=[s + m for s, m in zip(shorts, mediums)],
           color='#C62828', alpha=0.85, label='Long (>24 steps, >12h)')
    ax.set_xticks(x); ax.set_xticklabels(pct_labels, fontsize=19)
    ax.set_xlabel('Missing Percentage Level', fontsize=17, fontweight='bold', color='black')
    ax.set_ylabel('Number of Gaps', fontsize=17, fontweight='bold', color='black')
    ax.set_title('Gap Type Distribution per Missing %',
                 fontsize=17, fontweight='bold', color='black')
    ax.legend(fontsize=19); ax.grid(True, axis='y', alpha=0.25)
    for spine in ax.spines.values():
        spine.set_edgecolor('black'); spine.set_linewidth(2.0)
    for i, (s, m, l) in enumerate(zip(shorts, mediums, longs)):
        if s > 0: ax.text(i, s/2,       str(s), ha='center', va='center',
                          fontsize=15, fontweight='bold', color='white')
        if m > 0: ax.text(i, s + m/2,   str(m), ha='center', va='center',
                          fontsize=15, fontweight='bold', color='white')
        if l > 0: ax.text(i, s + m + l/2, str(l), ha='center', va='center',
                          fontsize=15, fontweight='bold', color='white')
    plt.tight_layout(pad=1.5)
    plt.savefig(os.path.join(GAP, 'gap_length_stacked_bar.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: gap_length_stacked_bar.png")

    # Boxplot
    fig, ax = plt.subplots(figsize=(14, 7))
    bp = ax.boxplot([np.array(all_lengths[p]) for p in pct_labels],
                    patch_artist=True, notch=False,
                    medianprops=dict(color='black', lw=2.5))
    for patch, color in zip(bp['boxes'], colors_pct):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax.set_xticklabels(pct_labels, fontsize=19)
    ax.set_xlabel('Missing Percentage Level', fontsize=17, fontweight='bold', color='black')
    ax.set_ylabel('Gap Length (steps)', fontsize=17, fontweight='bold')
    ax.set_title('Gap Length Boxplot per Missing %',
                 fontsize=17, fontweight='bold', color='black')
    ax.grid(True, axis='y', alpha=0.25)
    for spine in ax.spines.values():
        spine.set_edgecolor('black'); spine.set_linewidth(2.0)
    plt.tight_layout(pad=1.5)
    plt.savefig(os.path.join(GAP, 'gap_length_boxplot.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: gap_length_boxplot.png")


# MAIN LOOP
for count in range(1, 5):
    df_train_corr, actual_pct, gap_row_sets = make_corrupted(count)
    pct_label = f"{round(actual_pct)}%"
    pct_tag   = pct_label.replace('%', 'pct')
    all_gap_row_sets[pct_label] = gap_row_sets
    print(f"\n=== {pct_label} missing ({len(gap_row_sets)} gap blocks) ===")

    lengths = [len(b) for b in gap_row_sets]
    if lengths:
        print(f"  Lengths: min={min(lengths)} max={max(lengths)} "
              f"mean={np.mean(lengths):.1f}  "
              f"short={sum(1 for l in lengths if l<=6)}  "
              f"medium={sum(1 for l in lengths if 6<l<=24)}  "
              f"long={sum(1 for l in lengths if l>24)}")

    # Gap visualisation
    fi = df_train_only.index; tot = len(fi); cs = tot // 4
    fig, axes = plt.subplots(4, 1, figsize=(22, 18), constrained_layout=True)
    for i, ax in enumerate(axes):
        s = i * cs; e = (i + 1) * cs if i < 3 else tot
        st, en = fi[s], fi[e - 1]
        oc = df_train_only.loc[st:en, 'AirTC_18m']
        cc = df_train_corr.loc[st:en, 'AirTC_18m']
        ax.plot(oc.index, oc.values, color='steelblue', lw=1.5,
                alpha=0.4, ls='--', label='Original')
        ax.plot(cc.index, cc.values, color='steelblue', lw=1.5,
                label='With gaps')
        for ts in cc[cc.isna()].index:
            ax.axvspan(ts - pd.Timedelta(minutes=15),
                       ts + pd.Timedelta(minutes=15),
                       color='red', alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        ax.tick_params(axis='x', rotation=30, labelsize=11)
        ax.set_ylabel('AirTC_18m (C)', fontsize=18)
        ax.set_title(f'Part {i+1}: {st.strftime("%d %b")} to '
                     f'{en.strftime("%d %b %Y")}', fontsize=15)
    legend_elements = [
        plt.Line2D([0], [0], color='steelblue', lw=1.5, alpha=0.4,
                   ls='--', label='Original'),
        plt.Line2D([0], [0], color='steelblue', lw=1.5, label='With gaps'),
        Patch(facecolor='red', alpha=0.3, label='Missing region'),
    ]
    axes[0].legend(handles=legend_elements, fontsize=18, loc='upper right')
    fig.suptitle(
        f'Training Temperature with Gaps ({actual_pct:.2f}% missing)',
        fontsize=19)
    plt.tight_layout(pad=1.5)
    plt.savefig(os.path.join(GAP, f'missing_gaps_{pct_tag}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # Interpolate — only AirTC_18m
    print("  Linear...")
    df_lin = df_train_corr.copy()
    df_lin['AirTC_18m'] = im.linear_inter(df_lin, 'AirTC_18m')['AirTC_18m']

    print("  Spline...")
    df_spl = df_train_corr.copy()
    df_spl['AirTC_18m'] = im.spline_inter(df_spl, 'AirTC_18m')['AirTC_18m']

    print("  Diagonal Climatology...")
    df_dc = df_train_corr.copy()
    temp_series = df_train_corr[['AirTC_18m']].copy()
    temp_series.index = df_train_corr.index
    df_dc['AirTC_18m'] = im.diag_climatology_practical(
        temp_series)['AirTC_18m']

    print("  LOCF...")
    df_locf = df_train_corr.copy()
    df_locf['AirTC_18m'] = im.locf_interpolation(
        df_locf, 'AirTC_18m')['AirTC_18m']

    all_method_dfs[('Linear',               pct_label)] = df_lin
    all_method_dfs[('Spline',               pct_label)] = df_spl
    all_method_dfs[('Diagonal Climatology', pct_label)] = df_dc
    all_method_dfs[('LOCF',                 pct_label)] = df_locf

    METHOD_DFS = {'Linear': df_lin, 'Spline': df_spl,
                  'Diagonal Climatology': df_dc, 'LOCF': df_locf}

    # Overlay plot
    METHOD_INTERPS = [
        ('Original',             df_train_only, '#1565C0', '-',  1.2, 0.9),
        ('With gaps',            df_train_corr, '#888888', '-',  0.6, 0.6),
        ('Linear',               df_lin,        METHOD_COLORS['Linear'],               '--', 1.0, 0.85),
        ('Spline',               df_spl,        METHOD_COLORS['Spline'],               '--', 1.0, 0.85),
        ('Diagonal Climatology', df_dc,         METHOD_COLORS['Diagonal Climatology'], '--', 1.0, 0.85),
        ('LOCF',                 df_locf,       METHOD_COLORS['LOCF'],                 '--', 1.0, 0.85),
    ]
    fi2 = df_train_only.index; tot2 = len(fi2); cs2 = tot2 // 4
    fig, axes = plt.subplots(4, 1, figsize=(26, 24))
    fig.suptitle(
        f'Interpolation Comparison — AirTC_18m ({actual_pct:.1f}% missing)',
        fontsize=20, fontweight='bold')
    for i, ax in enumerate(axes):
        s = i * cs2; e = (i + 1) * cs2 if i < 3 else tot2
        st, en = fi2[s], fi2[e - 1]
        gap_col = df_train_corr.loc[st:en, 'AirTC_18m']
        for ts in gap_col[gap_col.isna()].index:
            ax.axvspan(ts - pd.Timedelta(minutes=15),
                       ts + pd.Timedelta(minutes=15),
                       color='red', alpha=0.15, zorder=0)
        for label, df_src, clr, ls, lw, alpha in METHOD_INTERPS:
            series = df_src.loc[st:en, 'AirTC_18m']
            ax.plot(series.index, series.values, color=clr, ls=ls, lw=lw,
                    alpha=alpha,
                    label=label if i == 0 else '_nolegend_')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        ax.tick_params(axis='x', rotation=30, labelsize=14, colors='black', width=2)
        ax.tick_params(axis='y', labelsize=14, colors='black', width=2)
        ax.set_ylabel('AirTC_18m (°C)', fontsize=16, fontweight='bold', color='black')
        ax.set_title(f'Part {i+1}: {st.strftime("%d %b")} to '
                     f'{en.strftime("%d %b %Y")}',
                     fontsize=17, fontweight='bold', color='black')
        ax.grid(True, alpha=0.2)
        for spine in ax.spines.values():
            spine.set_edgecolor('black'); spine.set_linewidth(1.8)
    legend_elements = [
        plt.Line2D([0], [0], color='#1565C0', lw=1.5, ls='-',
                   label='Original'),
        plt.Line2D([0], [0], color='#888888', lw=1.5, ls='-',
                   label='With gaps'),
        plt.Line2D([0], [0], color=METHOD_COLORS['Linear'],
                   lw=1.5, ls='--', label='Linear'),
        plt.Line2D([0], [0], color=METHOD_COLORS['Spline'],
                   lw=1.5, ls='--', label='Spline'),
        plt.Line2D([0], [0], color=METHOD_COLORS['Diagonal Climatology'],
                   lw=1.5, ls='--', label='Diagonal Climatology'),
        plt.Line2D([0], [0], color=METHOD_COLORS['LOCF'],
                   lw=1.5, ls='--', label='LOCF'),
        Patch(facecolor='red', alpha=0.15, label='Missing region'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=len(legend_elements), fontsize=14,
               framealpha=1.0, edgecolor='black',
               prop={'weight': 'bold', 'size': 14},
               bbox_to_anchor=(0.5, 0.0))
    plt.tight_layout(rect=[0, 0.05, 1, 0.97], pad=1.5)
    plt.savefig(os.path.join(GAP, f'interpolation_overlay_{pct_tag}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Overlay plot saved for {pct_label}")

    # Gap-indexed plot
    gap_nums = []; actual_means = []
    method_means = {m: [] for m in METHOD_DFS}
    for gnum, block in enumerate(gap_row_sets, start=1):
        av = df_train_only.iloc[block]['AirTC_18m'].values
        if len(av) == 0: continue
        gap_nums.append(gnum)
        actual_means.append(float(np.mean(av)))
        for mname, mdf in METHOD_DFS.items():
            method_means[mname].append(
                float(np.nanmean(mdf.iloc[block]['AirTC_18m'].values)))

    fig, ax = plt.subplots(figsize=(max(18, len(gap_nums) * 0.12 + 4), 8))
    ax.plot(gap_nums, actual_means, color='black', lw=2.0, ls='-',
            marker='o', markersize=3, label='Actual', zorder=5)
    ls_cycle = ['--', '-.', ':', '--']
    for (mname, clr), ls in zip(METHOD_COLORS.items(), ls_cycle):
        ax.plot(gap_nums, method_means[mname], color=clr, lw=2.0, ls=ls,
                marker='s', markersize=2.5, alpha=0.9, label=mname)
    ax.set_xlabel('Gap Number', fontsize=18, fontweight='bold', color='black')
    ax.set_ylabel('Mean AirTC (C)', fontsize=18, fontweight='bold')
    ax.set_title(
        f'Interpolation Comparison at Each Gap — {pct_label} Missing',
        fontsize=19, fontweight='bold', color='black')
    tick_step = max(1, len(gap_nums) // 30)
    ax.set_xticks(gap_nums[::tick_step])
    ax.set_xticklabels([str(g) for g in gap_nums[::tick_step]],
                        fontsize=19, fontweight='bold', rotation=45)
    ax.grid(True, alpha=0.3, linestyle='--')
    for spine in ax.spines.values():
        spine.set_edgecolor('black'); spine.set_linewidth(2.0)
    ax.legend(fontsize=19, framealpha=1.0, edgecolor='black',
              loc='upper right', ncol=2,
              prop={'weight': 'bold', 'size': 12})
    plt.tight_layout(pad=1.5)
    plt.savefig(os.path.join(GAP, f'gap_labelled_{pct_tag}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # Error histogram
    all_errors = {}
    for mname, mdf in METHOD_DFS.items():
        errors = []
        for block in gap_row_sets:
            av = df_train_only.iloc[block]['AirTC_18m'].values
            iv = mdf.iloc[block]['AirTC_18m'].values
            valid = ~(np.isnan(av) | np.isnan(iv))
            if valid.sum() > 0:
                errors.extend((iv[valid] - av[valid]).tolist())
        all_errors[mname] = errors

    all_vals = [e for errs in all_errors.values() for e in errs]
    bin_min  = np.percentile(all_vals, 1)
    bin_max  = np.percentile(all_vals, 99)
    bins     = np.linspace(bin_min, bin_max, 60)

    fig, ax = plt.subplots(figsize=(16, 8))
    for mname, errors in all_errors.items():
        errors_arr = np.array(errors); clr = METHOD_COLORS[mname]
        kde     = gaussian_kde(errors_arr, bw_method=0.3)
        x_range = np.linspace(bin_min, bin_max, 300)
        ax.plot(x_range, kde(x_range), color=clr, lw=2.5,
                label=f'{mname} (n={len(errors)})')
        ax.fill_between(x_range, kde(x_range), alpha=0.08, color=clr)
        ax.axvline(np.mean(errors_arr), color=clr, lw=1.8, ls='--',
                   alpha=0.9)
    ax.axvline(0, color='black', lw=2.0, ls='-', label='Zero error')
    ax.set_xlabel('Interpolation Error (Filled - Actual) (C)',
                  fontsize=18, fontweight='bold')
    ax.set_ylabel('Density', fontsize=18, fontweight='bold', color='black')
    ax.set_title(
        f'Error Distribution at Gap Locations — {pct_label} Missing',
        fontsize=19, fontweight='bold', color='black')
    ax.legend(fontsize=19, framealpha=1.0, edgecolor='black',
              prop={'weight': 'bold', 'size': 12})
    ax.grid(True, alpha=0.25, linestyle='--')
    for spine in ax.spines.values():
        spine.set_edgecolor('black'); spine.set_linewidth(2.0)
    plt.tight_layout(pad=1.5)
    plt.savefig(os.path.join(GAP, f'error_histogram_{pct_tag}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # Variability + recovery evaluation
    for method_name, df_interp in METHOD_DFS.items():
        analysis_datasets.append({
            'Method':        method_name,
            'Missing %':     pct_label,
            'Temp_STD':      df_interp['AirTC_18m'].std(),
            'Temp_Diff_STD': df_interp['AirTC_18m'].diff().std(),
        })
        recs = evaluate_recovery(df_interp, gap_row_sets,
                                  method_name, pct_label)
        recovery_records.extend(recs)
        print(f"    Recovery eval {method_name}: {len(recs)} gap blocks")

    # Model evaluation
    run_method('Linear',               df_lin,  pct_label, pct_tag)
    run_method('Spline',               df_spl,  pct_label, pct_tag)
    run_method('Diagonal Climatology', df_dc,   pct_label, pct_tag)
    run_method('LOCF',                 df_locf, pct_label, pct_tag)


# GAP LENGTH ANALYSIS
print("\n=== Gap Length Analysis ===")
plot_gap_length_analysis(all_gap_row_sets)


# OVERALL MODEL METRICS
results_df = pd.DataFrame(model_results)

# deduplicate in case any pct labels collided
results_df = results_df.drop_duplicates(
    subset=['Method', 'Missing %'], keep='last')

# derive pct_order dynamically from actual data — never hardcode
pct_order = sorted(
    results_df['Missing %'].unique(),
    key=lambda x: float(x.replace('%', ''))
)
PCT_ORDER = pct_order
print("\nActual pct_order:", pct_order)
print(results_df[['Method', 'Missing %']].to_string())

pivot_df = results_df.pivot_table(
    index='Method', columns='Missing %',
    values=['MAE', 'RMSE', 'R2', 'Bias'])
pivot_df = pivot_df.swaplevel(axis=1).sort_index(axis=1)
pivot_df = pivot_df.reindex(pct_order, axis=1, level=0)  # dynamic
pivot_df.to_csv(os.path.join(COMP, 'interpolation_metrics.csv'))
results_df.to_csv(os.path.join(COMP, 'interpolation_metrics_with_std.csv'),
                  index=False)
print("\n=== Overall model metrics ===")
print(pivot_df.to_string())

COLORS  = ['steelblue', 'darkorange', 'green', 'red']
metrics = ['RMSE', 'MAE', 'R2', 'Bias']

# Original interpolation downstream metrics line chart (standalone)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()
handles_legend = None
for ax, metric in zip(axes, metrics):
    h_list = []
    for method, color in zip(METHODS, [METHOD_COLORS[m] for m in METHODS]):
        mdf  = results_df[results_df['Method'] == method]\
               .set_index('Missing %').reindex(pct_order)
        vals = mdf[metric].values
        line, = ax.plot(pct_order, vals, marker='o', lw=2.5, markersize=8,
                        color=color, label=method)
        h_list.append(line)
    if handles_legend is None:
        handles_legend = h_list
    ax.set_title(f'{metric} vs Missing % (mean +/- std, {len(SEEDS)} seeds)',
                 fontsize=16, fontweight='bold', color='black')
    ax.set_xlabel('Missing %', fontsize=15, fontweight='bold', color='black')
    ax.set_ylabel(metric, fontsize=15, fontweight='bold', color='black')
    ax.tick_params(labelsize=13, colors='black', labelcolor='black', width=2)
    ax.grid(True, ls='--', alpha=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor('black'); spine.set_linewidth(1.5)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color('black'); lbl.set_fontweight('bold')
fig.suptitle('Interpolation Comparison \u2014 Downstream Model Metrics',
             fontsize=19, fontweight='bold', color='black')
fig.legend(handles=handles_legend, labels=METHODS,
           loc='lower center', ncol=len(METHODS),
           fontsize=15, framealpha=1.0, edgecolor='black',
           prop={'weight': 'bold', 'size': 15},
           bbox_to_anchor=(0.5, 0.01))
plt.tight_layout(rect=[0, 0.07, 1, 0.96], pad=0.4, h_pad=0.5, w_pad=0.6)
plt.savefig(os.path.join(COMP, 'interpolation_downstream_metrics.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: interpolation_downstream_metrics.png")

# Grouped bar
fig, axes = plt.subplots(1, 2, figsize=(18, 7), constrained_layout=True)
x = np.arange(len(METHODS)); w = 0.2
bar_colors = ['#1565C0', '#E53935', '#2E7D32', '#F57C00']
for ax, metric in zip(axes, ['RMSE', 'MAE']):
    for j, pct in enumerate(pct_order):
        vals = []
        for m in METHODS:
            sub = results_df[(results_df['Method'] == m) &
                             (results_df['Missing %'] == pct)]
            vals.append(sub[metric].values[0] if len(sub) > 0 else 0.0)
        ax.bar(x + j * w, vals, w, label=pct,
               color=bar_colors[j % len(bar_colors)], alpha=0.85)
    ax.set_title(f'{metric} by Method and Missing %',
                 fontsize=13, fontweight='bold', color='black')
    ax.set_xticks(x + w * 1.5)
    ax.set_xticklabels(METHODS, rotation=15, ha='right',
                       fontsize=11, fontweight='bold')
    ax.set_ylabel(metric, fontsize=12, fontweight='bold', color='black')
    ax.tick_params(labelsize=11, colors='black')
    ax.legend(title='Missing %', fontsize=11, framealpha=1.0,
              edgecolor='black', prop={'weight': 'bold'})
    ax.grid(True, axis='y', alpha=0.3)
    for spine in ax.spines.values():
        spine.set_edgecolor('black'); spine.set_linewidth(1.5)
plt.tight_layout(pad=2.5)
plt.savefig(os.path.join(COMP, 'grouped_bar_rmse_mae.png'),
            dpi=150, bbox_inches='tight')
plt.close()

# R2 heatmap
r2_mat = np.zeros((len(METHODS), len(pct_order)))
for i, m in enumerate(METHODS):
    for j, p in enumerate(pct_order):
        sub = results_df[(results_df['Method'] == m) &
                         (results_df['Missing %'] == p)]
        r2_mat[i, j] = sub['R2'].values[0] if len(sub) > 0 else 0.0

fig, ax = plt.subplots(figsize=(12, 6))
fig.subplots_adjust(top=0.88, bottom=0.10, left=0.14, right=0.95)
im_h = ax.imshow(r2_mat, cmap='RdYlGn', aspect='auto',
                  vmin=r2_mat.min() - 0.01, vmax=1.0)
ax.set_xticks(range(len(pct_order)))
ax.set_xticklabels(pct_order, fontsize=16, fontweight='bold', color='black')
ax.set_yticks(range(len(METHODS)))
ax.set_yticklabels(METHODS, fontsize=16, fontweight='bold', color='black')
ax.tick_params(axis='both', which='both', length=0)
ax.set_title('R\u00b2 Heatmap — Methods vs Missing % (Chronological)',
             fontsize=18, fontweight='bold', color='black', pad=10)
cb = plt.colorbar(im_h, ax=ax)
cb.set_label('R\u00b2', fontsize=15, fontweight='bold', color='black')
cb.ax.tick_params(labelsize=14, colors='black')
for i in range(len(METHODS)):
    for j in range(len(pct_order)):
        ax.text(j, i, f"{r2_mat[i,j]:.4f}",
                ha='center', va='center', fontsize=15,
                fontweight='bold', color='black')
plt.savefig(os.path.join(COMP, 'r2_heatmap.png'),
            dpi=150, bbox_inches='tight')
plt.close()


# RECOVERY ANALYSIS
rec_df = pd.DataFrame(recovery_records)

# ── COMBINED FIGURE: Dual y-axis per subplot ──
# Left y-axis  = Interpolation (dotted lines)  — independent scale
# Right y-axis = Recovery      (solid lines)   — independent scale
ylabels_map = {
    'RMSE': 'RMSE (°C)',
    'MAE':  'MAE (°C)',
    'R2':   'R²',
    'Bias': 'Bias (°C)',
}
metrics_order = ['RMSE', 'MAE', 'R2', 'Bias']

fig, axes_left = plt.subplots(2, 2, figsize=(38, 30))
axes_left = axes_left.flatten()

# Build legend handles once
legend_handles = []
legend_labels  = []
_leg_done = False

for ax_l, metric in zip(axes_left, metrics_order):
    ax_r = ax_l.twinx()   # right y-axis for recovery

    for method in METHODS:
        color = METHOD_COLORS[method]

        # Interpolation (dotted) on LEFT axis
        mdf_interp = results_df[results_df['Method'] == method] \
                         .set_index('Missing %').reindex(pct_order)
        vals_interp = mdf_interp[metric].values
        li, = ax_l.plot(pct_order, vals_interp,
                        marker='o', lw=3.5, markersize=13,
                        color=color, ls=':', alpha=0.85,
                        label=f'{method} (Interpolation)')

        # Recovery (solid) on RIGHT axis
        vals_rec = []
        for pct in pct_order:
            sub = rec_df[(rec_df['Method'] == method) &
                         (rec_df['Missing %'] == pct)][metric]
            vals_rec.append(sub.mean() if len(sub) > 0 else float('nan'))
        lr, = ax_r.plot(pct_order, vals_rec,
                        marker='o', lw=5.5, markersize=13,
                        color=color, ls='-',
                        label=f'{method} (Recovery)')

        if not _leg_done:
            legend_handles += [li, lr]
            legend_labels  += [f'{method} (Interpolation)',
                                f'{method} (Recovery)']

    _leg_done = True

    # Left axis styling (Interpolation)
    ax_l.set_title(f'{metric} vs Missing %',
                   fontsize=52, fontweight='bold', color='black')
    ax_l.set_xlabel('Missing %', fontsize=44, fontweight='bold', color='black')
    ax_l.set_ylabel(f'Interpolation  {ylabels_map[metric]}',
                    fontsize=38, fontweight='bold', color='#444444')
    ax_l.tick_params(axis='both', labelsize=38, colors='black',
                     labelcolor='black', width=3.0)
    ax_l.grid(True, ls='--', alpha=0.35)
    for lbl in ax_l.get_xticklabels() + ax_l.get_yticklabels():
        lbl.set_color('black'); lbl.set_fontweight('bold')
    for spine in ax_l.spines.values():
        spine.set_edgecolor('black'); spine.set_linewidth(2.5)

    # Right axis styling (Recovery)
    ax_r.set_ylabel(f'Recovery  {ylabels_map[metric]}',
                    fontsize=38, fontweight='bold', color='#444444')
    ax_r.tick_params(axis='y', labelsize=38, colors='black',
                     labelcolor='black', width=3.0)
    for lbl in ax_r.get_yticklabels():
        lbl.set_color('black'); lbl.set_fontweight('bold')
    # Right spine visible
    ax_r.spines['right'].set_edgecolor('black')
    ax_r.spines['right'].set_linewidth(2.5)

fig.legend(handles=legend_handles, labels=legend_labels,
           loc='lower center', ncol=4,
           fontsize=32, framealpha=1.0, edgecolor='black',
           prop={'weight': 'bold', 'size': 32},
           bbox_to_anchor=(0.5, 0.0))
plt.tight_layout(rect=[0, 0.07, 1, 1.0], pad=0.5, h_pad=1.2, w_pad=1.0)
plt.savefig(os.path.join(COMP, 'interpolation_comparison.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: interpolation_comparison.png (dual y-axis: interpolation left, recovery right)")



# GAP TEMPERATURE ANOMALY ANALYSIS
def gap_temp_analysis():
    overall_mean = df_train_only['AirTC_18m'].mean()
    overall_std  = df_train_only['AirTC_18m'].std()
    print(f"\n=== Gap Temperature Anomaly Analysis ===")
    print(f"  Overall mean={overall_mean:.2f} C  std={overall_std:.2f} C")
    rows = []
    for pct_label, gap_row_sets in all_gap_row_sets.items():
        all_gap_rows  = [r for block in gap_row_sets for r in block]
        gap_temps     = df_train_only.iloc[all_gap_rows]['AirTC_18m']
        non_gap_mask  = ~df_train_only.index.isin(
            df_train_only.index[all_gap_rows])
        non_gap_temps = df_train_only[non_gap_mask]['AirTC_18m']
        gap_mean      = gap_temps.mean()
        non_gap_mean  = non_gap_temps.mean()
        z_score       = (gap_mean - overall_mean) / overall_std
        frac_ext      = ((gap_temps >= _TEMP_VERY_HIGH_THRESH) |
                         (gap_temps <= _TEMP_LOW_THRESH)).mean()
        frac_ext_all  = ((df_train_only['AirTC_18m'] >= _TEMP_VERY_HIGH_THRESH) |
                         (df_train_only['AirTC_18m'] <= _TEMP_LOW_THRESH)).mean()
        overrep       = frac_ext / frac_ext_all
        print(f"  {pct_label}: gap_mean={gap_mean:.2f}  "
              f"non_gap={non_gap_mean:.2f}  "
              f"diff={gap_mean-non_gap_mean:.2f}  "
              f"Z={z_score:.3f}  overrep={overrep:.2f}x")
        rows.append({
            'Missing_%':                 pct_label,
            'Overall_Mean':              round(overall_mean, 3),
            'Gap_Mean':                  round(gap_mean, 3),
            'NonGap_Mean':               round(non_gap_mean, 3),
            'Mean_Difference':           round(gap_mean - non_gap_mean, 3),
            'Z_Score':                   round(z_score, 3),
            'Frac_Extreme_In_Gaps':      round(frac_ext, 4),
            'Frac_Extreme_Overall':      round(frac_ext_all, 4),
            'Overrepresentation_Factor': round(overrep, 3),
        })
    pd.DataFrame(rows).to_csv(
        os.path.join(COND_DIR, 'gap_temperature_anomaly_analysis.csv'),
        index=False)
    pd.DataFrame(rows).to_excel(
        os.path.join(COND_DIR, 'gap_temperature_anomaly_analysis.xlsx'),
        index=False)
    print("  Saved: gap_temperature_anomaly_analysis.csv/.xlsx")

gap_temp_analysis()


# RECOVERY METRICS LINE GRAPHS
def plot_recovery_line_graphs():
    if rec_df.empty: return
    metrics_rec = ['RMSE', 'MAE', 'R2', 'Bias']
    ylabels_rec = {
        'RMSE': 'RMSE (°C)',
        'MAE':  'MAE (°C)',
        'R2':   'R² (unitless)',
        'Bias': 'Bias (°C)',
    }

    # Overlaid: all methods per metric — 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(20, 13))
    axes = axes.flatten()
    fig.suptitle('Recovery Metrics vs Missing %',
                 fontsize=24, fontweight='bold', color='black')
    handles_rl = None
    for ax, metric in zip(axes, metrics_rec):
        h_list = []
        for method in METHODS:
            vals = []
            for pct in pct_order:
                sub = rec_df[(rec_df['Method'] == method) &
                             (rec_df['Missing %'] == pct)][metric]
                vals.append(sub.mean() if len(sub) > 0 else np.nan)
            line, = ax.plot(pct_order, vals, marker='o', lw=3.0, ms=9,
                    color=METHOD_COLORS[method], label=method)
            h_list.append(line)
        if handles_rl is None:
            handles_rl = h_list
        ax.set_xlabel('Missing %', fontsize=18, fontweight='bold', color='black')
        ax.set_ylabel(ylabels_rec[metric], fontsize=18, fontweight='bold',
                      color='black')
        ax.set_title(f'Recovery {metric} vs Missing %', fontweight='bold',
                     color='black', fontsize=18)
        ax.tick_params(axis='both', labelsize=15, colors='black',
                       labelcolor='black', width=2)
        if metric == 'R2':
            ax.set_ylim(top=1.0)
        ax.grid(True, alpha=0.25)
        for spine in ax.spines.values():
            spine.set_edgecolor('black'); spine.set_linewidth(2.0)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_color('black'); lbl.set_fontweight('bold')
    fig.legend(handles=handles_rl, labels=METHODS,
               loc='lower center', ncol=len(METHODS),
               fontsize=16, framealpha=1.0, edgecolor='black',
               prop={'weight': 'bold', 'size': 16},
               bbox_to_anchor=(0.5, 0.01))
    plt.tight_layout(rect=[0, 0.07, 1, 0.96], pad=0.3, h_pad=0.4, w_pad=0.5)
    plt.savefig(os.path.join(COND_DIR, 'recovery_metrics_line_overlaid.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: recovery_metrics_line_overlaid.png")

    # Per method: all metrics on same plot
    fig, axes = plt.subplots(2, 2, figsize=(22, 16))
    axes = axes.flatten()
    fig.suptitle('Recovery Metrics per Method',
                 fontsize=24, fontweight='bold', color='black')
    metric_colors_local = {
        'RMSE': '#E53935', 'MAE': '#1565C0', 'Bias': '#2E7D32'}
    handles_pm = None
    for ax_idx, (ax, method) in enumerate(zip(axes, METHODS)):
        h_list = []
        for metric, mclr in metric_colors_local.items():
            vals = []
            for pct in pct_order:
                sub = rec_df[(rec_df['Method'] == method) &
                             (rec_df['Missing %'] == pct)][metric]
                vals.append(sub.mean() if len(sub) > 0 else np.nan)
            line, = ax.plot(pct_order, vals, marker='o', lw=2.5, ms=8,
                    color=mclr, label=metric)
            h_list.append(line)
        if handles_pm is None:
            handles_pm = h_list
        ax.axhline(0, color='black', lw=1.0, ls='--', alpha=0.5)
        ax.set_xlabel('Missing %', fontsize=19, fontweight='bold', color='black')
        ax.set_ylabel('Metric Value (°C)', fontsize=19, fontweight='bold',
                      color='black')
        ax.set_title(method, fontsize=21, fontweight='bold', color='black')
        ax.tick_params(axis='both', labelsize=15, colors='black',
                       labelcolor='black', width=2)
        ax.grid(True, alpha=0.25)
        for spine in ax.spines.values():
            spine.set_edgecolor('black'); spine.set_linewidth(2.0)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_color('black'); lbl.set_fontweight('bold')
    fig.legend(handles=handles_pm, labels=list(metric_colors_local.keys()),
               loc='lower center', ncol=len(metric_colors_local),
               fontsize=17, framealpha=1.0, edgecolor='black',
               prop={'weight': 'bold', 'size': 17},
               bbox_to_anchor=(0.5, 0.01))
    plt.tight_layout(rect=[0, 0.06, 1, 0.95], pad=0.3, h_pad=0.5, w_pad=0.6)
    plt.savefig(os.path.join(COND_DIR, 'recovery_metrics_per_method.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: recovery_metrics_per_method.png")

    # RMSE by condition across missing %
    CONDITIONS_LOCAL = ['extreme_high_temp', 'high_temp', 'low_temp', 'normal']
    cond_colors = ['#C62828', '#FF9800', '#1565C0', '#6A1B9A']
    fig, axes = plt.subplots(2, 2, figsize=(22, 16))
    axes = axes.flatten()
    fig.suptitle('Recovery RMSE by Condition (per method)',
                 fontsize=24, fontweight='bold', color='black')
    handles_rc = []
    for ax_idx, (ax, method) in enumerate(zip(axes, METHODS)):
        sub_m = rec_df[rec_df['Method'] == method]
        for cond, clr in zip(CONDITIONS_LOCAL, cond_colors):
            vals = []
            for pct in pct_order:
                sub = sub_m[(sub_m['Condition'] == cond) &
                            (sub_m['Missing %'] == pct)]['RMSE']
                vals.append(sub.mean() if len(sub) > 0 else np.nan)
            if not all(np.isnan(v) for v in vals):
                line, = ax.plot(pct_order, vals, marker='o', lw=2.0, ms=6,
                        color=clr, label=cond, alpha=0.85)
                if ax_idx == 0:
                    handles_rc.append(line)
        ax.set_xlabel('Missing %', fontsize=19, fontweight='bold', color='black')
        ax.set_ylabel('Recovery RMSE (°C)', fontsize=19, fontweight='bold',
                      color='black')
        ax.set_title(method, fontsize=21, fontweight='bold', color='black')
        ax.tick_params(axis='both', labelsize=15, colors='black',
                       labelcolor='black', width=2)
        ax.grid(True, alpha=0.25)
        for spine in ax.spines.values():
            spine.set_edgecolor('black'); spine.set_linewidth(2.0)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_color('black'); lbl.set_fontweight('bold')
    if handles_rc:
        fig.legend(handles=handles_rc,
                   labels=[h.get_label().replace('_', ' ') for h in handles_rc],
                   loc='lower center', ncol=len(CONDITIONS_LOCAL),
                   fontsize=17, framealpha=1.0, edgecolor='black',
                   prop={'weight': 'bold', 'size': 17},
                   bbox_to_anchor=(0.5, 0.01))
    plt.tight_layout(rect=[0, 0.06, 1, 0.95], pad=0.3, h_pad=0.5, w_pad=0.6)
    plt.savefig(os.path.join(COND_DIR, 'recovery_rmse_by_condition_line.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: recovery_rmse_by_condition_line.png")

plot_recovery_line_graphs()


# RECOVERY vs PREDICTION SCATTER
def plot_recovery_vs_prediction():
    recovery_summary = (
        rec_df.groupby(['Method', 'Missing %'])['RMSE']
              .mean().reset_index()
              .rename(columns={'RMSE': 'Recovery_RMSE'})
    )
    pred_summary = (
        results_df[['Method', 'Missing %', 'RMSE']]
        .rename(columns={'RMSE': 'Prediction_RMSE'})
    )
    scatter_df = recovery_summary.merge(
        pred_summary, on=['Method', 'Missing %'])

    fig, ax = plt.subplots(figsize=(14, 11))
    for method in METHODS:
        sub = scatter_df[scatter_df['Method'] == method]
        ax.scatter(sub['Recovery_RMSE'], sub['Prediction_RMSE'],
                   label=method, color=METHOD_COLORS[method],
                   s=120, zorder=5, edgecolors='black', linewidths=0.8)
        for _, row in sub.iterrows():
            ax.annotate(row['Missing %'],
                        (row['Recovery_RMSE'], row['Prediction_RMSE']),
                        fontsize=18, fontweight='bold',
                        xytext=(5, 3), textcoords='offset points')
    ax.set_xlabel('Recovery RMSE at Gap Locations (C)',
                  fontsize=24, fontweight='bold', color='black')
    ax.set_ylabel('Model Prediction RMSE on Test Set (C)',
                  fontsize=24, fontweight='bold', color='black')
    ax.set_title(
        'Recovery Accuracy vs Downstream Model Performance\n'
        '(Chronological split, lag features)',
        fontsize=22, fontweight='bold', color='black')
    ax.tick_params(axis='both', labelsize=26, colors='black',
                   labelcolor='black', width=2.5)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color('black'); lbl.set_fontweight('bold')
    ax.legend(fontsize=19, framealpha=1.0, edgecolor='black')
    ax.grid(True, alpha=0.25)
    for spine in ax.spines.values():
        spine.set_edgecolor('black'); spine.set_linewidth(2.0)
    plt.tight_layout(pad=1.5)
    plt.savefig(os.path.join(COND_DIR, 'recovery_vs_prediction_scatter.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: recovery_vs_prediction_scatter.png")

plot_recovery_vs_prediction()


# RECOVERY vs PREDICTION vs VARIABILITY TABLE
recovery_summary = (
    rec_df.groupby(['Method', 'Missing %'])['RMSE']
          .mean().reset_index()
          .rename(columns={'RMSE': 'Recovery_RMSE'})
)
pred_summary = (
    results_df[['Method', 'Missing %', 'RMSE']]
    .rename(columns={'RMSE': 'Prediction_RMSE'})
)
variability_summary = pd.DataFrame(analysis_datasets)
analysis_df = (
    recovery_summary
    .merge(pred_summary,        on=['Method', 'Missing %'])
    .merge(variability_summary, on=['Method', 'Missing %'])
)
analysis_df.to_excel(os.path.join(COND_DIR,
    'recovery_prediction_variability_analysis.xlsx'), index=False)
analysis_df.to_csv(os.path.join(COND_DIR,
    'recovery_prediction_variability_analysis.csv'), index=False)
print("\n=== Recovery vs Prediction Analysis ===")
print(analysis_df.round(3).to_string(index=False))

rec_df.to_csv(os.path.join(COND_DIR, 'recovery_records_raw.csv'), index=False)

METHOD_CLR = METHOD_COLORS
CONDITIONS = ['extreme_high_temp', 'high_temp', 'low_temp', 'normal']
BOUNDARIES = ['stable', 'rising', 'falling', 'crossing']
LENGTHS    = ['short', 'medium', 'long']


def pivot_for(df, group_col):
    return (df.groupby([group_col, 'Method'])['RMSE']
              .mean().unstack('Method').reindex(columns=METHODS))


def plot_grouped_bar(grouped_df, title, xlabel, savepath):
    categories = grouped_df.index.tolist()
    x = np.arange(len(categories)); w = 0.18
    fig, ax = plt.subplots(figsize=(max(10, len(categories) * 2.2), 6))
    for j, method in enumerate(METHODS):
        if method not in grouped_df.columns: continue
        ax.bar(x + j * w, grouped_df[method].values, w,
               label=method, color=METHOD_CLR[method], alpha=0.85)
    ax.set_title(title, fontweight='bold', color='black')
    ax.set_xlabel(xlabel); ax.set_ylabel('Mean RMSE (C)')
    ax.set_xticks(x + w * 1.5)
    ax.set_xticklabels(categories, rotation=15, ha='right')
    ax.legend(fontsize=18); ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout(pad=1.5)
    plt.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {savepath}")


print("\n=== Recovery RMSE by meteorological condition ===")
cond_pivot = pivot_for(rec_df, 'Condition')
cond_pivot = cond_pivot.reindex(
    [c for c in CONDITIONS if c in cond_pivot.index])
print(cond_pivot.round(3).to_string())
cond_pivot.to_csv(os.path.join(COND_DIR, 'recovery_by_condition.csv'))
plot_grouped_bar(cond_pivot, 'Recovery RMSE by Meteorological Condition',
                 'Condition',
                 os.path.join(COND_DIR, 'recovery_by_condition.png'))

print("\n=== Recovery RMSE by boundary context ===")
bound_pivot = pivot_for(rec_df, 'Boundary')
bound_pivot = bound_pivot.reindex(
    [b for b in BOUNDARIES if b in bound_pivot.index])
print(bound_pivot.round(3).to_string())
bound_pivot.to_csv(os.path.join(COND_DIR, 'recovery_by_boundary.csv'))
plot_grouped_bar(bound_pivot, 'Recovery RMSE by Boundary Context',
                 'Boundary type',
                 os.path.join(COND_DIR, 'recovery_by_boundary.png'))

print("\n=== Recovery RMSE by gap length ===")
len_pivot = pivot_for(rec_df, 'Length')
len_pivot = len_pivot.reindex(
    [l for l in LENGTHS if l in len_pivot.index])
print(len_pivot.round(3).to_string())
len_pivot.to_csv(os.path.join(COND_DIR, 'recovery_by_length.csv'))
plot_grouped_bar(len_pivot, 'Recovery RMSE by Gap Length',
                 'Gap length',
                 os.path.join(COND_DIR, 'recovery_by_length.png'))

# Heatmaps — single shared colorbar across all 4 method panels
for heatmap_col, heatmap_vals, fname, fig_size in [
    ('Boundary', BOUNDARIES, 'heatmap_condition_x_boundary.png', (44, 34)),
    ('Length',   LENGTHS,    'heatmap_condition_x_length.png',   (38, 34)),
]:
    # Clean, concise single-line x-axis labels (no redundant sub-descriptions)
    label_map = {
        'short':    'Short',
        'medium':   'Medium',
        'long':     'Long',
        'stable':   'Stable',
        'rising':   'Rising',
        'falling':  'Falling',
        'crossing': 'Crossing',
    }
    display_labels = [label_map.get(v, v) for v in heatmap_vals]

    # First pass — compute all matrices to find global vmin/vmax
    all_mats     = {}
    all_cnt_mats = {}
    for method in METHODS:
        sub     = rec_df[rec_df['Method'] == method]
        mat     = np.full((len(CONDITIONS), len(heatmap_vals)), np.nan)
        cnt_mat = np.zeros_like(mat, dtype=int)
        for ci, cond in enumerate(CONDITIONS):
            for vi, val in enumerate(heatmap_vals):
                cell = sub[(sub['Condition'] == cond) &
                           (sub[heatmap_col] == val)]['RMSE']
                if len(cell) > 0:
                    mat[ci, vi]     = cell.mean()
                    cnt_mat[ci, vi] = len(cell)
        all_mats[method]     = mat
        all_cnt_mats[method] = cnt_mat

    # Global colour scale
    all_valid = np.concatenate(
        [m[~np.isnan(m)] for m in all_mats.values()])
    g_vmin = float(all_valid.min()) if len(all_valid) else 0.0
    g_vmax = float(all_valid.max()) if len(all_valid) else 1.0

    # Build figure: 2x2 subplots + thin column for shared colorbar
    fig = plt.figure(figsize=fig_size)
    gs_bottom = 0.05 if heatmap_col == 'Length' else 0.02
    gs  = fig.add_gridspec(2, 3,
                           width_ratios=[1, 1, 0.025],
                           hspace=0.10, wspace=0.06,
                           top=0.99, bottom=gs_bottom,
                           left=0.13, right=0.97)
    subplot_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    axes_list = [fig.add_subplot(gs[r, c]) for r, c in subplot_positions]
    cbar_ax   = fig.add_subplot(gs[:, 2])

    cond_label_map = {
        'extreme_high_temp': 'Ext. High Temp',
        'high_temp':         'High Temp',
        'low_temp':          'Low Temp',
        'stagnant_wind':     'Stagnant Wind',
        'normal':            'Normal',
    }
    y_labels = [cond_label_map.get(c, c) for c in CONDITIONS]

    im_last = None
    # is_left_col: True for idx 0,2 (left column), False for idx 1,3 (right column)
    is_left_col = [True, False, True, False]

    for idx, (ax, method) in enumerate(zip(axes_list, METHODS)):
        mat     = all_mats[method]
        cnt_mat = all_cnt_mats[method]
        im_h    = ax.imshow(mat, cmap='RdYlGn_r', aspect='auto',
                             vmin=g_vmin, vmax=g_vmax,
                             interpolation='nearest')
        im_last = im_h

        ax.set_xticks(range(len(heatmap_vals)))
        ax.set_xticklabels(display_labels, rotation=0,
                            fontsize=44, fontweight='bold', color='black')
        ax.tick_params(axis='x', which='both', length=0, colors='black')

        ax.set_yticks(range(len(CONDITIONS)))
        # Only show y-labels on the left column — no redundant labels on right
        if is_left_col[idx]:
            ax.set_yticklabels(y_labels, fontsize=44, fontweight='bold',
                                color='black')
        else:
            ax.set_yticklabels([])
        ax.tick_params(axis='y', which='both', length=0, colors='black')

        ax.grid(False)
        ax.set_xticks([], minor=True)
        ax.set_yticks([], minor=True)

        ax.set_title(method, fontweight='bold', fontsize=52, color='black',
                     pad=10)
        for spine in ax.spines.values():
            spine.set_edgecolor('black'); spine.set_linewidth(3.0)

        # Cell annotations — RMSE value (top) and n= count (bottom), both large
        for ci in range(len(CONDITIONS)):
            for vi in range(len(heatmap_vals)):
                v = mat[ci, vi]; n = cnt_mat[ci, vi]
                if not np.isnan(v):
                    ax.text(vi, ci - 0.15, f'{v:.2f}',
                            ha='center', va='center',
                            fontsize=46, fontweight='bold', color='black')
                    ax.text(vi, ci + 0.22, f'n={n}',
                            ha='center', va='center',
                            fontsize=36, fontweight='bold', color='black')
                else:
                    ax.text(vi, ci, 'n/a',
                            ha='center', va='center',
                            fontsize=44, fontweight='bold', color='black')

    # ONE shared colorbar
    cbar = fig.colorbar(im_last, cax=cbar_ax)
    cbar.set_label('Mean Recovery RMSE (°C)',
                   fontsize=38, fontweight='bold', color='black', labelpad=10)
    cbar.ax.tick_params(labelsize=36, colors='black')
    cbar.ax.yaxis.label.set_color('black')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='black',
             fontweight='bold')

    # For the Length heatmap only: single-line key below the bottom subplots
    if heatmap_col == 'Length':
        key_text = ('Gap Length Key  —  '
                    'Short: ≤6 steps (≤3 hrs)     '
                    'Medium: 7–24 steps (3–12 hrs)     '
                    'Long: >24 steps (>12 hrs)')
        fig.text(0.52, 0.004, key_text,
                 ha='center', va='bottom',
                 fontsize=34, fontweight='bold', color='black',
                 bbox=dict(boxstyle='round,pad=0.4',
                           facecolor='white', edgecolor='black',
                           linewidth=2.0, alpha=0.92))

    plt.savefig(os.path.join(COND_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")

# Best method summaries
for group_col, group_vals, fname in [
    ('Condition', CONDITIONS, 'best_method_per_condition.csv'),
    ('Boundary',  BOUNDARIES, 'best_method_per_boundary.csv'),
    ('Length',    LENGTHS,    'best_method_per_length.csv'),
]:
    rows = []
    for val in group_vals:
        sub = rec_df[rec_df[group_col] == val]
        if sub.empty: continue
        method_rmse = sub.groupby('Method')['RMSE'].mean().reindex(METHODS)
        best  = method_rmse.idxmin(); worst = method_rmse.idxmax()
        row   = {group_col: val, 'Best': best,
                 'Best_RMSE': round(method_rmse[best], 3),
                 'Worst': worst,
                 'Worst_RMSE': round(method_rmse[worst], 3)}
        for m in METHODS:
            row[f'{m}_RMSE'] = round(method_rmse[m], 3)
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(COND_DIR, fname), index=False)


# FINAL SUMMARY
print("\n=== FINAL SUMMARY — Recovery RMSE vs Prediction RMSE ===")
rec_s  = rec_df.groupby(['Method', 'Missing %'])['RMSE']\
               .mean().reset_index()\
               .rename(columns={'RMSE': 'Recovery_RMSE'})
pred_s = results_df[['Method', 'Missing %', 'RMSE']]\
               .rename(columns={'RMSE': 'Prediction_RMSE'})
final  = rec_s.merge(pred_s, on=['Method', 'Missing %'])
print(final.sort_values(['Missing %', 'Recovery_RMSE']).to_string(index=False))


# OUTPUT STRUCTURE
print("\n=== Output structure ===")
for root_d, dirs, files in os.walk(ROOT):
    level  = root_d.replace(ROOT, '').count(os.sep)
    indent = '  ' * level
    print(f"{indent}{os.path.basename(root_d)}/")
    for f in sorted(files):
        print(f"{indent}  {f}")

print("\nDone — Experiment 3 complete.")


# PAPER FIGURES
# Generate Figure 1 (missing gaps first quarter)
# and Figure 2 (combined error histograms)
print("\n=== Generating Paper Figures ===")

# Figure 1: Missing gaps — first quarter of training data
fi    = df_train_only.index
total = len(fi)
q_end = fi[total // 4]

fig, axes = plt.subplots(4, 1, figsize=(32, 42))

for ax, pct_label in zip(axes, pct_order):
    # get corrupted df — rebuild from all_method_dfs or all_gap_row_sets
    # use the gap row sets to reconstruct which rows are NaN
    gap_row_sets_here = all_gap_row_sets[pct_label]
    all_rows_here     = sorted(
        set(r for block in gap_row_sets_here for r in block))
    actual_pct_here   = len(all_rows_here) / total * 100

    df_corr_here = df_train_only.copy()
    df_corr_here.iloc[
        all_rows_here,
        df_corr_here.columns.get_loc('AirTC_18m')
    ] = np.nan

    orig_q1 = df_train_only.loc[fi[0]:q_end, 'AirTC_18m']
    corr_q1 = df_corr_here.loc[fi[0]:q_end, 'AirTC_18m']

    # shade gap regions
    gap_ts = corr_q1[corr_q1.isna()].index
    if len(gap_ts) > 0:
        gap_arr    = gap_ts.sort_values()
        dt         = pd.Timedelta(minutes=15)
        in_span    = False
        span_start = None
        for i, ts in enumerate(gap_arr):
            if not in_span:
                span_start = ts; in_span = True
            if i == len(gap_arr) - 1 or \
               (gap_arr[i+1] - ts) > pd.Timedelta(minutes=35):
                ax.axvspan(span_start - dt, ts + dt,
                           color='#EF5350', alpha=0.4, zorder=0)
                in_span = False

    ax.plot(orig_q1.index, orig_q1.values,
            color='#1565C0', lw=2.0, alpha=0.5,
            ls='--', label='Original (clean)', zorder=2)
    ax.plot(corr_q1.index, corr_q1.values,
            color='#212121', lw=2.0, alpha=0.9,
            label='With gaps', zorder=3)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.tick_params(axis='x', rotation=0, labelsize=42,
                   colors='black', width=3.0)
    ax.tick_params(axis='y', labelsize=42, colors='black', width=3.0)
    ax.set_ylabel('AirTC_18m (°C)', fontsize=46,
                  fontweight='bold', color='black')
    ax.set_title(
        f'{pct_label} missing  (actual: {actual_pct_here:.1f}%)  '
        f'— First Quarter (Jan–Mar 2017)',
        fontsize=46, fontweight='bold', color='black')
    ax.grid(True, alpha=0.25, color='grey')
    ax.set_xlim(fi[0], q_end)
    for spine in ax.spines.values():
        spine.set_edgecolor('black'); spine.set_linewidth(2.5)

legend_elements_gap = [
    plt.Line2D([0], [0], color='#1565C0', lw=4.0,
               ls='--', alpha=0.7, label='Original (clean)'),
    plt.Line2D([0], [0], color='#212121', lw=4.0,
               label='With gaps'),
    Patch(facecolor='#EF5350', alpha=0.4,
          label='Missing region'),
]
fig.legend(handles=legend_elements_gap,
           loc='lower center', ncol=3,
           fontsize=44, framealpha=1.0, edgecolor='black',
           prop={'weight': 'bold', 'size': 44},
           bbox_to_anchor=(0.5, 0.0))

plt.tight_layout(rect=[0, 0.035, 1, 1.0], pad=2.5)
path_fig1 = os.path.join(ROOT, 'missing_gaps_first_quarter.png')
plt.savefig(path_fig1, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {path_fig1}")

# Figure 2: Combined error histograms
# collect error values per method per pct level
all_level_errors = {}
for pct_label in pct_order:
    gap_row_sets_here = all_gap_row_sets[pct_label]
    level_errors = {}
    for method in METHODS:
        key       = (method, pct_label)
        df_interp = all_method_dfs.get(key, None)
        if df_interp is None:
            level_errors[method] = []
            continue
        errors = []
        for block in gap_row_sets_here:
            actual = df_train_only.iloc[block]['AirTC_18m'].values
            filled = df_interp.iloc[block]['AirTC_18m'].values
            valid  = ~(np.isnan(actual) | np.isnan(filled))
            if valid.sum() > 0:
                errors.extend((filled[valid] - actual[valid]).tolist())
        level_errors[method] = errors
    all_level_errors[pct_label] = level_errors

fig, axes = plt.subplots(2, 2, figsize=(26, 20), constrained_layout=True)
axes = axes.flatten()
fig.suptitle(
    'Interpolation Error Distribution at Gap Locations\n'
    '(Filled − Actual Temperature, °C) — KDE curves per method',
    fontsize=22, fontweight='bold')

for ax, pct_label in zip(axes, pct_order):
    level_errors = all_level_errors[pct_label]
    all_vals = [e for errs in level_errors.values() for e in errs]
    if not all_vals:
        ax.set_visible(False); continue

    bin_min = np.percentile(all_vals, 1)
    bin_max = np.percentile(all_vals, 99)
    x_range = np.linspace(bin_min, bin_max, 400)

    for method in METHODS:
        errors = level_errors[method]
        if not errors: continue
        errors_arr = np.array(errors)
        clr        = METHOD_COLORS[method]
        kde        = gaussian_kde(errors_arr, bw_method=0.3)
        ax.plot(x_range, kde(x_range), color=clr, lw=3.0)
        ax.fill_between(x_range, kde(x_range),
                        alpha=0.08, color=clr)
        ax.axvline(np.mean(errors_arr), color=clr,
                   lw=2.0, ls='--', alpha=0.85)

    ax.axvline(0, color='black', lw=2.5, ls='-', zorder=5)

    # single n value (same for all methods at a given %)
    n_val = len(list(level_errors.values())[0])
    ax.text(0.02, 0.97,
            f'n = {n_val:,} errors per method',
            transform=ax.transAxes,
            fontsize=16, fontweight='bold', color='black',
            va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.4', fc='white',
                      ec='black', alpha=1.0))

    ax.set_xlabel('Interpolation Error (Filled − Actual, °C)',
                  fontsize=17, fontweight='bold', color='black')
    ax.set_ylabel('Density', fontsize=17,
                  fontweight='bold', color='black')
    ax.set_title(
        f'{pct_label} missing',
        fontsize=20, fontweight='bold', color='black')
    ax.tick_params(axis='both', labelsize=15,
                   colors='black', width=2)
    ax.grid(True, alpha=0.25, color='grey')
    for spine in ax.spines.values():
        spine.set_edgecolor('black'); spine.set_linewidth(2.0)

# shared legend below all panels
legend_elements_hist = [
    plt.Line2D([0], [0], color=METHOD_COLORS[m], lw=3.0, label=m)
    for m in METHODS
] + [
    plt.Line2D([0], [0], color='black', lw=2.5, ls='-',
               label='Zero error'),
    plt.Line2D([0], [0], color='grey', lw=2.0, ls='--',
               label='Method mean error'),
]
fig.legend(handles=legend_elements_hist,
           loc='lower center',
           ncol=len(legend_elements_hist),
           fontsize=15,
           framealpha=1.0,
           edgecolor='black',
           bbox_to_anchor=(0.5, -0.04),
           prop={'weight': 'bold', 'size': 15})

plt.tight_layout(pad=1.5)
path_fig2 = os.path.join(ROOT, 'error_histograms_combined.png')
plt.savefig(path_fig2, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {path_fig2}")