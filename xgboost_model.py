import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'text.color':           'black',
    'font.family':          'DejaVu Sans',
    'font.weight':          'bold',
    'font.size':            14,
    'axes.labelcolor':      'black',
    'axes.labelweight':     'bold',
    'axes.labelsize':       14,
    'axes.titlesize':       15,
    'axes.titleweight':     'bold',
    'axes.edgecolor':       'black',
    'axes.linewidth':       1.5,
    'xtick.color':          'black',
    'ytick.color':          'black',
    'xtick.labelsize':      12,
    'ytick.labelsize':      12,
    'xtick.major.width':    1.5,
    'ytick.major.width':    1.5,
    'legend.fontsize':      12,
    'legend.title_fontsize':12,
    'legend.edgecolor':     'black',
    'legend.framealpha':    1.0,
    'figure.titlesize':     16,
    'figure.titleweight':   'bold',
    'savefig.facecolor':    'white',
    'savefig.edgecolor':    'black',
})

from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.stattools import acf, pacf
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

OUTPUT_ROOT       = "outputs"
FOLDER_GAP_PLOTS  = os.path.join(OUTPUT_ROOT, "gap_visualisations")
FOLDER_RESULTS    = os.path.join(OUTPUT_ROOT, "model_results")
FOLDER_COMPARISON = os.path.join(OUTPUT_ROOT, "comparison_reports")
FOLDER_EXTRA      = os.path.join(OUTPUT_ROOT, "extra_analysis")

for folder in [FOLDER_GAP_PLOTS, FOLDER_RESULTS, FOLDER_COMPARISON, FOLDER_EXTRA]:
    os.makedirs(folder, exist_ok=True)

TEST_START = pd.Timestamp("2017-11-01")

# Lag selection — justified by ACF, PACF, and cross-correlation
# TARGET variable: AirTC_18m (temperature)

JUSTIFIED_LAGS = [1, 2, 3, 4, 6, 12, 48]   # AirTC own lags (unchanged)

# Per-variable lag sets for multivariate feature engineering
LAGS_AIRTEMP = [1, 2, 3, 4, 6, 12, 48]   # own lags — full set
LAGS_RH      = [1, 2, 4, 6, 12, 48]       # strong cross-correlation at all
LAGS_WS      = [1, 4, 6, 48]              # weak but significant cross-corr
LAGS_WINDDIR = [1, 6, 12]                 # only statistically significant lags


def out(folder, filename):
    return os.path.join(folder, filename)

# LOAD & CLEAN
def load_and_clean(filepath):
    FEATURES_RAW = ["RH_18m", "WS_ms_18m_Avg", "WindDir_18m"]
    TARGET       = "AirTC_18m"
    COLS         = ["TIMESTAMP"] + FEATURES_RAW + [TARGET]

    df = pd.read_csv(filepath, parse_dates=["TIMESTAMP"], dayfirst=True)
    df = df.loc[:, df.columns.str.strip() != '']
    df.columns = df.columns.str.strip()
    print("Columns found:", df.columns.tolist())

    df = df[COLS].copy()
    df.dropna(inplace=True)

    for col in FEATURES_RAW + [TARGET]:
        df = df[df[col] > -50]

    df.reset_index(drop=True, inplace=True)
    print(f"Clean dataset shape : {df.shape}")
    print(f"Date range          : {df['TIMESTAMP'].min()} to {df['TIMESTAMP'].max()}")
    return df


# ACF / PACF + CROSS-CORRELATION ANALYSIS
def plot_acf_pacf_lag_selection(df_clean, target='AirTC_18m'):
    """
    Plot ACF and PACF for temperature (own lags) plus cross-correlation
    between each predictor variable and temperature at different lags.
    Both justify the lag selection used in build_features.
    """
    from scipy.stats import pearsonr

    df_sorted = df_clean.sort_values('TIMESTAMP')
    temp      = df_sorted[target].values
    rh        = df_sorted['RH_18m'].values
    ws        = df_sorted['WS_ms_18m_Avg'].values
    wd        = df_sorted['WindDir_18m'].values

    n_lags   = 100
    acf_vals  = acf(temp, nlags=n_lags, fft=True)
    pacf_vals = pacf(temp, nlags=min(50, len(temp)//2 - 1))
    conf      = 1.96 / np.sqrt(len(temp))

    # Plot 1: ACF/PACF of temperature
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), constrained_layout=True)
    fig.suptitle('ACF and PACF — AirTC_18m Lag Justification\n'
                 f'Own lags selected: {LAGS_AIRTEMP} (each step = 30 min)',
                 fontsize=14, fontweight='bold')

    lags_acf = np.arange(len(acf_vals))
    axes[0].bar(lags_acf, acf_vals, color='#1565C0', alpha=0.5, width=0.8)
    axes[0].axhline( conf, color='red', ls='--', lw=1.5, alpha=0.8,
                     label=f'95% CI (±{conf:.3f})')
    axes[0].axhline(-conf, color='red', ls='--', lw=1.5, alpha=0.8)
    axes[0].axhline(0, color='black', lw=1.0)
    for lag in LAGS_AIRTEMP:
        if lag < len(acf_vals):
            axes[0].bar(lag, acf_vals[lag], color='#E53935', alpha=0.9, width=0.8)
            axes[0].text(lag, acf_vals[lag] + 0.02, f'{acf_vals[lag]:.3f}',
                         ha='center', fontsize=11, fontweight='bold',
                         color='#B71C1C')
    axes[0].set_xlabel('Lag (steps, 1 step = 30 min)', fontsize=12)
    axes[0].set_ylabel('ACF', fontsize=12)
    axes[0].set_title('Autocorrelation Function (ACF) of AirTC_18m',
                      fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].set_xlim(-1, n_lags + 1)
    axes[0].grid(True, alpha=0.25)

    lags_pacf = np.arange(len(pacf_vals))
    axes[1].bar(lags_pacf, pacf_vals, color='#2E7D32', alpha=0.5, width=0.8)
    axes[1].axhline( conf, color='red', ls='--', lw=1.5, alpha=0.8,
                     label=f'95% CI (±{conf:.3f})')
    axes[1].axhline(-conf, color='red', ls='--', lw=1.5, alpha=0.8)
    axes[1].axhline(0, color='black', lw=1.0)
    for lag in LAGS_AIRTEMP:
        if lag < len(pacf_vals):
            axes[1].bar(lag, pacf_vals[lag], color='#E53935', alpha=0.9,
                        width=0.8)
            axes[1].text(lag, pacf_vals[lag] + 0.01, f'{pacf_vals[lag]:.3f}',
                         ha='center', fontsize=11, fontweight='bold',
                         color='#B71C1C')
    axes[1].set_xlabel('Lag (steps, 1 step = 30 min)', fontsize=12)
    axes[1].set_ylabel('PACF', fontsize=12)
    axes[1].set_title('Partial Autocorrelation Function (PACF) — confirms AR order',
                      fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].set_xlim(-1, min(51, len(pacf_vals) + 1))
    axes[1].grid(True, alpha=0.25)

    plt.tight_layout(pad=2.0)
    path = out(FOLDER_EXTRA, 'acf_pacf_lag_justification.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ACF/PACF plot -> {path}")

    # Plot 2: Cross-correlation of predictors vs AirTC 
    check_lags = [1, 2, 3, 4, 6, 12, 24, 48]
    var_info = {
        'RH_18m':        (rh,  '#1565C0', LAGS_RH,      'Relative Humidity'),
        'WS_ms_18m_Avg': (ws,  '#2E7D32', LAGS_WS,      'Wind Speed'),
        'WindDir_18m':   (wd,  '#F57C00', LAGS_WINDDIR,  'Wind Direction'),
    }

    fig, axes = plt.subplots(1, 3, figsize=(20, 7), constrained_layout=True)
    fig.suptitle('Cross-Correlation: Predictor Variables vs AirTC_18m\n'
                 '(Pearson r at each lag — selected lags marked in red)',
                 fontsize=14, fontweight='bold')

    for ax, (vname, (vseries, clr, sel_lags, vlabel)) in \
            zip(axes, var_info.items()):
        rs = []
        for lag in check_lags:
            r, _ = pearsonr(vseries[:-lag], temp[lag:])
            rs.append(r)

        bars = ax.bar(range(len(check_lags)), rs,
                      color=[('#E53935' if check_lags[i] in sel_lags
                               else clr)
                             for i in range(len(check_lags))],
                      alpha=0.8, edgecolor='black', lw=0.5)
        ax.axhline(0, color='black', lw=1.0)
        ax.axhline( 0.05, color='grey', ls='--', lw=1.0, alpha=0.6)
        ax.axhline(-0.05, color='grey', ls='--', lw=1.0, alpha=0.6)
        ax.set_xticks(range(len(check_lags)))
        ax.set_xticklabels([f'{l}\n({l*0.5:.1f}h)' for l in check_lags],
                            fontsize=11)
        ax.set_xlabel('Lag (steps)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Pearson r with AirTC_18m', fontsize=11,
                      fontweight='bold')
        ax.set_title(f'{vlabel} ({vname})\nSelected lags: {sel_lags}',
                     fontweight='bold', fontsize=11)
        ax.grid(True, alpha=0.25, axis='y')
        for i, (lag, r) in enumerate(zip(check_lags, rs)):
            ax.text(i, r + (0.01 if r >= 0 else -0.02),
                    f'{r:+.3f}', ha='center', fontsize=11,
                    fontweight='bold',
                    color='#B71C1C' if lag in sel_lags else 'black')
        for spine in ax.spines.values():
            spine.set_edgecolor('black'); spine.set_linewidth(1.5)

    plt.tight_layout(pad=2.0)
    path2 = out(FOLDER_EXTRA, 'cross_correlation_lag_justification.png')
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Cross-correlation plot -> {path2}")

    # Print summary
    print("\n  === Lag Selection Justification ===")
    print(f"\n  AirTC own lags {LAGS_AIRTEMP}:")
    reasons_temp = {
        1: 'ACF=0.991, strongest autocorrelation',
        2: 'ACF=0.972, strong 1h persistence',
        3: 'ACF=0.945, PACF significant',
        4: 'ACF=0.911, PACF significant, AR(4) confirmed',
        6: 'ACF=0.829, sub-diurnal transitions',
        12:'ACF=0.522, half-day cycle',
        48:'ACF=0.930, dominant diurnal cycle',
    }
    for lag in LAGS_AIRTEMP:
        print(f"    lag {lag:2d} ({lag*0.5:4.1f}h): {reasons_temp[lag]}")

    print(f"\n  RH lags {LAGS_RH}:")
    print("    Cross-correlation r=-0.68 to -0.65 at lags 1,2,4,6,12,48")
    print("    Strong negative relationship (when temp rises, RH drops)")
    print("    Diurnal cycle preserved at lag 48 (r=-0.645)")

    print(f"\n  Wind Speed lags {LAGS_WS}:")
    print("    Peak cross-correlation at lag 4-6 (r=+0.085)")
    print("    lag 48 retains diurnal signal (r=+0.082)")
    print("    lag 12 excluded — cross-correlation drops to r=0.015 (not significant)")

    print(f"\n  Wind Direction lags {LAGS_WINDDIR}:")
    print("    lag 12 most informative (r=-0.092, p=2e-32)")
    print("    lag 48 excluded — r=-0.007, p=0.35 (not significant)")
    print("    Applied to sin/cos encoding of wind direction")

    return JUSTIFIED_LAGS


# FEATURE ENGINEERING
# Lag features for ALL variables with variable-
# specific lag sets justified by cross-correlation
# and ACF analysis.
def build_features(df_input, target='AirTC_18m'):
    df = df_input.copy()

    # Time features
    df["Hour"]      = df["TIMESTAMP"].dt.hour + df["TIMESTAMP"].dt.minute / 60
    df["Month"]     = df["TIMESTAMP"].dt.month
    df["Hour_sin"]  = np.sin(2 * np.pi * df["Hour"]  / 24)
    df["Hour_cos"]  = np.cos(2 * np.pi * df["Hour"]  / 24)
    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)
    df["timestamp_ordinal"] = df["TIMESTAMP"].map(
        pd.Timestamp.toordinal).astype(float)

    # Wind direction circular encoding
    df["WindDir_sin"] = np.sin(np.radians(df["WindDir_18m"]))
    df["WindDir_cos"] = np.cos(np.radians(df["WindDir_18m"]))

    # Lag features MUST be computed on chronologically sorted data
    df = df.sort_values("TIMESTAMP").reset_index(drop=True)

    FEATURES_ENC = [
    # time features only — always known at prediction time
    "Hour_sin", "Hour_cos",
    "Month_sin", "Month_cos",
    "timestamp_ordinal",
]

    # AirTC own lags (ACF/PACF justified)
    for lag in LAGS_AIRTEMP:
        col = f'AirTC_18m_lag{lag}'
        df[col] = df['AirTC_18m'].shift(lag)
        FEATURES_ENC.append(col)

    # AirTC rolling features
    df['AirTC_18m_roll6_mean']  = df['AirTC_18m'].shift(1).rolling(6).mean()
    df['AirTC_18m_roll12_mean'] = df['AirTC_18m'].shift(1).rolling(12).mean()
    df['AirTC_18m_roll6_std']   = df['AirTC_18m'].shift(1).rolling(6).std()
    FEATURES_ENC += ['AirTC_18m_roll6_mean',
                     'AirTC_18m_roll12_mean',
                     'AirTC_18m_roll6_std']

    # RH lags (strong cross-correlation r=-0.68 to -0.65)
    for lag in LAGS_RH:
        col = f'RH_18m_lag{lag}'
        df[col] = df['RH_18m'].shift(lag)
        FEATURES_ENC.append(col)
    df['RH_18m_roll6_mean'] = df['RH_18m'].shift(1).rolling(6).mean()
    df['RH_18m_roll6_std']  = df['RH_18m'].shift(1).rolling(6).std()
    FEATURES_ENC += ['RH_18m_roll6_mean', 'RH_18m_roll6_std']

    # Wind Speed lags (peak cross-correlation at lag 4-6)
    for lag in LAGS_WS:
        col = f'WS_ms_18m_Avg_lag{lag}'
        df[col] = df['WS_ms_18m_Avg'].shift(lag)
        FEATURES_ENC.append(col)
    df['WS_ms_18m_Avg_roll6_mean'] = df['WS_ms_18m_Avg'].shift(1).rolling(6).mean()
    FEATURES_ENC.append('WS_ms_18m_Avg_roll6_mean')

    # ── Wind Direction lags (applied to sin/cos — significant lags only) ──
    for lag in LAGS_WINDDIR:
        sin_col = f'WindDir_sin_lag{lag}'
        cos_col = f'WindDir_cos_lag{lag}'
        df[sin_col] = df['WindDir_sin'].shift(lag)
        df[cos_col] = df['WindDir_cos'].shift(lag)
        FEATURES_ENC += [sin_col, cos_col]

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"  Features built: {len(FEATURES_ENC)} total")
    print(f"    Time:         5")
    print(f"    AirTC lags:   {len(LAGS_AIRTEMP)} lags + 3 rolling = "
        f"{len(LAGS_AIRTEMP)+3}")
    print(f"    RH lags:      {len(LAGS_RH)} lags + 2 rolling = "
        f"{len(LAGS_RH)+2}")
    print(f"    WS lags:      {len(LAGS_WS)} lags + 1 rolling = "
        f"{len(LAGS_WS)+1}")
    print(f"    WindDir lags: {len(LAGS_WINDDIR)} lags x2 (sin/cos) = "
        f"{len(LAGS_WINDDIR)*2}")

    return df, FEATURES_ENC


# CHRONOLOGICAL 80/10/10 SPLIT
def _chrono_split(df):
    df  = df.sort_values("TIMESTAMP").reset_index(drop=True)
    n   = len(df)
    n_train = int(n * 0.80)
    n_val   = int(n * 0.10)

    train_df = df.iloc[:n_train].reset_index(drop=True)
    val_df   = df.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test_df  = df.iloc[n_train + n_val:].reset_index(drop=True)

    print(f"  Train: {len(train_df):,} "
          f"({train_df['TIMESTAMP'].iloc[0].strftime('%d %b %Y')} to "
          f"{train_df['TIMESTAMP'].iloc[-1].strftime('%d %b %Y')})")
    print(f"  Val  : {len(val_df):,} "
          f"({val_df['TIMESTAMP'].iloc[0].strftime('%d %b %Y')} to "
          f"{val_df['TIMESTAMP'].iloc[-1].strftime('%d %b %Y')})")
    print(f"  Test : {len(test_df):,} "
          f"({test_df['TIMESTAMP'].iloc[0].strftime('%d %b %Y')} to "
          f"{test_df['TIMESTAMP'].iloc[-1].strftime('%d %b %Y')})")
    return train_df, val_df, test_df


# OPTUNA TUNING
_BEST_PARAMS = None

def tune_hyperparameters(df_clean, target='AirTC_18m', n_trials=25):
    global _BEST_PARAMS

    if _BEST_PARAMS is not None:
        print("  [Optuna] Using cached params.")
        return _BEST_PARAMS

    print(f"  [Optuna] Starting ({n_trials} trials, chronological 80/10/10) ...")

    df, FEATURES_ENC = build_features(df_clean, target)
    train_df, val_df, _ = _chrono_split(df)

    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(train_df[FEATURES_ENC].values)
    X_val_s   = scaler.transform(val_df[FEATURES_ENC].values)
    y_train   = train_df[target].values
    y_val     = val_df[target].values

    def objective(trial):
        params = dict(
            n_estimators     = trial.suggest_int("n_estimators",
                                                  200, 1000, step=50),
            learning_rate    = trial.suggest_float("learning_rate",
                                                    0.01, 0.2,  log=True),
            max_depth        = trial.suggest_int("max_depth",          3, 10),
            subsample        = trial.suggest_float("subsample",        0.5, 1.0),
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.4, 1.0),
            min_child_weight = trial.suggest_int("min_child_weight",   1, 10),
            reg_alpha        = trial.suggest_float("reg_alpha",
                                                    1e-4, 10.0, log=True),
            reg_lambda       = trial.suggest_float("reg_lambda",
                                                    1e-4, 10.0, log=True),
            gamma            = trial.suggest_float("gamma",            0.0, 5.0),
            random_state=42, n_jobs=-1, verbosity=0,
        )
        model = XGBRegressor(**params)
        model.fit(X_train_s, y_train,
                  eval_set=[(X_val_s, y_val)], verbose=False)
        return float(np.sqrt(mean_squared_error(y_val,
                                                model.predict(X_val_s))))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    _BEST_PARAMS = study.best_params
    print(f"  [Optuna] Best RMSE: {study.best_value:.4f}")
    print(f"  [Optuna] Best params: {_BEST_PARAMS}")

    try:
        fig_hist = optuna.visualization.matplotlib\
            .plot_optimization_history(study)
        fig_hist.figure.savefig(
            out(FOLDER_EXTRA, "optuna_optimisation_history.png"),
            dpi=150, bbox_inches="tight")
        plt.close(fig_hist.figure)
        fig_imp = optuna.visualization.matplotlib\
            .plot_param_importances(study)
        fig_imp.figure.savefig(
            out(FOLDER_EXTRA, "optuna_param_importances.png"),
            dpi=150, bbox_inches="tight")
        plt.close(fig_imp.figure)
    except Exception as e:
        print(f"  [Optuna] Could not save plots: {e}")

    _write_hyperparam_report(_BEST_PARAMS, study.best_value)
    return _BEST_PARAMS


def _write_hyperparam_report(params, best_rmse):
    explanations = {
        "n_estimators":     "Number of boosting trees. Searched 200-1000.",
        "learning_rate":    "Shrinkage per tree. Log scale 0.01-0.2.",
        "max_depth":        "Max tree depth. Searched 3-10.",
        "subsample":        "Row subsampling per tree. Searched 0.5-1.0.",
        "colsample_bytree": "Feature subsampling per tree. Searched 0.4-1.0.",
        "min_child_weight": "Min samples in a leaf. Searched 1-10.",
        "reg_alpha":        "L1 regularisation. Log scale 1e-4 to 10.",
        "reg_lambda":       "L2 regularisation. Log scale 1e-4 to 10.",
        "gamma":            "Min loss reduction for a split. Searched 0-5.",
    }
    lines = [
        "=" * 70,
        "  XGBOOST HYPERPARAMETER TUNING REPORT",
        "  Chronological 80/10/10 | Multivariate lags (ACF + cross-corr)",
        "=" * 70,
        f"\n  Best validation RMSE: {best_rmse:.4f} degC\n",
        "  LAG FEATURES:",
        f"    AirTC own lags : {LAGS_AIRTEMP} (ACF/PACF justified)",
        f"    RH lags        : {LAGS_RH} (cross-corr r=-0.68 to -0.65)",
        f"    Wind Speed lags: {LAGS_WS} (cross-corr peak at lag 4-6)",
        f"    Wind Dir lags  : {LAGS_WINDDIR} (sig. lags only, on sin/cos)",
        "\n" + "-" * 70,
    ]
    for param, value in params.items():
        val_str = f"{value:.6f}" if isinstance(value, float) else str(value)
        lines += [f"\n  PARAMETER : {param}", f"  VALUE     : {val_str}",
                  f"  REASON    : {explanations.get(param, 'N/A')}",
                  "-" * 70]
    
    path = out(OUTPUT_ROOT, "hyperparameter_report.txt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Hyperparameter report -> {path}")


# TRAIN & EVALUATE — chronological 80/10/10
def train_and_evaluate(df_input, target='AirTC_18m', block_seed=42):
    global _BEST_PARAMS

    df, FEATURES_ENC = build_features(df_input, target)
    train_df, val_df, test_df = _chrono_split(df)

    X_train = train_df[FEATURES_ENC].values
    y_train = train_df[target].values
    X_val   = val_df[FEATURES_ENC].values
    y_val   = val_df[target].values
    X_test  = test_df[FEATURES_ENC].values
    y_test  = test_df[target].values
    ts_test = test_df["TIMESTAMP"].reset_index(drop=True)
    ts_val  = val_df["TIMESTAMP"].reset_index(drop=True)

    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    model_params = dict(_BEST_PARAMS, random_state=42,
                        n_jobs=-1, verbosity=0) \
        if _BEST_PARAMS else dict(
            n_estimators=500, learning_rate=0.05, max_depth=7,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=-1, verbosity=0)

    model = XGBRegressor(**model_params)
    model.fit(X_train_s, y_train,
              eval_set=[(X_train_s, y_train), (X_val_s, y_val)],
              verbose=False)
    evals_result = model.evals_result()

    y_pred     = model.predict(X_test_s)
    y_val_pred = model.predict(X_val_s)

    mae       = mean_absolute_error(y_test, y_pred)
    rmse      = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2        = r2_score(y_test, y_pred)
    residuals = y_test - y_pred
    bias      = float(np.mean(y_pred - y_test))
    val_rmse  = float(np.sqrt(mean_squared_error(y_val, y_val_pred)))
    val_r2    = r2_score(y_val, y_val_pred)

    print(f"  MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.4f}  Bias={bias:.3f}  "
          f"[Val RMSE={val_rmse:.3f}  Val R2={val_r2:.4f}]")

    return {
        'mae': mae, 'rmse': rmse, 'r2': r2, 'bias': bias,
        'y_test': y_test, 'y_pred': y_pred,
        'y_val': y_val, 'y_val_pred': y_val_pred,
        'ts_test': ts_test, 'ts_val': ts_val,
        'residuals': residuals,
        'model': model, 'features': FEATURES_ENC,
        'scaler': scaler, 'X_test_s': X_test_s,
        'evals_result': evals_result,
        'val_rmse': val_rmse, 'val_r2': val_r2,
        '_train_ts': train_df['TIMESTAMP'],
        '_val_ts':   val_df['TIMESTAMP'],
        '_test_ts':  test_df['TIMESTAMP'],
        '_train_y':  train_df[target].values,
        '_val_y':    val_df[target].values,
        '_test_y':   test_df[target].values,
    }


# FULL DATASET SPLIT PLOT
def plot_full_dataset_split(results, df_full_indexed, tag='base'):
    import matplotlib.dates as mdates

    ts_train   = pd.to_datetime(results['_train_ts'])
    ts_val     = pd.to_datetime(results['_val_ts'])
    ts_test    = pd.to_datetime(results['ts_test'])
    y_test     = results['y_test']
    y_pred     = results['y_pred']
    y_val      = results['y_val']
    y_val_pred = results['y_val_pred']
    val_rmse   = results['val_rmse']
    val_r2     = results['val_r2']

    # Plot 1: Full timeline
    fig, ax = plt.subplots(figsize=(22, 11))
    ax.plot(ts_train, results['_train_y'],
            color='#1565C0', lw=0.8, alpha=0.85,
            label=f'Training ({len(ts_train):,} samples, 80%)')
    ax.plot(ts_val, results['_val_y'],
            color='#FF9800', lw=0.8, alpha=0.85,
            label=f'Validation ({len(ts_val):,} samples, 10%)')
    ax.plot(ts_test, y_test,
            color='#212121', lw=0.9, alpha=0.85,
            label=f'Test — Actual ({len(ts_test):,} samples, 10%)')
    ax.plot(ts_test, y_pred,
            color='#E53935', lw=1.0, ls='--', alpha=0.85,
            label=f'Test — Predicted  RMSE={results["rmse"]:.3f} C  '
                  f'R2={results["r2"]:.4f}')
    ax.axvline(ts_val.iloc[0],  color='#FF9800', lw=1.8,
               ls='--', alpha=0.8, zorder=6)
    ax.axvline(ts_test.iloc[0], color='#212121', lw=1.8,
               ls='--', alpha=0.8, zorder=6)
    ax.axvspan(ts_train.iloc[0], ts_val.iloc[0],
               color='#1565C0', alpha=0.04, zorder=0)
    ax.axvspan(ts_val.iloc[0],  ts_test.iloc[0],
               color='#FF9800', alpha=0.06, zorder=0)
    ax.axvspan(ts_test.iloc[0], ts_test.iloc[-1],
               color='#E53935', alpha=0.04, zorder=0)
    ax.set_ylabel('Air Temperature (°C)', fontsize=26, fontweight='bold',
                  color='black')
    ax.set_xlabel('Date', fontsize=26, fontweight='bold', color='black')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.tick_params(axis='x', rotation=0, labelsize=22, colors='black', width=2.0)
    ax.tick_params(axis='y', labelsize=22, colors='black', width=2.0)
    ax.legend(fontsize=20, loc='upper right', framealpha=1.0,
              edgecolor='black', ncol=1, prop={'weight': 'bold', 'size': 20})
    ax.grid(True, alpha=0.2)
    # Add region labels at the bottom of the axes (transform-based) to
    # prevent overlap with the temperature line data
    ymin, ymax = ax.get_ylim()
    label_y = ymin + (ymax - ymin) * 0.04
    ax.text(ts_train.iloc[len(ts_train)//2], label_y, 'TRAINING (80%)',
            ha='center', va='bottom', fontsize=20, fontweight='bold',
            color='#1565C0', alpha=0.9,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='none',
                      alpha=0.7))
    ax.text(ts_val.iloc[len(ts_val)//2], label_y, 'VALIDATION\n(10%)',
            ha='center', va='bottom', fontsize=18, fontweight='bold',
            color='#E65100', alpha=0.95,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='none',
                      alpha=0.7))
    ax.text(ts_test.iloc[len(ts_test)//2], label_y, 'TEST (10%)',
            ha='center', va='bottom', fontsize=20, fontweight='bold',
            color='#212121', alpha=0.9,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='none',
                      alpha=0.7))
    for spine in ax.spines.values():
        spine.set_edgecolor('black'); spine.set_linewidth(2.0)
    plt.tight_layout(pad=2.0)
    path1 = out(FOLDER_RESULTS, f'full_dataset_split_overview_{tag}.png')
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Full dataset split overview -> {path1}")

    # Plot 2: Three panel detail
    fig, axes = plt.subplots(3, 1, figsize=(22, 20),
                              gridspec_kw={'height_ratios': [2, 1, 1]})
    fig.suptitle('Chronological 80/10/10 Split — Detail',
                 fontsize=15, fontweight='bold', color='black')

    ax = axes[0]
    ax.plot(ts_train, results['_train_y'],
            color='#1565C0', lw=0.7, alpha=0.85, label='Training (80%)')
    ax.plot(ts_val, results['_val_y'],
            color='#FF9800', lw=0.7, alpha=0.85, label='Validation (10%)')
    ax.plot(ts_test, y_test,
            color='#212121', lw=0.8, alpha=0.85, label='Test — Actual (10%)')
    ax.plot(ts_test, y_pred,
            color='#E53935', lw=1.0, ls='--', alpha=0.85,
            label='Test — Predicted')
    ax.axvline(ts_val.iloc[0],  color='#FF9800', lw=1.8, ls='--', alpha=0.7)
    ax.axvline(ts_test.iloc[0], color='#212121', lw=1.8, ls='--', alpha=0.7)
    ax.axvspan(ts_train.iloc[0], ts_val.iloc[0],
               color='#1565C0', alpha=0.04)
    ax.axvspan(ts_val.iloc[0],  ts_test.iloc[0],
               color='#FF9800', alpha=0.06)
    ax.axvspan(ts_test.iloc[0], ts_test.iloc[-1],
               color='#E53935', alpha=0.04)
    ymin, ymax = ax.get_ylim()
    text_y = ymin + (ymax - ymin) * 0.05
    ax.text(ts_train.iloc[len(ts_train)//2], text_y, 'TRAINING (80%)',
            ha='center', va='bottom', fontsize=10, fontweight='bold',
            color='#1565C0', alpha=0.9,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='none',
                      alpha=0.7))
    ax.text(ts_val.iloc[len(ts_val)//2], text_y, 'VAL (10%)',
            ha='center', va='bottom', fontsize=11, fontweight='bold',
            color='#E65100', alpha=0.95,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='none',
                      alpha=0.7))
    ax.text(ts_test.iloc[len(ts_test)//2], text_y, 'TEST (10%)',
            ha='center', va='bottom', fontsize=10, fontweight='bold',
            color='#212121', alpha=0.9,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='none',
                      alpha=0.7))
    ax.set_ylabel('Air Temperature (°C)', fontsize=12, fontweight='bold',
                  color='black')
    ax.set_title('Full Year Overview', fontsize=13, fontweight='bold',
                 color='black')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.tick_params(axis='x', rotation=30, labelsize=11, colors='black')
    ax.tick_params(axis='y', labelsize=11, colors='black')
    ax.legend(fontsize=10, loc='upper right', framealpha=1.0,
              edgecolor='black', prop={'weight': 'bold'})
    ax.grid(True, alpha=0.2)
    for spine in ax.spines.values():
        spine.set_edgecolor('black'); spine.set_linewidth(1.5)

    ax2 = axes[1]
    ax2.plot(ts_val, y_val, color='#FF9800', lw=1.2, alpha=0.9,
             label='Actual')
    ax2.plot(ts_val, y_val_pred, color='#FB8C00', lw=1.2, ls='--',
             alpha=0.85,
             label=f'Predicted   RMSE={val_rmse:.3f} °C  R2={val_r2:.4f}')
    ax2.fill_between(ts_val, y_val, y_val_pred,
                     alpha=0.15, color='#FF9800', label='Prediction error')
    ax2.set_ylabel('Air Temperature (°C)', fontsize=12, fontweight='bold',
                   color='black')
    ax2.set_title('Validation Period — Actual vs Predicted',
                  fontsize=13, fontweight='bold', color='black')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    ax2.tick_params(axis='x', rotation=30, labelsize=11, colors='black')
    ax2.tick_params(axis='y', labelsize=11, colors='black')
    ax2.legend(fontsize=10, loc='upper right', framealpha=1.0,
               edgecolor='black', prop={'weight': 'bold'})
    ax2.grid(True, alpha=0.2)
    for spine in ax2.spines.values():
        spine.set_edgecolor('black'); spine.set_linewidth(1.5)

    ax3 = axes[2]
    ax3.plot(ts_test, y_test, color='#212121', lw=1.2, alpha=0.9,
             label='Actual', zorder=5)
    ax3.plot(ts_test, y_pred, color='#E53935', lw=1.2, ls='--',
             alpha=0.85,
             label=f'Predicted   RMSE={results["rmse"]:.3f} °C  '
                   f'MAE={results["mae"]:.3f} °C  '
                   f'R2={results["r2"]:.4f}  '
                   f'Bias={results["bias"]:.3f} °C')
    ax3.fill_between(ts_test, y_test, y_pred,
                     alpha=0.15, color='#E53935', label='Prediction error')
    ax3.set_ylabel('Air Temperature (°C)', fontsize=12, fontweight='bold',
                   color='black')
    ax3.set_title('Test Period — Actual vs Predicted',
                  fontsize=13, fontweight='bold', color='black')
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
    ax3.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    ax3.tick_params(axis='x', rotation=30, labelsize=11, colors='black')
    ax3.tick_params(axis='y', labelsize=11, colors='black')
    ax3.legend(fontsize=10, loc='upper right', framealpha=1.0,
               edgecolor='black', prop={'weight': 'bold'})
    ax3.grid(True, alpha=0.2)
    for spine in ax3.spines.values():
        spine.set_edgecolor('black'); spine.set_linewidth(1.5)

    plt.tight_layout(pad=2.5, h_pad=3.0)
    path2 = out(FOLDER_RESULTS, f'full_dataset_split_detail_{tag}.png')
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Full dataset split detail -> {path2}")


# MULTIVARIATE FORECAST — all 4 variables
ALL_VARS = ['AirTC_18m', 'RH_18m', 'WS_ms_18m_Avg', 'WindDir_18m']

VAR_LABELS = {
    'AirTC_18m':     'Air Temperature (C)',
    'RH_18m':        'Relative Humidity (%)',
    'WS_ms_18m_Avg': 'Wind Speed (m/s)',
    'WindDir_18m':   'Wind Direction (deg)',
}

VAR_COLORS = {
    'AirTC_18m':     '#E53935',
    'RH_18m':        '#1565C0',
    'WS_ms_18m_Avg': '#2E7D32',
    'WindDir_18m':   '#F57C00',
}



def forecast_confidence_intervals(df_input, target='AirTC_18m',
                                   n_bootstrap=30, show_steps=48,
                                   tag='base'):
    """
    One-step-ahead forecast with bootstrap confidence intervals.
    At each step, all lag features come from actual observed data.
    No recursion, no future observations assumed.
    Shows predictions over a show_steps window from the test period.
    """
    global _BEST_PARAMS

    print(f"\n  [Forecast CI] One-step-ahead bootstrap forecast ({tag})...")

    model_params = dict(_BEST_PARAMS, random_state=42,
                        n_jobs=-1, verbosity=0) \
        if _BEST_PARAMS else dict(
            n_estimators=300, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, verbosity=0)

    df, FEATURES_ENC = build_features(df_input, target)
    train_df, val_df, test_df = _chrono_split(df)

    # use first show_steps rows of test set as the forecast window
    test_sorted  = test_df.sort_values('TIMESTAMP').reset_index(drop=True)
    show_df      = test_sorted.iloc[:show_steps]
    actual_vals  = show_df[target].values
    X_show       = show_df[FEATURES_ENC].values
    time_h       = np.arange(show_steps) * 0.5

    # bootstrap: train n_bootstrap models on resampled training data
    print(f"  [Forecast CI] Running {n_bootstrap} bootstrap samples...")
    all_boot_preds = []

    for boot_seed in range(n_bootstrap):
        boot_idx = np.random.choice(
            len(train_df), len(train_df), replace=True)
        df_boot  = train_df.iloc[boot_idx].sort_index()

        scaler_b  = StandardScaler()
        X_boot_s  = scaler_b.fit_transform(df_boot[FEATURES_ENC].values)
        m_boot    = XGBRegressor(**{**model_params,
                                    'random_state': boot_seed})
        m_boot.fit(X_boot_s, df_boot[target].values, verbose=False)

        # predict on the show window using actual observed lag features
        X_show_s   = scaler_b.transform(X_show)
        preds_boot = m_boot.predict(X_show_s)
        all_boot_preds.append(preds_boot)

        if (boot_seed + 1) % 10 == 0:
            print(f"    {boot_seed+1}/{n_bootstrap} done")

    boot_arr  = np.array(all_boot_preds)   # shape: (n_bootstrap, show_steps)
    pred_mean = np.mean(boot_arr, axis=0)
    pred_p5   = np.percentile(boot_arr,  5, axis=0)
    pred_p25  = np.percentile(boot_arr, 25, axis=0)
    pred_p75  = np.percentile(boot_arr, 75, axis=0)
    pred_p95  = np.percentile(boot_arr, 95, axis=0)

    # ── Plot 1: Forecast with CI bands 
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(time_h, actual_vals, color='black', lw=2.0,
            label='Actual', zorder=5)
    ax.plot(time_h, pred_mean, color='#E53935', lw=2.0, ls='--',
            label=f'Predicted (bootstrap mean, n={n_bootstrap})', zorder=4)
    ax.fill_between(time_h, pred_p5, pred_p95,
                    alpha=0.15, color='#E53935',
                    label='90% Confidence Interval')
    ax.fill_between(time_h, pred_p25, pred_p75,
                    alpha=0.30, color='#E53935',
                    label='50% Confidence Interval')
    ax.set_xlabel('Forecast Horizon (hours)', fontsize=13, fontweight='bold',
                  color='black')
    ax.set_ylabel('Air Temperature (°C)', fontsize=13, fontweight='bold',
                  color='black')
    ax.set_title(
        f'One-Step-Ahead Forecast with Bootstrap Confidence Intervals\n'
        f'(24-hour window from test period, {n_bootstrap} bootstrap samples)\n'
        f'All lag features from actual observed data — no recursion',
        fontsize=12, fontweight='bold', color='black')
    ax.legend(fontsize=11, framealpha=1.0, edgecolor='black',
              prop={'weight': 'bold'})
    ax.tick_params(labelsize=11, colors='black')
    ax.grid(True, alpha=0.25)
    for spine in ax.spines.values():
        spine.set_edgecolor('black'); spine.set_linewidth(1.5)
    plt.tight_layout(pad=2.0)
    path1 = out(FOLDER_EXTRA, f'forecast_confidence_intervals_{tag}.png')
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path1}")

    # Plot 2: CI width over the forecast window
    ci_width_90 = pred_p95 - pred_p5
    ci_width_50 = pred_p75 - pred_p25
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time_h, ci_width_90, color='#E53935', lw=2.5,
            marker='o', ms=4, label='90% CI width')
    ax.plot(time_h, ci_width_50, color='#1565C0', lw=2.5,
            marker='o', ms=4, label='50% CI width')
    ax.set_xlabel('Forecast Horizon (hours)', fontsize=13, fontweight='bold',
                  color='black')
    ax.set_ylabel('CI Width (°C)', fontsize=13, fontweight='bold',
                  color='black')
    ax.set_title(
        'Prediction Uncertainty — Bootstrap CI Width\n'
        '(One-step-ahead forecast, 24-hour test window)',
        fontsize=13, fontweight='bold', color='black')
    ax.legend(fontsize=12, framealpha=1.0, edgecolor='black',
              prop={'weight': 'bold'})
    ax.tick_params(labelsize=11, colors='black')
    ax.grid(True, alpha=0.25)
    for spine in ax.spines.values():
        spine.set_edgecolor('black'); spine.set_linewidth(1.5)
    plt.tight_layout(pad=2.0)
    path2 = out(FOLDER_EXTRA, f'forecast_ci_width_{tag}.png')
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path2}")
# STANDARD PLOT FUNCTIONS (unchanged)
def plot_results(results, title='XGBoost Prediction', out_path=None):
    if out_path is None:
        out_path = out(FOLDER_RESULTS, "results.png")
    y_test  = results['y_test']; y_pred = results['y_pred']
    ts_test = results['ts_test']
    mae, rmse, r2 = results['mae'], results['rmse'], results['r2']
    step = 6; total = len(y_test); chunk = total // 4

    fig, axes = plt.subplots(3, 2, figsize=(22, 20), facecolor="white")
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
    for i in range(4):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        s = i * chunk; e = (i + 1) * chunk if i < 3 else total
        ax.plot(ts_test.iloc[s:e:step], y_test[s:e:step],
                color="#1565C0", linewidth=1.4, alpha=0.9, label="Actual")
        ax.plot(ts_test.iloc[s:e:step], y_pred[s:e:step],
                color="#E53935", linewidth=1.4, alpha=0.85,
                linestyle="--", label="Predicted")
        ax.set_title(f"Part {i+1}: {ts_test.iloc[s].strftime('%b %Y')} to "
                     f"{ts_test.iloc[e-1].strftime('%b %Y')}",
                     fontsize=13, fontweight="bold", color="black")
        ax.set_xlabel("Date", fontsize=12, fontweight="bold", color="black")
        ax.set_ylabel("Air Temperature (°C)", fontsize=12, fontweight="bold",
                      color="black")
        ax.legend(loc="upper right", framealpha=1.0, fontsize=11,
                  edgecolor="black", prop={"weight": "bold"})
        ax.tick_params(axis="x", rotation=30, labelsize=10, colors="black")
        ax.tick_params(axis="y", labelsize=10, colors="black")
        ax.grid(True, alpha=0.25)
        for spine in ax.spines.values():
            spine.set_edgecolor("black"); spine.set_linewidth(1.5)

    ax_info = axes[2, 0]; ax_info.axis('off')
    mae_str  = (f"{mae:.3f} +/- {results['mae_std']:.3f}"
                if 'mae_std' in results else f"{mae:.3f}")
    rmse_str = (f"{rmse:.3f} +/- {results['rmse_std']:.3f}"
                if 'rmse_std' in results else f"{rmse:.3f}")
    r2_str   = (f"{r2:.4f} +/- {results['r2_std']:.4f}"
                if 'r2_std' in results else f"{r2:.4f}")
    bias_str = (f"{results['bias']:.3f} +/- {results['bias_std']:.3f}"
                if 'bias_std' in results else f"{results['bias']:.3f}")
    val_line = (f"\nVal RMSE = {results['val_rmse']:.3f}  "
                f"Val R2 = {results['val_r2']:.4f}"
                if 'val_rmse' in results else "")
    ax_info.text(0.5, 0.5,
                 f"MAE  = {mae_str} °C\n"
                 f"RMSE = {rmse_str} °C\n"
                 f"R2   = {r2_str}\n"
                 f"Bias = {bias_str} °C{val_line}",
                 transform=ax_info.transAxes, fontsize=13,
                 va="center", ha="center", fontfamily="monospace",
                 fontweight="bold", color="black",
                 bbox=dict(boxstyle="round,pad=1.0", fc="white",
                           ec="black", lw=1.5, alpha=1.0))

    ax_sc = axes[2, 1]
    ax_sc.scatter(y_test, y_pred, alpha=0.18, s=5,
                  color="#6A1B9A", rasterized=True)
    lims = [min(y_test.min(), y_pred.min()) - 1,
            max(y_test.max(), y_pred.max()) + 1]
    ax_sc.plot(lims, lims, color="#F57C00", linewidth=2.0,
               linestyle="--", label="Perfect fit")
    ax_sc.set_xlim(lims); ax_sc.set_ylim(lims)
    ax_sc.set_title(f"Actual vs Predicted  (R2 = {r2:.3f})",
                    fontsize=13, fontweight="bold", color="black")
    ax_sc.set_xlabel("Actual (°C)", fontsize=12, fontweight="bold",
                     color="black")
    ax_sc.set_ylabel("Predicted (°C)", fontsize=12, fontweight="bold",
                     color="black")
    ax_sc.legend(framealpha=1.0, fontsize=11, edgecolor="black",
                 prop={"weight": "bold"})
    ax_sc.tick_params(labelsize=10, colors="black")
    ax_sc.grid(True, alpha=0.25)
    for spine in ax_sc.spines.values():
        spine.set_edgecolor("black"); spine.set_linewidth(1.5)
    plt.tight_layout(rect=[0, 0, 1, 0.97], pad=2.5, h_pad=3.5, w_pad=2.5)
    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Plot -> {out_path}")


def plot_residuals(results, title_prefix, tag):
    residuals = results['residuals']; ts_test = results['ts_test']
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), constrained_layout=True)
    fig.suptitle(f"Residual Analysis — {title_prefix}",
                 fontsize=15, fontweight="bold", color="black")
    axes[0].hist(residuals, bins=60, color="#5C6BC0",
                 edgecolor="white", linewidth=0.5)
    axes[0].axvline(0, color="red", linewidth=2.0, linestyle="--",
                    label="Zero line")
    axes[0].set_title("Residual Distribution", fontsize=13,
                      fontweight="bold", color="black")
    axes[0].set_xlabel("Residual (°C)", fontsize=12, fontweight="bold",
                       color="black")
    axes[0].set_ylabel("Count", fontsize=12, fontweight="bold", color="black")
    axes[0].tick_params(labelsize=11, colors="black")
    mu, sigma = np.mean(residuals), np.std(residuals)
    axes[0].text(0.97, 0.95, f"mu = {mu:.3f}\nsigma = {sigma:.3f}",
                 transform=axes[0].transAxes, ha="right", va="top",
                 fontsize=11, fontfamily="monospace", fontweight="bold",
                 color="black",
                 bbox=dict(boxstyle="round", fc="white", ec="black",
                           lw=1.5, alpha=1.0))
    axes[0].legend(fontsize=11, prop={"weight": "bold"})
    for spine in axes[0].spines.values():
        spine.set_edgecolor("black"); spine.set_linewidth(1.5)

    step = max(1, len(ts_test) // 1000)
    axes[1].plot(ts_test.iloc[::step], residuals[::step],
                 color="#5C6BC0", linewidth=0.7, alpha=0.75)
    axes[1].axhline(0, color="red", linewidth=1.8, linestyle="--",
                    label="Zero line")
    axes[1].fill_between(ts_test.iloc[::step], residuals[::step], 0,
                         alpha=0.18, color="#5C6BC0")
    axes[1].set_title("Residuals Over Time", fontsize=13,
                      fontweight="bold", color="black")
    axes[1].set_xlabel("Date", fontsize=12, fontweight="bold", color="black")
    axes[1].set_ylabel("Residual (°C)", fontsize=12, fontweight="bold",
                       color="black")
    axes[1].tick_params(axis="x", rotation=30, labelsize=10, colors="black")
    axes[1].tick_params(axis="y", labelsize=11, colors="black")
    axes[1].legend(fontsize=11, prop={"weight": "bold"})
    axes[1].grid(True, alpha=0.2)
    for spine in axes[1].spines.values():
        spine.set_edgecolor("black"); spine.set_linewidth(1.5)
    plt.tight_layout(pad=2.5)
    path = out(FOLDER_RESULTS, f"residuals_{tag}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Residuals -> {path}")


def plot_feature_importance(results, title_prefix, tag):
    model = results['model']; features = results['features']
    imp   = model.feature_importances_
    sorted_idx = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(12, max(8, len(features) * 0.38)))
    ax.barh([features[i] for i in sorted_idx], imp[sorted_idx],
            color="#26A69A", edgecolor="white")
    ax.set_title(f"Feature Importances — {title_prefix}",
                 fontsize=14, fontweight="bold", color="black")
    ax.set_xlabel("Importance Score", fontsize=13, fontweight="bold",
                  color="black")
    ax.tick_params(labelsize=10, colors="black")
    ax.grid(True, axis="x", alpha=0.3)
    for spine in ax.spines.values():
        spine.set_edgecolor("black"); spine.set_linewidth(1.5)
    plt.tight_layout(pad=2.0)
    path = out(FOLDER_EXTRA, f"feature_importance_{tag}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Feature importance -> {path}")


def plot_error_by_hour(results, title_prefix, tag):
    ts_test = results['ts_test']; residuals = results['residuals']
    hours   = ts_test.dt.hour
    mae_by_hour = pd.Series(
        np.abs(residuals), index=hours).groupby(level=0).mean()
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(mae_by_hour.index, mae_by_hour.values,
           color="#EF5350", edgecolor="white")
    ax.set_title(f"MAE by Hour of Day — {title_prefix}",
                 fontsize=14, fontweight="bold", color="black")
    ax.set_xlabel("Hour of Day", fontsize=13, fontweight="bold", color="black")
    ax.set_ylabel("MAE (°C)", fontsize=13, fontweight="bold", color="black")
    ax.set_xticks(range(0, 24))
    ax.tick_params(labelsize=11, colors="black")
    ax.grid(True, axis="y", alpha=0.3)
    for spine in ax.spines.values():
        spine.set_edgecolor("black"); spine.set_linewidth(1.5)
    plt.tight_layout(pad=2.0)
    path = out(FOLDER_EXTRA, f"mae_by_hour_{tag}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  MAE by hour -> {path}")


def plot_boosting_curve(results, title_prefix, tag):
    evals = results.get('evals_result', {})
    if not evals:
        print("  [Boosting curve] No evals_result, skipping."); return
    keys      = list(evals.keys())
    train_key = keys[0] if len(keys) > 0 else None
    val_key   = keys[1] if len(keys) > 1 else None
    metric    = 'rmse'
    fig, ax   = plt.subplots(figsize=(14, 6))
    if train_key and metric in evals[train_key]:
        ax.plot(evals[train_key][metric], color='steelblue',
                lw=1.8, label='Train RMSE')
    if val_key and metric in evals[val_key]:
        val_scores = evals[val_key][metric]
        ax.plot(val_scores, color='#E53935', lw=1.8,
                label='Validation RMSE')
        best_iter  = int(np.argmin(val_scores))
        ax.axvline(best_iter, color='grey', lw=1.5, ls='--', alpha=0.8)
        # Place annotation well clear of the line
        y_pos = min(val_scores) + (max(val_scores) - min(val_scores)) * 0.6
        ax.text(best_iter + max(5, len(val_scores) * 0.02), y_pos,
                f'Best: tree {best_iter}\nRMSE = {val_scores[best_iter]:.3f}',
                fontsize=11, fontweight='bold', color='black', va='center',
                bbox=dict(boxstyle='round,pad=0.4', fc='white',
                          ec='black', lw=1.2, alpha=1.0))
    ax.set_title(f'Boosting Curve — {title_prefix}',
                 fontsize=14, fontweight='bold', color='black')
    ax.set_xlabel('Number of Trees', fontsize=13, fontweight='bold',
                  color='black')
    ax.set_ylabel('RMSE (°C)', fontsize=13, fontweight='bold', color='black')
    ax.tick_params(labelsize=11, colors='black')
    ax.legend(fontsize=12, prop={'weight': 'bold'}, framealpha=1.0,
              edgecolor='black')
    ax.grid(True, alpha=0.25)
    for spine in ax.spines.values():
        spine.set_edgecolor('black'); spine.set_linewidth(1.5)
    plt.tight_layout(pad=2.0)
    path = out(FOLDER_EXTRA, f'boosting_curve_{tag}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Boosting curve -> {path}")