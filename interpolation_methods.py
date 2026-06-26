import pandas as pd
import numpy as np

def linear_inter(df, col):
    df_linear = df.copy()
    df_linear[col] = df_linear[col].interpolate(method='linear')
    df_linear[col] = df_linear[col].ffill().bfill()
    return df_linear

def spline_inter(df, col):
    df_spline = df.copy()
    df_spline[col] = df_spline[col].interpolate(method='spline', order=3)
    df_spline[col] = df_spline[col].ffill().bfill()
    return df_spline

def diag_climatology_practical(df_corr):
    df_diag = df_corr.copy()

    for col in df_diag.columns:

        missing_mask = df_diag[col].isna()
        if not missing_mask.any():
            continue
        available_col = df_corr[col].dropna()
        lookup = (
            available_col
            .groupby([
                available_col.index.month,
                available_col.index.time
            ])
            .mean()
        )

        lookup.index.names = ['month', 'time']

        missing_idx = df_diag.index[missing_mask]

        keys = list(zip(
            missing_idx.month,
            missing_idx.time
        ))

        fills = pd.Series(
            [lookup.get(k, np.nan) for k in keys],
            index=missing_idx
        )

        df_diag.loc[missing_mask, col] = fills

    df_diag = df_diag.ffill().bfill()

    return df_diag

def locf_interpolation(df_corr, col):
    df_locf = df_corr.copy()
    df_locf[col] = df_locf[col].ffill().bfill()
    return df_locf
