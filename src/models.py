import numpy as np


from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor


from sklearn.metrics import (
        mean_absolute_error,
        root_mean_squared_error,
        r2_score,
        median_absolute_error
    )
import config


def get_models():
    """Возвращает словарь с «чистыми» моделями."""
    return {
        "XGBoost"     : XGBRegressor(**config.XGB_PARAMS),
        "CatBoost"    : CatBoostRegressor(**config.CATBOOST_PARAMS),
        "LightGBM"    : LGBMRegressor(**config.LGBM_PARAMS),
        "RandomForest": RandomForestRegressor(**config.RF_PARAMS)
    }

def compute_traffic_metrics(y_true, y_pred):
    """
    Метрики
    -------
    sMAPE : Symmetric Mean Absolute Percentage Error
    MAE   : Mean Absolute Error
    R²    : Coefficient of Determination
    MSE   : Mean Squared Error
    RMSE  : Root Mean Squared Error
    MedAE : Median Absolute Error
    Bias  : Mean signed error  (pred – true)
    MASE  : Mean Absolute Scaled Error  (naive baseline = lag-1)
    """

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape = np.mean(np.where(denom == 0, 0, np.abs(y_true - y_pred) / denom)) * 100
    

    if len(y_true) < 2:
        mase = np.nan
    else:
        naive_mae = mean_absolute_error(y_true[1:], y_true[:-1]) or 1e-9
        mase = mean_absolute_error(y_true, y_pred) / naive_mae

    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mse = rmse ** 2
    r2 = r2_score(y_true, y_pred)
    med = median_absolute_error(y_true, y_pred)
    bias = float(np.mean(y_pred - y_true))

    return {
        "sMAPE (%)": round(smape, 4),
        "MAE"      : round(mae,   4),
        "R²"       : round(r2,    4),
        "MSE"      : round(mse,   4),
        "RMSE"     : round(rmse,  4),
        "MedAE"    : round(med,   4),
        "Bias"     : round(bias,  4),
        "MASE"     : round(mase,  4),
    }

def compute_business_metrics(
    y_true, y_pred,
    conversions, revenue,
    cpc, users,
    is_paid_mask,
    conv_rate
):
    total_users = float(np.sum(users)) or 1e-9
    total_conv = conversions.sum()
    total_rev = revenue.sum()

    ad_spend = cpc * users[is_paid_mask].sum()
    pred_conversions = y_pred * conv_rate
    pred_revenue = pred_conversions * config.AVG_ORDER_VALUE

    roi = ((pred_revenue.sum() - ad_spend) / ad_spend * 100) if ad_spend > 0 else 0
    roas = (total_rev / ad_spend) if ad_spend > 0 else 0
    cpa = (ad_spend / total_conv) if total_conv > 0 else 0

    churn = config.MONTHLY_CHURN_RATE
    ltv = config.AVG_ORDER_VALUE * (1 / churn) * config.GROSS_MARGIN
    retention = (1 - churn) * 100
    cr = total_conv / total_users * 100
    ctr = config.CTR_DEFAULT * 100

    return {
        "Ad Spend ($)": round(ad_spend, 2),
        "Revenue ($)": round(total_rev, 2),
        "ROI (%)": round(roi, 2),
        "ROAS": round(roas, 4),
        "CPA ($)": round(cpa, 4),
        "CAC ($)": round(cpa, 4),
        "LTV/CLV ($)": round(ltv, 2),
        "Churn Rate (%)": round(churn * 100, 2),
        "Retention (%)": round(retention, 2),
        "CR (%)": round(cr, 4),
        "CTR (%)": round(ctr, 2),
    }
