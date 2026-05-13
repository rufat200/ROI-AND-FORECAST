import numpy as np
import pandas as pd

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


def get_models() -> dict:
    xgb_p = dict(**config.XGB_PARAMS)
    cat_p = dict(**config.CATBOOST_PARAMS)
    lgb_p = dict(**config.LGBM_PARAMS)

    if config.USE_GPU:
        xgb_p["device"] = "cuda"

        lgb_p["device_type"] = "gpu"

        cat_p["task_type"] = "GPU"
        cat_p["devices"] = "0"

    return {
        "XGBoost"     : XGBRegressor(**xgb_p),
        "CatBoost"    : CatBoostRegressor(**cat_p),
        "LightGBM"    : LGBMRegressor(**lgb_p),
        # "RandomForest": RandomForestRegressor(**config.RF_PARAMS)
    }


def compute_traffic_metrics(
    y_true: np.ndarray | pd.Series, 
    y_pred: np.ndarray | pd.Series,
) -> dict:
    """
    Метрики
    -------
    sMAPE : Symmetric Mean Absolute Percentage Error
    MAE   : Mean Absolute Error
    R²    : Coefficient of Determination
    MSE   : Mean Squared Error
    RMSE  : Root Mean Squared Error
    MedAE : Median Absolute Error
    Bias  : Mean signed error  (pred – true) # Среднее смещение
    MASE  : Mean Absolute Scaled Error  (naive baseline = lag-1)
    """

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) == 0:
        return {k: np.nan for k in
                ["sMAPE (%)", "MAE", "R²", "MSE", "RMSE", "MedAE", "Bias", "MASE"]}

    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape = np.mean(np.where(denom == 0, 0.0, np.abs(y_true - y_pred) / denom)) * 100
    
    naive_mae = mean_absolute_error(y_true[1:], y_true[:-1]) if len(y_true) > 1 else 1e-9
    mase = mean_absolute_error(y_true, y_pred) / max(naive_mae, 1e-9)

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
    y_true: np.ndarray,
    y_pred: np.ndarray,
    conversions: np.ndarray,
    revenue: np.ndarray,
    cpc: float,
    users: np.ndarray,
    is_paid_mask: np.ndarray,
    conv_rate: float,
) -> dict:
    
    total_users = float(np.sum(users)) or 1e-9
    total_conv = float(np.sum(conversions))
    total_rev = float(np.sum(revenue))
    paid_users = float(np.sum(users[is_paid_mask]))

    ad_spend = cpc * paid_users

    pred_conversions = y_pred * conv_rate
    pred_revenue = pred_conversions * config.AVG_ORDER_VALUE

    paid_pred = float(np.sum(y_pred[is_paid_mask]))
    organic_pred = float(np.sum(y_pred[~is_paid_mask]))
    delta_users = paid_pred - organic_pred

    delta_cost = ad_spend if ad_spend > 0 else 1e-9
    elasticity = delta_users / delta_cost

    roi = ((pred_revenue.sum() - ad_spend) / ad_spend * 100) if ad_spend > 0 else 0.0
    roas = (total_rev / ad_spend) if ad_spend > 0 else 0.0
    cpa = (ad_spend / total_conv) if total_conv > 0 else 0.0

    churn = config.MONTHLY_CHURN_RATE
    ltv = config.AVG_ORDER_VALUE * (1.0 / churn) * config.GROSS_MARGIN
    retention = (1 - churn) * 100
    cr = total_conv / total_users * 100
    ctr = config.CTR_DEFAULT * 100

    pred_total_users = float(np.sum(y_pred)) or 1e-9
    predicted_cr = pred_conversions.sum() / pred_total_users * 100

    return {
        "Elasticity"       : round(elasticity,         4),
        "Ad Spend ($)"     : round(ad_spend,           2),
        "Revenue ($)"      : round(total_rev,          2),
        "Pred Revenue ($)" : round(pred_revenue.sum(), 2),
        "ROI (%)"          : round(roi,                2),
        "ROAS"             : round(roas,               4),
        "CPA ($)"          : round(cpa,                4),
        "CAC ($)"          : round(cpa,                4),
        "LTV/CLV ($)"      : round(ltv,                2),
        "Churn Rate (%)"   : round(churn * 100,        2),
        "Retention (%)"    : round(retention,          2),
        "CR (%)"           : round(cr,                 4),
        "Pred CR (%)"      : round(predicted_cr,       4),
        "CTR (%)"          : round(ctr,                2),
    }
