"""
preview_dataset.py
------------------
Выводит первые 10 строк финального датасета (матрица признаков X + целевая y),
который подаётся в модели. Используй для скриншота в дипломной работе.

Запуск:
    python preview_dataset.py
или с другим путём к данным:
    python preview_dataset.py --data "путь/к/файлу.csv"
"""

import sys
import argparse
from pathlib import Path

import pandas as pd

# Подключаем корень проекта
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "src")

from src import config
from src.init_data import load_data
from src.features import build_feature_matrix, FEATURE_COLS


def preview_feature_matrix(data_path: str, cpc: float = 0.30):
    print(f"\n{'='*70}")
    print(f"  ПРЕДПРОСМОТР ФИНАЛЬНОГО ДАТАСЕТА")
    print(f"{'='*70}\n")

    df = load_data(data_path)

    X, _, _, _, _ = build_feature_matrix(df, cpc)
    print(X.head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(config.DATA_PATH), help="Путь к CSV-файлу")
    parser.add_argument("--cpc", default=0.30, type=float, help="CPC для расчёта рекламных признаков")
    args = parser.parse_args()

    preview_feature_matrix(args.data, cpc=args.cpc)


"""


  target_log(page_views) hour   day_of_week is_weekend month hour_sin hour_cos dow_sin dow_cos month_sin month_cos hour_x_dow is_holiday_season days_since_jan1  page_views_per_user ad_cost cpc_value adstock_cost ad_cost_lag_1
0 2.3979                 0      6           1          11    0.0000   1.0000   -0.7818 0.6235  -0.5000   0.8660    6          1                 0.8384           2.0000              0.0000  0.3000    0.0000       0.0000
1 1.0986                 0      6           1          11    0.0000   1.0000   -0.7818 0.6235  -0.5000   0.8660    6          1                 0.8384           2.0000              0.0000  0.3000    0.0000       0.0000
2 1.7918                 0      6           1          11    0.0000   1.0000   -0.7818 0.6235  -0.5000   0.8660    6          1                 0.8384           2.0000              0.0000  0.3000    0.0000       0.0000
3 2.0794                 0      6           1          11    0.0000   1.0000   -0.7818 0.6235  -0.5000   0.8660    6          1                 0.8384           2.0000              0.0000  0.3000    0.0000       0.0000
4 0.6931                 0      6           1          11    0.0000   1.0000   -0.7818 0.6235  -0.5000   0.8660    6          1                 0.8384           2.0000              0.0000  0.3000    0.0000       0.0000
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  ad_cost_lag_24 lag_1  lag_4  lag_8  lag_12 lag_24 lag_48 lag_168 rolling_mean_24 rolling_std_24 slope_24 rolling_mean_48 rolling_mean_168 seg_mean_users seg_std_users seg_ratio seg_lag_1 seg_lag_24 source medium device_type os
0 0.0000         6.4394 6.4378 6.4378 6.4378 6.4329 6.4167 6.4378  6.4377          0.0000         -0.0001  6.4356          6.3642           3.8957         0.0000        0.7701    6.0000    6.0000     0      0      0           3
1 0.0000         6.4394 6.4378 6.4378 6.4378 6.4329 6.4167 6.4378  6.4377          0.0000         -0.0001  6.4356          6.3642           3.8957         0.0000        0.2567    6.0000    6.0000     2      4      1           3
2 0.0000         6.4394 6.4378 6.4378 6.4378 6.4329 6.4167 6.4378  6.4377          0.0000         -0.0001  6.4356          6.3642           3.8957         0.0000        0.7701    6.0000    6.0000     0      0      1           3
3 0.0000         6.4394 6.4378 6.4378 6.4378 6.4329 6.4167 6.4378  6.4377          0.0000         -0.0001  6.4356          6.3642           3.8957         0.0000        0.2567    6.0000    6.0000     0      0      1           5
4 0.0000         6.4394 6.4378 6.4378 6.4378 6.4329 6.4167 6.4378  6.4377          0.0000         -0.0001  6.4356          6.3642           3.8957         0.0000        0.2567    6.0000    6.0000     1      1      0           0




"""
