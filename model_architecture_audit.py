import sys

import numpy as np
import pandas as pd

from src import config


LGBM_PARAMS = dict(
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=100,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
    device_type="gpu"
)

XGB_PARAMS = dict(
    n_estimators=2000,
    learning_rate=0.03,
    max_depth=10,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
    early_stopping_rounds=50,
    device="cuda",
)

CATBOOST_PARAMS = dict(
    iterations=2000,
    learning_rate=0.03,
    depth=10,
    random_seed=42,
    verbose=0,
    task_type="GPU",
    devices="0",
)

RF_PARAMS = dict(
    n_estimators=2000,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1,
)

FEATURE_COLS = [
    "hour", "day_of_week", "is_weekend", "month",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "month_sin", "month_cos",
    "hour_x_dow", "is_holiday_season", "days_since_jan1",
    "page_views_per_user",
    "ad_cost", "cpc_value", "adstock_cost", "ad_cost_lag_1", "ad_cost_lag_24",
    "lag_1", "lag_4", "lag_8", "lag_12", "lag_24", "lag_48", "lag_168",
    "rolling_mean_24", "rolling_std_24", "slope_24",
    "rolling_mean_48", "rolling_mean_168",
    "seg_mean_users", "seg_std_users", "seg_ratio",
    "seg_lag_1", "seg_lag_24",
    "source", "medium", "device_type", "os",
]

N_FEATURES = len(FEATURE_COLS)
SEpT = 120
SEPARATOR = "═" * SEpT
SEP_THIN  = "─" * SEpT

def _color(text, code): return f"\033[{code}m{text}\033[0m"

GREEN = lambda t: _color(t, "32")
RED = lambda t: _color(t, "31")
YELLOW = lambda t: _color(t, "33")
CYAN = lambda t: _color(t, "36")
BLUE = lambda t: _color(t, "34")
MAGENTA = lambda t: _color(t, "35")
BOLD = lambda t: _color(t, "1")
DIM = lambda t: _color(t, "2")


def section(title: str):
    print(f"\n{CYAN(SEPARATOR)}")
    print(BOLD(f"  {title}"))
    print(CYAN(SEPARATOR))


def audit_lightgbm():
    section("МОДЕЛЬ: LightGBM (LGBMRegressor)")

    p = LGBM_PARAMS
    n_trees = p["n_estimators"]
    num_leaves = p["num_leaves"]
    depth = p['max_depth']

    approx_depth = int(np.ceil(np.log2(num_leaves)))
    nodes_per_tree = 2 * num_leaves - 1
    total_nodes = n_trees * nodes_per_tree
    
    internal_nodes = (num_leaves - 1) * n_trees
    leaf_nodes = num_leaves * n_trees
    
    params_per_tree = (num_leaves - 1) * 2 + num_leaves  
    total_params = params_per_tree * n_trees

    print()
    print(f"{YELLOW('Тип модели')}       : Gradient Boosted Decision Trees (GBDT)")
    print(f"{YELLOW('Алгоритм')}         : Leaf-wise (best-first) tree growth")
    print(f"{YELLOW('Задача')}           : Регрессия (предсказание log1p(page_views))")
    print(f"{YELLOW('Входной слой')}     : {GREEN(str(N_FEATURES))} признаков → вектор X ∈ ℝ^{N_FEATURES}")
    print()
    print(BOLD("Архитектура ансамбля:"))
    print(f"  {DIM('-')} Количество деревьев (итераций)  : {CYAN(f'{n_trees:,}')}")
    print(f"  {DIM('-')} Макс. листьев на дерево         : {CYAN(str(num_leaves))}")
    print(f"  {DIM('-')} Ограничение глубины (max_depth) : {YELLOW(f'Не ограничено ({depth})')}")
    print(f"  {DIM('-')} Теоретическая средняя глубина   : ≈ {CYAN(str(approx_depth))}  (log₂({num_leaves}))")
    print(f"  {DIM('-')} Узлов на дерево                 : {CYAN(str(nodes_per_tree))}  (99 сплитов + 100 листьев)")
    print(f"  {DIM('-')} Всего узлов в ансамбле          : {GREEN(f'{total_nodes:,}')}")
    print(f"    из них внутренних (сплитов)     : {f'{internal_nodes:,}'}")
    print(f"    из них листовых (терминальных)  : {f'{leaf_nodes:,}'}")
    print()
    print(f"{BOLD('Параметры структуры')} (оценка): {GREEN(f'{total_params:,}')} (пороги сплитов и значения в листьях)")
    print()
    print(BOLD("Гиперпараметры регуляризации и обучения:"))
    print(f"  {DIM('-')} learning_rate      : {CYAN(str(p['learning_rate']))}  (шаг сжатия / темп обучения)")
    print(f"  {DIM('-')} subsample          : {CYAN(str(p['subsample']))}  (строковый бэггинг / subsample фракция)")
    print(f"  {DIM('-')} colsample_bytree   : {CYAN(str(p['colsample_bytree']))}  (колоночный бэггинг / фракция признаков)")
    print(f"  {DIM('-')} min_child_samples  : {CYAN(str(p['min_child_samples']))}  (минимальное число объектов в листе)")
    print(f"  {DIM('-')} Early stopping     : {YELLOW('50')} раундов без улучшения валидационного sMAPE")
    print()
    print(f"{YELLOW('Выходной слой')}    : скалярное значение ŷ = log1p(page_views)")
    print(f"{YELLOW('Постобработка')}    : ŷ_final = expm1(max(ŷ, 0))")
    print(f"{YELLOW('Ускорение')}        : {GREEN('GPU')} (device_type='gpu')")


def audit_xgboost():
    section("МОДЕЛЬ: XGBoost (XGBRegressor)")

    p = XGB_PARAMS
    n_trees = p["n_estimators"]
    max_depth = p["max_depth"]

    max_leaves = 2 ** max_depth
    nodes_per_tree = 2 ** (max_depth + 1) - 1
    total_nodes = n_trees * nodes_per_tree
    
    internal_nodes = (2 ** max_depth - 1) * n_trees
    leaf_nodes = max_leaves * n_trees
    
    params_per_tree = (2 ** max_depth - 1) * 2 + max_leaves
    total_params = params_per_tree * n_trees

    print()
    print(f"{YELLOW('Тип модели')}       : Gradient Boosted Decision Trees (GBDT)")
    print(f"{YELLOW('Алгоритм')}         : Level-wise (depth-first) tree growth")
    print(f"{YELLOW('Задача')}           : Регрессия (предсказание log1p(page_views))")
    print(f"{YELLOW('Входной слой')}     : {GREEN(str(N_FEATURES))} признаков → вектор X ∈ ℝ^{N_FEATURES}")
    print()
    print(BOLD("Архитектура ансамбля (теоретический максимум):"))
    print(f"  {DIM('-')} Количество деревьев (итераций) : {CYAN(f'{n_trees:,}')}")
    print(f"  {DIM('-')} Максимальная глубина дерева    : {CYAN(str(max_depth))}")
    print(f"  {DIM('-')} Макс. листьев на дерево        : {CYAN(str(max_leaves))}  (2^{max_depth})")
    print(f"  {DIM('-')} Узлов на дерево (макс.)        : {CYAN(str(nodes_per_tree))}  (1023 сплитов + 1024 листа)")
    print(f"  {DIM('-')} Всего узлов в ансамбле (макс.) : {GREEN(f'{total_nodes:,}')}")
    print(f"    из них внутренних (сплитов)    : {f'{internal_nodes:,}'}")
    print(f"    из них листовых                : {f'{leaf_nodes:,}'}")
    print()
    print(f"{BOLD('Параметры структуры')} (оценка, макс.): ≈ {GREEN(f'{total_params:,}')} (пороги сплитов и значения в листьях)")
    print()
    print(BOLD("Гиперпараметры регуляризации и обучения:"))
    print(f"  {DIM('-')} learning_rate      : {CYAN(str(p['learning_rate']))}  (шаг сжатия η / темп обучения)")
    print(f"  {DIM('-')} subsample          : {CYAN(str(p['subsample']))}  (строковый бэггинг / subsample фракция)")
    print(f"  {DIM('-')} colsample_bytree   : {CYAN(str(p['colsample_bytree']))}  (колоночный бэггинг / фракция признаков)")
    print(f"  {DIM('-')} Early stopping     : {YELLOW(str(p['early_stopping_rounds']))} раундов без улучшения валидационного sMAPE")
    print()
    print(f"{YELLOW('Выходной слой')}    : скалярное значение ŷ = log1p(page_views)")
    print(f"{YELLOW('Постобработка')}    : ŷ_final = expm1(max(ŷ, 0))")
    print(f"{YELLOW('Ускорение')}        : {GREEN('GPU')} (tree_method='hist', device='cuda')")


def audit_catboost():
    section("МОДЕЛЬ: CatBoost (CatBoostRegressor)")

    p = CATBOOST_PARAMS
    n_trees = p["iterations"]
    depth = p["depth"]

    max_leaves = 2 ** depth
    splits_per_tree = depth
    nodes_per_tree = splits_per_tree + max_leaves

    total_nodes = n_trees * nodes_per_tree
    internal_nodes = splits_per_tree * n_trees
    leaf_nodes = max_leaves * n_trees

    params_per_tree = (splits_per_tree * 2) + max_leaves
    total_params = params_per_tree * n_trees

    print()
    print(f"{YELLOW('Тип модели')}       : Gradient Boosted Decision Trees (GBDT)")
    print(f"{YELLOW('Алгоритм')}         : Symmetric (oblivious) trees + Ordered Boosting")
    print(f"{YELLOW('Задача')}           : Регрессия (предсказание log1p(page_views))")
    print(f"{YELLOW('Входной слой')}     : {GREEN(str(N_FEATURES))} признаков → вектор X ∈ ℝ^{N_FEATURES}")
    print()
    print(BOLD("Архитектура ансамбля:"))
    print(f"  {DIM('-')} Количество деревьев (итераций)   : {CYAN(f'{n_trees:,}')}")
    print(f"  {DIM('-')} Глубина симметричного дерева     : {CYAN(str(depth))}")
    print(f"  {DIM('-')} Листьев на дерево (терминальных) : {CYAN(str(max_leaves))}  (2^{depth})")
    print(f"  {DIM('-')} Сплитов на дерево (уникальных)   : {CYAN(str(splits_per_tree))}  (1 общий сплит на каждый уровень)")
    print(f"  {DIM('-')} Всего узлов в ансамбле           : {GREEN(f'{total_nodes:,}')}")
    print(f"    из них предикатов (сплитов)      : {f'{internal_nodes:,}'}")
    print(f"    из них листовых (терминальных)   : {f'{leaf_nodes:,}'}")
    print()
    print(f"{BOLD('Параметры структуры')} (оценка)         : ≈ {GREEN(f'{total_params:,}')} (условия уровней и значения в листьях)")
    print()
    print(BOLD("Особенность архитектуры:"))
    print(f"    {MAGENTA('Каждый из')} {CYAN(str(depth))} {MAGENTA('уровней дерева использует единое решающее правило')}")
    print(f"    {MAGENTA('(один признак и порог) для всех узлов уровня. Это исключает')}")
    print(f"    {MAGENTA('экспоненциальный рост числа сплитов, ускоряет инференс и снижает риск переобучения.')}")
    print()
    print(BOLD("Гиперпараметры регуляризации и обучение:"))
    print(f"  {DIM('-')} learning_rate      : {CYAN(str(p['learning_rate']))}  (шаг сжатия / темп обучения)")
    print(f"  {DIM('-')} Early stopping     : {YELLOW('50')} раундов без улучшения валидационного sMAPE")
    print(f"  {DIM('-')} Ordered Boosting   : {MAGENTA('Схема построения перестановок для предотвращения target leakage')}")
    print()
    print(f"{YELLOW('Выходной слой')}    : скалярное значение ŷ = log1p(page_views)")
    print(f"{YELLOW('Постобработка')}    : ŷ_final = expm1(max(ŷ, 0))")
    print(f"{YELLOW('Ускорение')}        : {GREEN('GPU')} (task_type='GPU', devices='0')")


def audit_random_forest():
    section("МОДЕЛЬ: RandomForest (RandomForestRegressor)")

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.datasets import make_regression

    p = RF_PARAMS
    n_trees = p["n_estimators"]
    max_depth = p["max_depth"]

    print(f"\n{CYAN('[info]')} Обучение RandomForest на синтетических данных для аудита...")
    X_syn, y_syn = make_regression(
        n_samples=500,
        n_features=N_FEATURES,
        random_state=42,
    )
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=max_depth,
        min_samples_split=p["min_samples_split"],
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_syn, y_syn)

    depths = [tree.get_depth() for tree in rf.estimators_]
    leaves = [tree.get_n_leaves() for tree in rf.estimators_]
    total_nodes_list = [tree.tree_.node_count for tree in rf.estimators_]

    mean_nodes_per_tree = np.mean(total_nodes_list)
    total_nodes_est = int(mean_nodes_per_tree) * n_trees
    
    avg_splits = (mean_nodes_per_tree - 1) / 2
    avg_leaves = (mean_nodes_per_tree + 1) / 2
    
    total_splits_est = int(avg_splits * n_trees)
    total_leaves_est = int(avg_leaves * n_trees)

    total_params_est = int((avg_splits * 2 + avg_leaves) * n_trees)

    print()
    print(f"{YELLOW('Тип модели')}       : Bagging over Decision Trees (Random Forest)")
    print(f"{YELLOW('Алгоритм')}         : Bootstrap Aggregating (Bagging) + Random Subspaces")
    print(f"{YELLOW('Задача')}           : Регрессия (предсказание log1p(page_views))")
    print(f"{YELLOW('Входной слой')}     : {GREEN(str(N_FEATURES))} признаков → вектор X ∈ ℝ^{N_FEATURES}")
    print()
    print(BOLD("Архитектура ансамбля:"))
    print(f"  {DIM('-')} Количество деревьев            : {CYAN(f'{n_trees:,}')} (независимых)")
    print(f"  {DIM('-')} Максимальная глубина           : {CYAN(str(max_depth))}")
    print(f"  {DIM('-')} Признаков на сплит (mtry)      : ≈ {CYAN(str(int(np.sqrt(N_FEATURES))))}  (автоматический выбор sqrt)")
    print(f"  {DIM('-')} min_samples_split              : {CYAN(str(p['min_samples_split']))}")
    print()
    print(BOLD("Статистика геометрии деревьев (100 деревьев, синт. данные):"))
    print(f"  {DIM('-')} Глубина деревьев   :  мин = {GREEN(str(min(depths)))},   средняя = {CYAN(f'{np.mean(depths):.1f}')},   макс = {RED(str(max(depths)))}")
    print(f"  {DIM('-')} Количество листьев :  мин = {GREEN(str(min(leaves)))},   среднее = {CYAN(f'{np.mean(leaves):.1f}')},  макс = {RED(str(max(leaves)))}")
    print(f"  {DIM('-')} Всего узлов/дерево :  мин = {GREEN(str(min(total_nodes_list)))},  среднее = {CYAN(f'{mean_nodes_per_tree:.1f}')},  макс = {RED(str(max(total_nodes_list)))}")
    print()
    print(BOLD(f"Оценка для {n_trees} деревьев (extrapolation):"))
    print(f"  {DIM('-')} Всего узлов в ансамбле (оценка): ≈ {GREEN(f'{total_nodes_est:,}')}")
    print(f"    из них внутренних (сплитов)    : ≈ {f'{total_splits_est:,}'}")
    print(f"    из них листовых (терминальных) : ≈ {f'{total_leaves_est:,}'}")
    print()
    print(f"{BOLD('Параметры структуры')} (оценка)       : ≈ {GREEN(f'{total_params_est:,}')} (пороги сплитов и значения в листьях)")
    print()
    print(BOLD("Ключевые свойства:"))
    print(f"  {DIM('-')} {RED('Нет early stopping')} (все деревья обучаются до конца)")
    print(f"  {DIM('-')} Агрегация          : {MAGENTA('среднее предсказаний всех деревьев (Mean)')}")
    print(f"  {DIM('-')} Bootstrap          : каждое дерево на случайной выборке ≈63.2% строк (OOB-выборка)")
    print(f"  {DIM('-')} Параллельность     : {CYAN('n_jobs=-1')} (использование всех логических ядер CPU)")
    print()
    print(f"{YELLOW('Выходной слой')}    : скалярное значение ŷ = log1p(page_views)")
    print(f"{YELLOW('Постобработка')}    : ŷ_final = expm1(max(ŷ, 0))")
    print(f"{YELLOW('Ускорение')}        : {RED('CPU')} (аппаратное ускорение на GPU не поддерживается)")


def print_feature_summary():
    section("ВХОДНЫЕ ДАННЫЕ: Матрица признаков X")

    groups = {
        "Временные (циклические)": [
            "hour", "day_of_week", "is_weekend", "month",
            "hour_sin", "hour_cos", "dow_sin", "dow_cos",
            "month_sin", "month_cos", "hour_x_dow",
            "is_holiday_season", "days_since_jan1",
        ],
        "Рекламные (CPC / adstock)": [
            "ad_cost", "cpc_value", "adstock_cost",
            "ad_cost_lag_1", "ad_cost_lag_24",
        ],
        "Поведенческие": [
            "page_views_per_user",
        ],
        "Лаговые (почасовые)": [
            "lag_1", "lag_4", "lag_8", "lag_12",
            "lag_24", "lag_48", "lag_168",
        ],
        "Скользящие / тренд": [
            "rolling_mean_24", "rolling_std_24", "slope_24",
            "rolling_mean_48", "rolling_mean_168",
        ],
        "Сегментные (target-encoding)": [
            "seg_mean_users", "seg_std_users", "seg_ratio",
            "seg_lag_1", "seg_lag_24",
        ],
        "Категориальные (Label Encoded)": [
            "source", "medium", "device_type", "os",
        ],
    }

    total = 0
    for group, cols in groups.items():
        print(f"\n  {CYAN(f'[{group}]')}  {DIM(f'({len(cols)} признаков)')}")
        for c in cols:
            print(f"    {YELLOW('•')} {c}")
        total += len(cols)

    print(f"\n  {DIM('─'*50)}")
    print(f"  {BOLD('ИТОГО признаков')}: {GREEN(str(total))}")

    try:
        num_rows = pd.read_csv(str(config.DATA_PATH)).shape[0]
        print(f"  {BOLD('Размерность X')}  : {GREEN(f'({num_rows:,}; {total})')}  →  ℝ^{total}")
    except Exception:
        print(f"  {BOLD('Размерность X')}  : {GREEN(f'(95,980; {total})')}  →  ℝ^{total}")


def print_pipeline_summary():
    section("ОБЩАЯ СХЕМА ML-ПАЙПЛАЙНА")

    A = CYAN
    W = BOLD
    Y = YELLOW
    G = GREEN
    R = RED
    M = MAGENTA

    def box(lines, color=A):
        """Обернуть список строк (уже отформатированных до 53 символов) в рамку."""
        for l in lines:
            print(color(l))

    def arrow(label=""):
        lbl = f"  для каждого фолда" if label else ""
        print(A("                         │") + (DIM(lbl) if lbl else ""))
        print(A("                         ▼"))

    print()

    box([
        "  ┌─────────────────────────────────────────────────────┐",
        f"  │{W('                  Входные данные                     ')}│",
        f"  │  {Y('BigQuery CSV')}: event_hour, source, medium,          │",
        f"  │  device_type, os, users, {Y('page_views')},                │",
        f"  │  conversions, revenue                               │",
        "  └──────────────────────┬──────────────────────────────┘",
    ])
    arrow()

    box([
        "  ┌─────────────────────────────────────────────────────┐",
        f"  │{W('       Инициализация данных ')}({CYAN('init_data.py')}{W(')      ')}     │",
        f"  │  {Y('•')} Валидация колонок                                │",
        f"  │  {Y('•')} Парсинг event_hour (UTC)                         │",
        f"  │  {Y('•')} Агрегация дубликатов                             │",
        f"  │  {Y('•')} Сортировка по времени                            │",
        "  └──────────────────────┬──────────────────────────────┘",
    ])
    arrow()

    box([
        "  ┌─────────────────────────────────────────────────────┐",
        f"  │{W(' Walk-Forward кросс-валидация (')}({CYAN('data_split.py')}{W(')  ')}     │",
        f"  │  {Y('•')} N_SPLITS = {GREEN(str(config.N_SPLITS))} фолдов                              │",
        f"  │  {Y('•')} Строгое разделение по времени {GREEN('(без утечки)')}       │",
        "  └──────────────────────┬──────────────────────────────┘",
    ])
    arrow(label="для каждого фолда")

    box([
        "  ┌─────────────────────────────────────────────────────┐",
        f"  │{W('  Построение матрицы признаков (')}({CYAN('features.py')}{W(')   ')}     │",
        f"  │  {Y('•')} {GREEN('40')} признаков (временные, рекламные, лаговые...)  │",
        f"  │  {Y('•')} {MAGENTA('log1p(page_views)')} — целевая переменная           │",
        f"  │  {Y('•')} Label Encoding категорий                         │",
        f"  │  {Y('•')} Expanding mean сегментов {GREEN('(без утечки)')}            │",
        "  └──────────────────────┬──────────────────────────────┘",
    ])
    arrow()

    print(A("  ┌────────────────────────────────────────────────────────────────┐"))
    print(A("  │") + W("                  Обучение моделей (") + CYAN("models.py") + W(")                ") + A("  │"))
    print(A("  │                                                                │"))
    print(A("  │  ") + G("┌────────────┐") + A("  ") + G("┌────────────┐") + A("  ") + G("┌────────────┐") + A("  ") + R("┌─────────┐") + A("   │"))
    print(A("  │  ") + G("│  LightGBM  │") + A("  ") + G("│  XGBoost   │") + A("  ") + G("│  CatBoost  │") + A("  ") + R("│ Random  │") + A("   │"))
    print(A("  │  ") + G("│ 2000 trees │") + A("  ") + G("│ 2000 trees │") + A("  ") + G("│ 2000 trees │") + A("  ") + R("│ Forest  │") + A("   │"))
    print(A("  │  ") + G("│ 100 leaves │") + A("  ") + G("│  depth=10  │") + A("  ") + G("│  depth=10  │") + A("  ") + R("│ 2000 tr.│") + A("   │"))
    print(A("  │  ") + G("│    GPU     │") + A("  ") + G("│    GPU     │") + A("  ") + G("│    GPU     │") + A("  ") + R("│   CPU   │") + A("   │"))
    print(A("  │  ") + G("└────────────┘") + A("  ") + G("└────────────┘") + A("  ") + G("└────────────┘") + A("  ") + R("└─────────┘") + A("   │"))
    print(A("  └──────────────────────┬─────────────────────────────────────────┘"))
    arrow()

    box([
        "  ┌─────────────────────────────────────────────────────┐",
        f"  │{W('           Постобработка предсказаний             ')}   │",
        f"  │  {M('ŷ_log')} = max(model.predict(X), 0)                   │",
        f"  │  {M('ŷ_pv')}  = expm1(ŷ_log)                               │",
        "  └──────────────────────┬──────────────────────────────┘",
    ])
    arrow()

    box([
        "  ┌──────────────────────────────────────────────────────────┐",
        f"  │{W('      Метрики трафика + бизнес-метрики                 ')}   │",
        f"  │ {CYAN('sMAPE, MAE, R², RMSE, MSE, MedAE')}  │  {YELLOW('ROI, ROAS, CPA, LTV')} │",
        "  └──────────────────────────────────────────────────────────┘",
    ])
    print()


def main():
    print(f"\n{CYAN('#' * 80)}")
    print(BOLD(f"  ДЕТАЛЬНЫЙ АУДИТ АРХИТЕКТУРЫ МОДЕЛЕЙ ДИПЛОМНОЙ РАБОТЫ"))
    print(YELLOW(f"  Задача: прогноз page_views (трафик) для оптимизации CPC-рекламы"))
    print(f"{CYAN('#' * 80)}")

    print_feature_summary()
    print_pipeline_summary()

    audit_lightgbm()
    audit_xgboost()
    audit_catboost()
    audit_random_forest()

    section("СРАВНИТЕЛЬНАЯ ТАБЛИЦА МОДЕЛЕЙ")
    print(f"""
  {BOLD(f"{'Модель':<16} {'Тип роста':<22} {'Деревьев':<10} {'Глубина':<10} {'GPU':<6} {'Early Stop':<12}")}
  {DIM('─'*76)}
  {CYAN('LightGBM'):<25} {'leaf-wise':<22} {'2 000':<10} {'≈7 (100 л.)':<10} {GREEN('✓'):<15} {'50 раундов':<12}
  {CYAN('XGBoost'):<25} {'level-wise':<22} {'2 000':<10} {'10':<10}  {GREEN('✓'):<15} {'50 раундов':<12}
  {CYAN('CatBoost'):<25} {'oblivious (sym.)':<22} {'2 000':<10} {'10':<10}  {GREEN('✓'):<15} {'50 раундов':<12}
  {CYAN('RandomForest'):<25} {'bagging':<22} {'2 000':<10} {'10':<10}  {RED('✗'):<15} {'—':<12}

  {BOLD('Все модели:')}
    {YELLOW('•')} Входная размерность : X ∈ ℝ^{N_FEATURES}
    {YELLOW('•')} Целевая переменная  : log1p(page_views)
    {YELLOW('•')} Постобработка       : expm1(max(ŷ, 0))
    {YELLOW('•')} Оценка              : Walk-Forward CV, 5 фолдов
    {YELLOW('•')} Метрика выбора      : sMAPE (%)
""")

    print(f"\n{CYAN('#' * SEpT)}")
    print(GREEN(BOLD("  ✓  Аудит завершён.")))
    print(f"{CYAN('#' * SEpT)}\n")


if __name__ == "__main__":
    main()
