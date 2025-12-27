"""
Основной файл с решением соревнования
Здесь должен быть весь ваш код для создания предсказаний
"""

            
def create_submission(predictions):
    """
    Пропишите здесь создание файла submission.csv в папку results
    !!! ВНИМАНИЕ !!! ФАЙЛ должен иметь именно такого названия
    """

    # Создать пандас таблицу submission

    import os
    import pandas as pd
    os.makedirs('results', exist_ok=True)
    
    submission_path = 'results/submission.csv'

    submission = predictions
    submission.to_csv(submission_path, index=False)
    
    print(f"Submission файл сохранен: {submission_path}")
    
    return submission_path
    
def main():
    """
    Главная функция программы
    
    Вы можете изменять эту функцию под свои нужды,
    но обязательно вызовите create_submission() в конце!
    """
    print("=" * 50)
    print("Запуск решения соревнования")
    print("=" * 50)

    import numpy as np
    import pandas as pd
    import warnings
    warnings.filterwarnings('ignore')
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split, TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.metrics import mean_absolute_error
    from tslearn.clustering import TimeSeriesKMeans
    
    import lightgbm as lgb
    import xgboost as xgb
    import catboost as cb
    
    import holidays
    from datetime import datetime, timedelta
    import random
    import os
    from scipy import sparse
    
    # Фиксируем сиды
    SEED = 322
    np.random.seed(SEED)
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)

    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    print("\nTrain columns:")
    print(train.columns.tolist())

    train["dt"] = pd.to_datetime(train["dt"])
    test["dt"]  = pd.to_datetime(test["dt"])
    train = train.sort_values(["product_id", "dt"]).reset_index(drop=True)
    test  = test.sort_values(["product_id", "dt"]).reset_index(drop=True)
    
    train["price_width"]  = train["price_p95"] - train["price_p05"]
    train["price_center"] = 0.5 * (train["price_p95"] + train["price_p05"])
    train["log_width"]    = np.log1p(train["price_width"].clip(lower=0))
    
    global_min_dt = min(train["dt"].min(), test["dt"].min())
    train["day_idx"] = (train["dt"] - global_min_dt).dt.days.astype(int)
    test["day_idx"]  = (test["dt"]  - global_min_dt).dt.days.astype(int)

    train.head()
    test.head()

    print("=== TRAIN INFO ===")
    train.info()
    print("\n=== TEST INFO ===")
    test.info()
    
    def summarize_columns(df, name="DF"):
        summary = []
        n = len(df)
        for col in df.columns:
            s = df[col]
            summary.append({
                "column": col,
                "dtype": s.dtype,
                "n_missing": s.isna().sum(),
                "missing_ratio": s.isna().mean(),
                "n_unique": s.nunique(),
                "sample_values": s.dropna().unique()[:5]
            })
        summary_df = pd.DataFrame(summary)
        summary_df = summary_df.sort_values(["dtype", "n_unique"])
        print(f"\n=== SUMMARY for {name} ===")
        return summary_df
    
    summary_train = summarize_columns(train, "TRAIN")
    summary_test = summarize_columns(test, "TEST")
    
    summary_train
    summary_test

    if "dt" in train.columns:
        if not np.issubdtype(train["dt"].dtype, np.datetime64):
            train["dt"] = pd.to_datetime(train["dt"])
        if not np.issubdtype(test["dt"].dtype, np.datetime64):
            test["dt"] = pd.to_datetime(test["dt"])
    
        print("Train dt range:", train["dt"].min(), "→", train["dt"].max())
        print("Test dt range :", test["dt"].min(), "→", test["dt"].max())
    
        plt.figure(figsize=(12, 4))
        train["dt"].value_counts().sort_index().plot()
        plt.title("Train: количество наблюдений по датам")
        plt.xlabel("Дата")
        plt.ylabel("Число строк")
        plt.show()
    
        plt.figure(figsize=(12, 4))
        test["dt"].value_counts().sort_index().plot()
        plt.title("Test: количество наблюдений по датам")
        plt.xlabel("Дата")
        plt.ylabel("Число строк")
        plt.show()
    else:
        print("Колонка 'dt' в train/test не найдена — временной анализ пропущен.")

    target_cols = [c for c in ["price_p05", "price_p95"] if c in train.columns]

    if target_cols:
        print("=== Target describe ===")
        train[target_cols].describe()
    
        # гистограммы таргетов
        fig, axes = plt.subplots(1, len(target_cols), figsize=(6 * len(target_cols), 4))
        if len(target_cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, target_cols):
            ax.hist(train[col].dropna(), bins=50)
            ax.set_title(col)
        plt.tight_layout()
        plt.show()
    
        if set(["price_p05", "price_p95"]).issubset(train.columns):
            # центр и ширина
            train["price_center"] = 0.5 * (train["price_p05"] + train["price_p95"])
            train["price_width"] = train["price_p95"] - train["price_p05"]
            print("\n=== price_center & price_width describe ===")
            train[["price_center", "price_width"]].describe()
    
            # ratio p95/p05
            ratio = train["price_p95"] / train["price_p05"].replace(0, np.nan)
            plt.figure(figsize=(6, 4))
            plt.hist(ratio.dropna(), bins=50)
            plt.title("ratio = price_p95 / price_p05")
            plt.xlabel("ratio")
            plt.show()
            print("ratio stats:")
            ratio.describe()
    else:
        print("Таргеты price_p05/price_p95 в train не найдены.")

    num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
    print("Numeric columns:", num_cols)
    
    if target_cols:
        corr = train[num_cols].corr()
    
        for tc in target_cols:
            print(f"\n=== Correlation with {tc} ===")
            corr_tc = corr[tc].sort_values(ascending=False)
            corr_tc
    
            # топ по модулю
            top_n = 20
            corr_tc_top = corr_tc.reindex(corr_tc.abs().sort_values(ascending=False).index)[:top_n]
    
            plt.figure(figsize=(6, 6))
            corr_tc_top[::-1].plot(kind="barh")
            plt.title(f"Top correlations with {tc}")
            plt.tight_layout()
            plt.show()
    else:
        print("Корреляции с таргетом пропущены (таргет не найден).")
    
    # Проверяем, нет ли констант/почти константных/очень странных числовых фичей
    suspicious = []
    for col in num_cols:
        s = train[col]
        suspicious.append({
            "column": col,
            "n_unique": s.nunique(),
            "missing_ratio": s.isna().mean(),
            "std": s.std()
        })
    
    suspicious_df = pd.DataFrame(suspicious).sort_values("n_unique")
    print("\nЧисловые фичи: кардинальность и разброс:")
    suspicious_df
    
    # посмотрим на несколько самых "подозрительных"
    for col in suspicious_df.head(5)["column"]:
        print(f"\n=== {col} ===")
        train[col].describe()
        plt.figure(figsize=(5, 3))
        plt.hist(train[col].dropna(), bins=50)
        plt.title(col)
        plt.tight_layout()
        plt.show()

    ##############################################################
    # object-подобные
    cat_cols_obj = train.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # часто в задачах признаки категорий лежат в int64, поэтому добавляем вручную по имени
    likely_cat_int = []
    for name in ["management_group_id", "first_category_id", "second_category_id", 
                 "third_category_id", "product_id", "dow", "month", "week_of_year"]:
        if name in train.columns and name not in cat_cols_obj:
            likely_cat_int.append(name)
    
    cat_cols = cat_cols_obj + likely_cat_int
    cat_cols = list(dict.fromkeys(cat_cols))  # remove duplicates, keep order
    
    print("Object-like categorical columns:", cat_cols_obj)
    print("All potential categorical columns:", cat_cols)
    
    cat_summary = []
    for col in cat_cols:
        s = train[col]
        cat_summary.append({
            "column": col,
            "dtype": s.dtype,
            "n_unique": s.nunique(),
            "most_common_values": list(s.value_counts().head(5).index)
        })
    
    cat_summary_df = pd.DataFrame(cat_summary).sort_values("n_unique")
    cat_summary_df
    
    # Для нескольких категорий — посмотрим распределение
    for col in cat_summary_df.head(5)["column"]:
        print(f"\n=== value_counts for {col} ===")
        train[col].value_counts().head(10)


    ##############################
    if set(["price_p05", "price_p95"]).issubset(train.columns):
        if "price_center" not in train.columns:
            train["price_center"] = 0.5 * (train["price_p05"] + train["price_p95"])
            train["price_width"] = train["price_p95"] - train["price_p05"]
    
        # выберем несколько ключевых категориальных (если есть)
        cats_to_check = [c for c in [
            "management_group_id", "first_category_id", "second_category_id",
            "third_category_id", "dow", "month"
        ] if c in train.columns]
    
        for col in cats_to_check:
            print(f"\n=== {col} vs price_center ===")
            tmp = (
                train
                .groupby(col)
                .agg(
                    n=("price_center", "size"),
                    center_mean=("price_center", "mean"),
                    width_mean=("price_width", "mean"),
                )
                .sort_values("center_mean")
            )
            tmp.head(10)
            tmp.tail(10)
    else:
        print("Таргеты не найдены, зависимость от категорий не анализируем.")
    if set(["price_p05", "price_p95"]).issubset(train.columns):
        train["price_center"] = 0.5 * (train["price_p05"] + train["price_p95"])
        train["price_width"] = train["price_p95"] - train["price_p05"]
        train["log_price_width"] = np.log1p(train["price_width"].clip(lower=0))
    
        def iqr_bounds(series, k=1.5):
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            low = q1 - k * iqr
            high = q3 + k * iqr
            return low, high
    
        for col in ["price_center", "price_width", "log_price_width"]:
            low, high = iqr_bounds(train[col], k=3.0)
            outliers = (~train[col].between(low, high)).sum()
            print(f"{col}: outliers (k=3 IQR) = {outliers} из {len(train)}")
    else:
        print("Таргеты не найдены, IQR-анализ не выполняем.")

    ##################################
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    print("\nTrain columns:")
    print(train.columns.tolist())

    train["dt"] = pd.to_datetime(train["dt"])
    test["dt"]  = pd.to_datetime(test["dt"])
    train = train.sort_values(["product_id", "dt"]).reset_index(drop=True)
    test  = test.sort_values(["product_id", "dt"]).reset_index(drop=True)
    
    train["price_width"]  = train["price_p95"] - train["price_p05"]
    train["price_center"] = 0.5 * (train["price_p95"] + train["price_p05"])
    train["log_width"]    = np.log1p(train["price_width"].clip(lower=0))
    
    global_min_dt = min(train["dt"].min(), test["dt"].min())
    train["day_idx"] = (train["dt"] - global_min_dt).dt.days.astype(int)
    test["day_idx"]  = (test["dt"]  - global_min_dt).dt.days.astype(int)

    # --- product clustering -> cluster_id ---
    pivot = train.pivot_table(index="product_id", columns="dt", values="price_center", aggfunc="mean")
    pivot = pivot.sort_index(axis=1).ffill(axis=1).bfill(axis=1)
    X_ts = pivot.values
    X_ts = (X_ts - X_ts.mean(axis=1, keepdims=True)) / (X_ts.std(axis=1, keepdims=True) + 1e-9)
    
    k_clusters = 7
    X_ts_3d = X_ts[:, :, None]
    tskm = TimeSeriesKMeans(
        n_clusters=k_clusters, metric="euclidean", max_iter=50,
        random_state=SEED, n_jobs=1
    )
    cluster_labels = tskm.fit_predict(X_ts_3d)
    # k_clusters = 8
    # try:
        # from tslearn.clustering import TimeSeriesKMeans
        # X_ts_3d = X_ts[:, :, None]
        # tskm = TimeSeriesKMeans(
        #     n_clusters=k_clusters, metric="euclidean", max_iter=50,
        #     random_state=SEED, n_jobs=1
    #     )
    #     cluster_labels = tskm.fit_predict(X_ts_3d)
    # except Exception:
    #     from sklearn.cluster import KMeans
    #     km = KMeans(n_clusters=k_clusters, random_state=SEED, n_init=20)
    #     cluster_labels = km.fit_predict(X_ts)
    
    prod2cluster = pd.DataFrame({"product_id": pivot.index.values, "cluster_id": cluster_labels})
    train = train.merge(prod2cluster, on="product_id", how="left")
    test  = test.merge(prod2cluster, on="product_id", how="left")
    train["cluster_id"] = train["cluster_id"].fillna(-1).astype(int)
    test["cluster_id"]  = test["cluster_id"].fillna(-1).astype(int)


    # --- IoU + calibration ---
    def iou_1d(l_true, u_true, l_pred, u_pred, eps=1e-6):
        if u_true - l_true < eps:
            mid = 0.5*(u_true + l_true)
            l_true, u_true = mid - eps/2, mid + eps/2
        if u_pred - l_pred < eps:
            mid = 0.5*(u_pred + l_pred)
            l_pred, u_pred = mid - eps/2, mid + eps/2
        inter = max(0.0, min(u_true, u_pred) - max(l_true, l_pred))
        union = (u_true - l_true) + (u_pred - l_pred) - inter
        return inter / (union + 1e-12)
    
    def mean_iou(p05_true, p95_true, p05_pred, p95_pred):
        return float(np.mean([iou_1d(a,b,c,d) for a,b,c,d in zip(p05_true, p95_true, p05_pred, p95_pred)]))
    
    # def calibrate_k_on_interval(p05_true, p95_true, lo_pred, hi_pred, grid=None):
    #     if grid is None:
    #         grid = np.linspace(0.6, 1.6, 51)
    #     c = 0.5 * (lo_pred + hi_pred)
    #     w = np.maximum(hi_pred - lo_pred, 1e-6)
    #     best_k, best_s = 1.0, -1.0
    #     for k in grid:
    #         lo = c - 0.5 * (w * k)
    #         hi = c + 0.5 * (w * k)
    #         s = mean_iou(p05_true, p95_true, lo, hi)
    #         if s > best_s:
    #             best_s, best_k = s, float(k)
    #     return best_k, best_s
    
    def calibrate_k_asym(p05_true, p95_true, lo_pred, hi_pred, grid=None):
        """
        Ассиметричная калибровка:
          center = 0.5*(lo_pred + hi_pred)
          width  = max(hi_pred - lo_pred, 1e-6)
          lo_cal = center - 0.5 * k_low  * width
          hi_cal = center + 0.5 * k_high * width
    
        Перебираем k_low, k_high по сетке и выбираем пару с максимальным IoU.
        """
        if grid is None:
            # можно поиграть с диапазоном/шагом
            grid = np.linspace(0.6, 1.8, 25)  # 25^2 = 625 комбинаций
    
        c = 0.5 * (lo_pred + hi_pred)
        w = np.maximum(hi_pred - lo_pred, 1e-6)
    
        best_k_low = 1.0
        best_k_high = 1.0
        best_iou = -1.0
    
        for kl in grid:
            for kh in grid:
                lo_cal = c - 0.5 * kl * w
                hi_cal = c + 0.5 * kh * w
                s = mean_iou(p05_true, p95_true, lo_cal, hi_cal)
                if s > best_iou:
                    best_iou = float(s)
                    best_k_low = float(kl)
                    best_k_high = float(kh)
    
        return best_k_low, best_k_high, best_iou

    # --- last 7 days validation split ---
    all_dates = np.array(sorted(train["dt"].unique()))
    val_days = 7
    val_dates = all_dates[-val_days:]
    tr_dates  = all_dates[:-val_days]
    
    train_tr = train.loc[train["dt"].isin(tr_dates)].copy()
    train_va = train.loc[train["dt"].isin(val_dates)].copy()
    
    # --- features (same as in your code) ---
    target_cols = ["price_p05", "price_p95", "price_width", "price_center", "log_width"]
    drop_cols = ["dt"] + target_cols
    feature_cols = [c for c in train.columns if c not in drop_cols]
    
    cat_cols = [
        "product_id",
        "management_group_id", "first_category_id", "second_category_id", "third_category_id",
        "dow", "day_of_month", "week_of_year", "month",
        "holiday_flag", "activity_flag",
        "cluster_id",
    ]
    cat_cols = [c for c in cat_cols if c in feature_cols]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    def make_ohe(train_part, valid_part):
        all_df = pd.concat([train_part[feature_cols], valid_part[feature_cols]], axis=0).copy()
        for c in cat_cols:
            all_df[c] = all_df[c].replace([np.inf, -np.inf], np.nan).fillna(-1).astype("int32")
        for c in num_cols:
            all_df[c] = all_df[c].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")
        all_ohe = pd.get_dummies(all_df, columns=cat_cols, dummy_na=False)
        n_tr = len(train_part)
        Xtr = sparse.csr_matrix(all_ohe.iloc[:n_tr].to_numpy(dtype=np.float32, copy=False))
        Xva = sparse.csr_matrix(all_ohe.iloc[n_tr:].to_numpy(dtype=np.float32, copy=False))
        return Xtr, Xva
    
    Xtr_csr, Xva_csr = make_ohe(train_tr, train_va)
    
    # --- EXPERIMENT B params (unchanged) ---
    xgb_params = {
        "eta": 0.01,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "min_child_weight": 5,
        "seed": SEED,
        "tree_method": "hist",
        "reg_lambda": 2.0,
        "reg_alpha": 0.0,
        "gamma": 0.0,
    }
    num_boost_round = 6000
    early_stopping_rounds = 1000
    
    # --- train val models (p05 / p95) ---
    dtr_p05 = xgb.DMatrix(Xtr_csr, label=train_tr["price_p05"].values)
    dva_p05 = xgb.DMatrix(Xva_csr, label=train_va["price_p05"].values)
    dtr_p95 = xgb.DMatrix(Xtr_csr, label=train_tr["price_p95"].values)
    dva_p95 = xgb.DMatrix(Xva_csr, label=train_va["price_p95"].values)
    
    # model_p05 = xgb.train(
    #     params=xgb_params, dtrain=dtr_p05, num_boost_round=num_boost_round,
    #     evals=[(dva_p05, "eval"), (dtr_p05, "train")],
    #     early_stopping_rounds=early_stopping_rounds, verbose_eval=200
    # )
    # model_p95 = xgb.train(
    #     params=xgb_params, dtrain=dtr_p95, num_boost_round=num_boost_round,
    #     evals=[(dva_p95, "eval"), (dtr_p95, "train")],
    #     early_stopping_rounds=early_stopping_rounds, verbose_eval=200
    # )
    def rmse(y, p):
        return float(np.sqrt(np.mean((y - p) ** 2)))
    
    yva_p05 = train_va["price_p05"].values
    yva_p95 = train_va["price_p95"].values
    
    model_p05 = xgb.Booster(params=xgb_params, cache=[dtr_p05, dva_p05])
    model_p95 = xgb.Booster(params=xgb_params, cache=[dtr_p95, dva_p95])
    
    best05 = best95 = 1e18
    best05_iter = best95_iter = -1
    wait05 = wait95 = 0
    alive05 = alive95 = True
    
    for i in range(num_boost_round):
        if alive05:
            model_p05.update(dtr_p05, i)
        if alive95:
            model_p95.update(dtr_p95, i)
    
        p05_va = model_p05.predict(dva_p05)
        p95_va = model_p95.predict(dva_p95)
    
        rmse05 = rmse(yva_p05, p05_va)
        rmse95 = rmse(yva_p95, p95_va)
    
        if alive05:
            if rmse05 < best05:
                best05, best05_iter, wait05 = rmse05, i, 0
            else:
                wait05 += 1
                if wait05 >= early_stopping_rounds:
                    alive05 = False
    
        if alive95:
            if rmse95 < best95:
                best95, best95_iter, wait95 = rmse95, i, 0
            else:
                wait95 += 1
                if wait95 >= early_stopping_rounds:
                    alive95 = False
    
        lo = np.minimum(p05_va, p95_va)
        hi = np.maximum(p05_va, p95_va)
        iou_now = mean_iou(yva_p05, yva_p95, lo, hi)
    
        if (i % 200 == 0) or (not alive05 and not alive95):
            print(f"[{i}] rmse05={rmse05:.6f} rmse95={rmse95:.6f} IoU={iou_now:.6f}")
    
        if not alive05 and not alive95:
            break
    
    # чтобы дальше код (n_round = best_iteration*1.1) не ломался
    model_p05.best_iteration = best05_iter
    model_p95.best_iteration = best95_iter
    
    p05_hat = model_p05.predict(dva_p05)
    p95_hat = model_p95.predict(dva_p95)
    lo_raw = np.minimum(p05_hat, p95_hat)
    hi_raw = np.maximum(p05_hat, p95_hat)

    # сырое IoU без калибровки
    iou_raw_B = mean_iou(
        train_va["price_p05"].values,
        train_va["price_p95"].values,
        lo_raw,
        hi_raw
    )
    
    # асимметричная калибровка k_low / k_high
    best_k_low_B, best_k_high_B, iou_cal_B = calibrate_k_asym(
        train_va["price_p05"].values,
        train_va["price_p95"].values,
        lo_raw,
        hi_raw
    )
    
    print("VALID IoU raw:", round(iou_raw_B, 6))
    print(
        "VALID IoU asym-calibrated:",
        round(iou_cal_B, 6),
        "k_low=", round(best_k_low_B, 3),
        "k_high=", round(best_k_high_B, 3),
    )

    ###########################
    # --- full train + test OHE ---
    all_df = pd.concat([train[feature_cols], test[feature_cols]], axis=0).copy()
    for c in cat_cols:
        all_df[c] = all_df[c].replace([np.inf, -np.inf], np.nan).fillna(-1).astype("int32")
    for c in num_cols:
        all_df[c] = all_df[c].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")
    
    all_ohe = pd.get_dummies(all_df, columns=cat_cols, dummy_na=False)
    n_train = len(train)
    X_full = sparse.csr_matrix(all_ohe.iloc[:n_train].to_numpy(dtype=np.float32, copy=False))
    X_test = sparse.csr_matrix(all_ohe.iloc[n_train:].to_numpy(dtype=np.float32, copy=False))
    
    d_full_p05 = xgb.DMatrix(X_full, label=train["price_p05"].values)
    d_full_p95 = xgb.DMatrix(X_full, label=train["price_p95"].values)
    d_test     = xgb.DMatrix(X_test)
    
    n_round_p05 = int(getattr(model_p05, "best_iteration", 1000) * 1.1) or 1000
    n_round_p95 = int(getattr(model_p95, "best_iteration", 1000) * 1.1) or 1000
    
    model_p05_full = xgb.train(xgb_params, d_full_p05, num_boost_round=n_round_p05)
    model_p95_full = xgb.train(xgb_params, d_full_p95, num_boost_round=n_round_p95)
    
    p05_hat_test = model_p05_full.predict(d_test)
    p95_hat_test = model_p95_full.predict(d_test)
    
    # lo = np.minimum(p05_hat_test, p95_hat_test)
    # hi = np.maximum(p05_hat_test, p95_hat_test)
    
    # center = 0.5 * (lo + hi)
    # width  = np.maximum(hi - lo, 1e-6) * float(best_k_B)
    # p05_out = center - 0.5 * width
    # p95_out = center + 0.5 * width
    
    lo = np.minimum(p05_hat_test, p95_hat_test)
    hi = np.maximum(p05_hat_test, p95_hat_test)
    
    center = 0.5 * (lo + hi)
    width  = np.maximum(hi - lo, 1e-6)
    
    p05_out = center - 0.5 * width * float(best_k_low_B)
    p95_out = center + 0.5 * width * float(best_k_high_B)
    # просто упорядочиваем две модели: нижняя = min, верхняя = max
    # p05_out = np.minimum(p05_hat_test, p95_hat_test)
    # p95_out = np.maximum(p05_hat_test, p95_hat_test)

    #######################################
    # ============================================================
    # ДИНАМИЧЕСКАЯ ПОПРАВКА ДЛЯ ХОЛОДНЫХ ТОВАРОВ ПО 3 СОСЕДЯМ В ДЕНЬ
    # ============================================================
    
    # 1) Определяем тёплые и холодные товары (по product_id)
    train_prods = set(train["product_id"].unique())
    test_prods  = set(test["product_id"].unique())
    
    warm_prods = sorted(test_prods & train_prods)      # есть и в train, и в test
    cold_prods = sorted(test_prods - train_prods)      # только в test
    
    print(f"[DYNAMIC KNN] warm_prods: {len(warm_prods)}, cold_prods: {len(cold_prods)}")
    
    # если холодных товаров нет — нечего корректировать
    if len(cold_prods) > 0:
        # мета-признаки для похожести
        meta_cols = [
            "management_group_id",
            "first_category_id",
            "second_category_id",
            "third_category_id",
        ]
        meta_cols = [c for c in meta_cols if c in test.columns]
    
        if len(meta_cols) > 0:
            # 2) готовим test с базовыми предиктами
            df_t = test.copy().reset_index(drop=True)
            df_t["p05_base"] = p05_out
            df_t["p95_base"] = p95_out
    
            # приведение типов мета-признаков
            for c in meta_cols:
                df_t[c] = (
                    df_t[c]
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(-1)
                    .astype("int32")
                )
    
            warm_mask = df_t["product_id"].isin(warm_prods)
            cold_mask = df_t["product_id"].isin(cold_prods)
    
            new_p05 = p05_out.copy()
            new_p95 = p95_out.copy()
    
            K = 3  # число соседей
    
            # 3) идём по датам, внутри каждой даты ищем соседей только среди товаров,
            #    у которых в ЭТУ дату есть строка
            unique_dates = df_t["dt"].unique()
            print(f"[DYNAMIC KNN] unique dates in test: {len(unique_dates)}")
    
            for cur_dt in unique_dates:
                mask_dt = df_t["dt"] == cur_dt
    
                # кандидаты-соседи: тёплые товары в этот день
                mask_warm_dt = mask_dt & warm_mask
                # холодные товары в этот день
                mask_cold_dt = mask_dt & cold_mask
    
                if not mask_cold_dt.any():
                    continue  # в этот день нет холодных товаров
    
                if not mask_warm_dt.any():
                    # в этот день нет тёплых товаров — нечем корректировать
                    continue
    
                warm_rows = df_t.loc[mask_warm_dt].copy()
                cold_rows = df_t.loc[mask_cold_dt].copy()
    
                warm_meta = warm_rows[meta_cols].to_numpy(dtype=np.int32)
                cold_meta = cold_rows[meta_cols].to_numpy(dtype=np.int32)
    
                warm_idx_arr = warm_rows.index.to_numpy()
                cold_idx_arr = cold_rows.index.to_numpy()
    
                # 4) для КАЖДОГО холодного товара в эту дату ищем K ближайших тёплых
                for i_cold in range(cold_meta.shape[0]):
                    v = cold_meta[i_cold]  # мета холодного
                    # Hamming distance по категориям
                    diff = (warm_meta != v).sum(axis=1)
                    order = np.argsort(diff)
                    k_eff = min(K, len(order))
                    neigh_pos = order[:k_eff]
    
                    neigh_idx = warm_idx_arr[neigh_pos]
    
                    # усредняем базовые предсказания по соседям
                    new_p05[cold_idx_arr[i_cold]] = df_t.loc[neigh_idx, "p05_base"].mean()
                    new_p95[cold_idx_arr[i_cold]] = df_t.loc[neigh_idx, "p95_base"].mean()
    
            p05_out = new_p05
            p95_out = new_p95
    
            print("[DYNAMIC KNN] adjustment applied for cold products.")
        else:
            print("[DYNAMIC KNN] no meta_cols for similarity, block skipped.")
    else:
        print("[DYNAMIC KNN] no cold products, block skipped.")

    submission = pd.DataFrame({
        "row_id": np.arange(len(test)),
        "price_p05": p05_out,
        "price_p95": p95_out,
    })
    
    create_submission(submission)
    
    print("=" * 50)
    print("Выполнение завершено успешно!")
    print("=" * 50)
    
if __name__ == "__main__":
    main()
