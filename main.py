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
    import matplotlib.pyplot as plt
    import seaborn as sns
    from wordcloud import WordCloud
    from collections import Counter
    import warnings
    warnings.filterwarnings('ignore')
    
    from sentence_transformers import SentenceTransformer, CrossEncoder
    from catboost import CatBoostRanker, Pool
    from sklearn.model_selection import GroupKFold
    import torch, os, random, numpy as np
    
    import os, re, random, gc, time
    from itertools import product
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import StandardScaler
    
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', 200)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    
    train_df = pd.read_csv('data/train.csv')  
    test_df = pd.read_csv('data/test.csv')    
    
    print("Размер train:", train_df.shape)
    print("Размер test:", test_df.shape)
    
    SEED = 993
    
    def seed_everything(seed=SEED):
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        try:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except:
            pass
    
    seed_everything(SEED)
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", DEVICE)
    
    
    # Жёстко проверим минимальные столбцы
    required_cols = [
        "id","query_id","query",
        "product_id","product_title","product_description","product_bullet_point",
        "product_brand","product_color","product_locale"
    ]
    for c in required_cols:
        if c not in train_df.columns:
            if c != "relevance":  # в test нет релевантности
                assert c in test_df.columns, f"Отсутствует столбец {c} в test.csv"
    
    assert "relevance" in train_df.columns, "В train.csv нет целевой колонки relevance"
    
    
    def clean_text(s: str) -> str:
        s = str(s).lower()
        s = re.sub(r"[^a-zа-я0-9]+", " ", s, flags=re.IGNORECASE)
        return s.strip()
    
    def make_clean_columns(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in ["query","product_title","product_description",
                    "product_bullet_point","product_brand","product_color"]:
            if col in out.columns:
                out[col + "_clean"] = out[col].fillna("").astype(str).map(clean_text)
            else:
                out[col + "_clean"] = ""
        return out
    
    train_df = make_clean_columns(train_df)
    test_df  = make_clean_columns(test_df)
    
    def concat_product_cols(df: pd.DataFrame) -> pd.Series:
        return (
            df["product_title_clean"] + " " +
            df["product_description_clean"] + " " +
            df["product_bullet_point_clean"] + " " +
            df["product_brand_clean"] + " " +
            df["product_color_clean"]
        ).astype(str)
    
    train_df["product_all_clean"] = concat_product_cols(train_df)
    test_df["product_all_clean"]  = concat_product_cols(test_df)
    
    def count_words(s: str) -> int:
        return len(s.split())
    
    train_df["query_len"]  = train_df["query_clean"].map(count_words)
    train_df["title_len"]  = train_df["product_title_clean"].map(count_words)
    train_df["desc_len"]   = train_df["product_description_clean"].map(count_words)
    train_df["bullet_len"] = train_df["product_bullet_point_clean"].map(count_words)
    
    test_df["query_len"]  = test_df["query_clean"].map(count_words)
    test_df["title_len"]  = test_df["product_title_clean"].map(count_words)
    test_df["desc_len"]   = test_df["product_description_clean"].map(count_words)
    test_df["bullet_len"] = test_df["product_bullet_point_clean"].map(count_words)
    
    # пересечение токенов запроса и тайтла
    def overlap_ratio(q, t):
        qs = set(q.split())
        ts = set(t.split())
        if not qs or not ts:
            return 0.0
        inter = len(qs & ts)
        return inter / max(1, len(qs | ts))
    
    def overlap_count(q, t):
        qs = set(q.split())
        ts = set(t.split())
        return float(len(qs & ts))
    
    train_df["overlap_q_title"]       = [overlap_count(q,t) for q,t in zip(train_df["query_clean"], train_df["product_title_clean"])]
    train_df["overlap_q_title_ratio"] = [overlap_ratio(q,t) for q,t in zip(train_df["query_clean"], train_df["product_title_clean"])]
    
    test_df["overlap_q_title"]       = [overlap_count(q,t) for q,t in zip(test_df["query_clean"], test_df["product_title_clean"])]
    test_df["overlap_q_title_ratio"] = [overlap_ratio(q,t) for q,t in zip(test_df["query_clean"], test_df["product_title_clean"])]
    
    # простейшие match-фичи по бренду/цвету: встречается ли слово из бренда в запросе и т.д.
    def contains_any(a: str, b: str) -> float:
        bs = [w for w in b.split() if len(w) >= 2]
        if not bs:
            return 0.0
        for w in bs:
            if w in a.split():
                return 1.0
        return 0.0
    
    train_df["brand_match"] = [contains_any(q, b) for q,b in zip(train_df["query_clean"], train_df["product_brand_clean"])]
    train_df["color_match"] = [contains_any(q, c) for q,c in zip(train_df["query_clean"], train_df["product_color_clean"])]
    
    test_df["brand_match"] = [contains_any(q, b) for q,b in zip(test_df["query_clean"], test_df["product_brand_clean"])]
    test_df["color_match"] = [contains_any(q, c) for q,c in zip(test_df["query_clean"], test_df["product_color_clean"])]
    
    all_text_for_vocab = pd.concat([train_df["query_clean"], test_df["query_clean"],
                                    train_df["product_all_clean"], test_df["product_all_clean"]], axis=0)
    
    tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=3, max_features=120000) ## БЫЛ 3
    tfidf.fit(all_text_for_vocab.values)
    
    q_train_tfidf = tfidf.transform(train_df["query_clean"])
    p_train_tfidf = tfidf.transform(train_df["product_all_clean"])
    
    q_test_tfidf  = tfidf.transform(test_df["query_clean"])
    p_test_tfidf  = tfidf.transform(test_df["product_all_clean"])
    
    def rowwise_cosine(a, b):
        num = a.multiply(b).sum(axis=1).A.ravel()
        denom = np.sqrt(a.multiply(a).sum(axis=1).A.ravel()) * np.sqrt(b.multiply(b).sum(axis=1).A.ravel())
        return np.divide(num, np.maximum(denom, 1e-9))
    
    train_df["cosine_q_product"] = rowwise_cosine(q_train_tfidf, p_train_tfidf)
    test_df["cosine_q_product"]  = rowwise_cosine(q_test_tfidf,  p_test_tfidf)
    
    # экономим память
    del q_train_tfidf, p_train_tfidf, q_test_tfidf, p_test_tfidf
    gc.collect();
    
    sbert_name = "sentence-transformers/all-MiniLM-L6-v2"
    sbert = SentenceTransformer(sbert_name)
    sbert.to(DEVICE)
    
    def encode_in_batches(model, texts, batch_size=128, device=DEVICE):
        outs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            with torch.no_grad():
                emb = model.encode(batch, device=device, convert_to_numpy=True,
                                   show_progress_bar=False, batch_size=len(batch))
            outs.append(emb)
        return np.vstack(outs)
    
    train_queries_clean = train_df["query_clean"].astype(str).tolist()
    test_queries_clean  = test_df["query_clean"].astype(str).tolist()
    
    train_products_clean = train_df["product_all_clean"].astype(str).tolist()
    test_products_clean  = test_df["product_all_clean"].astype(str).tolist()
    
    t0 = time.time()
    q_emb_train = encode_in_batches(sbert, train_queries_clean, batch_size=64)
    p_emb_train = encode_in_batches(sbert, train_products_clean, batch_size=64)
    q_emb_test  = encode_in_batches(sbert, test_queries_clean,  batch_size=64)
    p_emb_test  = encode_in_batches(sbert, test_products_clean,  batch_size=64)
    print(f"SBERT embeddings done in {time.time()-t0:.1f}s")
    
    def make_bert_features(q_emb, p_emb):
        dot = np.sum(q_emb * p_emb, axis=1)
        q_norm = np.linalg.norm(q_emb, axis=1)
        p_norm = np.linalg.norm(p_emb, axis=1)
        cosine = dot / (q_norm * p_norm + 1e-9)
    
        l2 = np.linalg.norm(q_emb - p_emb, axis=1)
        abs_diff_mean = np.mean(np.abs(q_emb - p_emb), axis=1)
        prod_mean = np.mean(q_emb * p_emb, axis=1)
    
        return np.vstack([cosine, l2, abs_diff_mean, prod_mean]).T
    
    bert_feats_train = make_bert_features(q_emb_train, p_emb_train)
    bert_feats_test  = make_bert_features(q_emb_test,  p_emb_test)
    
    train_df["bert_cosine"]        = bert_feats_train[:,0]
    train_df["bert_l2"]            = bert_feats_train[:,1]
    train_df["bert_absdiff_mean"]  = bert_feats_train[:,2]
    train_df["bert_prod_mean"]     = bert_feats_train[:,3]
    
    test_df["bert_cosine"]         = bert_feats_test[:,0]
    test_df["bert_l2"]             = bert_feats_test[:,1]
    test_df["bert_absdiff_mean"]   = bert_feats_test[:,2]
    test_df["bert_prod_mean"]      = bert_feats_test[:,3]
    
    # можно освободить память
    del q_emb_train, p_emb_train, q_emb_test, p_emb_test, bert_feats_train, bert_feats_test
    gc.collect();
    
    ce_model_name = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    ce_model = CrossEncoder(ce_model_name, device=DEVICE)
    
    def ce_predict_in_batches(model, queries, products, batch_size=256):
        out = []
        for i in range(0, len(queries), batch_size):
            pairs = list(zip(queries[i:i+batch_size], products[i:i+batch_size]))
            with torch.no_grad():
                s = model.predict(pairs, batch_size=len(pairs), show_progress_bar=False)
            out.append(np.asarray(s))
        return np.concatenate(out, axis=0)
    
    ce_train_scores = ce_predict_in_batches(ce_model, train_queries_clean, train_products_clean, batch_size=256)
    ce_test_scores  = ce_predict_in_batches(ce_model, test_queries_clean,  test_products_clean,  batch_size=256)
    
    train_df["ce_score_raw"] = ce_train_scores
    test_df["ce_score_raw"]  = ce_test_scores
    
    def add_querywise_ce_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        ranks = np.zeros(len(df), dtype=np.float32)
        inv_ranks = np.zeros(len(df), dtype=np.float32)
        zscores = np.zeros(len(df), dtype=np.float32)
    
        for qid, idx in df.groupby("query_id").indices.items():
            idx = np.asarray(idx)
            scores = df.loc[idx, "ce_score_raw"].values
            order = np.argsort(-scores)
            rank = np.empty_like(order)
            rank[order] = np.arange(1, len(scores)+1)
    
            inv_rank = 1.0 / rank
            mean = scores.mean()
            std = scores.std() if scores.std() > 0 else 1.0
            z = (scores - mean) / std
    
            ranks[idx] = rank
            inv_ranks[idx] = inv_rank
            zscores[idx] = z
    
        df["ce_rank"] = ranks
        df["ce_inv_rank"] = inv_ranks
        df["ce_zscore"] = zscores
        return df
    
    train_df = add_querywise_ce_features(train_df)
    test_df  = add_querywise_ce_features(test_df)
    
    # 1) биграммы + LCS по title
    def bigrams(tokens):
        toks = [t for t in tokens.split() if t]
        return set(zip(toks, toks[1:])) if len(toks) >= 2 else set()
    
    def bigram_overlap_title(q, t):
        bq, bt = bigrams(q), bigrams(t)
        if not bq or not bt: 
            return 0.0
        return len(bq & bt) / len(bq)
    
    def lcs_len(a, b):
        A, B = a.split(), b.split()
        n, m = len(A), len(B)
        if n == 0 or m == 0: 
            return 0
        dp = [0]*(m+1)
        for i in range(1, n+1):
            prev = 0
            for j in range(1, m+1):
                cur = dp[j]
                if A[i-1] == B[j-1]:
                    dp[j] = prev + 1
                else:
                    dp[j] = max(dp[j], dp[j-1])
                prev = cur
        return dp[m]
    
    def lcs_title_ratio(q, t):
        qlen = max(1, len(q.split()))
        return lcs_len(q, t) / qlen
    
    # 2) жёсткий SKU и юниты
    SKU_PAT = re.compile(r"\b([a-z]{1,5}\d+[a-z\d]*|\d+[a-z]{1,5}[a-z\d]*)\b", re.I)
    UNITS_PAT = re.compile(r"\b(\d+(?:[.,]\d+)?)(mm|cm|in|inch|ml|l|gb|tb|hz|khz|mhz)\b", re.I)
    
    def has_sku(text): 
        return 1.0 if SKU_PAT.search(text) else 0.0
    
    def has_units(text): 
        return 1.0 if UNITS_PAT.search(text) else 0.0
    
    def sku_exact_match(q, title_brand_bullets):
        m = SKU_PAT.search(q)
        if not m: 
            return 0.0
        sku = m.group(1).lower()
        return 1.0 if re.search(rf"\b{re.escape(sku)}\b", title_brand_bullets.lower()) else 0.0
    
    # 3) rare_query через средний idf токенов (используем уже обученный tfidf)
    idf_map = {}
    if hasattr(tfidf, "idf_"):
        for tok, idx in tfidf.vocabulary_.items():
            idf_map[tok] = float(tfidf.idf_[idx])
    
    def mean_idf_of_query(q):
        toks = [t for t in q.split() if t in idf_map]
        if not toks: 
            return 0.0
        return float(np.mean([idf_map[t] for t in toks]))
    
    def z_by_query(df, col):
        # per-query z-score
        g = df.groupby("query_id")[col]
        z = (df[col] - g.transform("mean")) / (g.transform("std").replace(0, 1.0))
        return z.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype("float32")
    
    def prepare_aux_block(df):
        # текст для sku-поиска в продукте
        prod_all = (df["product_title_clean"] + " " + 
                    df["product_brand_clean"] + " " + 
                    df["product_bullet_point_clean"])
    
        df["bigram_overlap_title"] = [
            bigram_overlap_title(q, t) 
            for q,t in zip(df["query_clean"], df["product_title_clean"])
        ]
        df["lcs_title_ratio"] = [
            lcs_title_ratio(q, t) 
            for q,t in zip(df["query_clean"], df["product_title_clean"])
        ]
        df["sku_exact"] = [
            sku_exact_match(q, p) 
            for q,p in zip(df["query_clean"], prod_all)
        ]
    
        # триггеры на уровне строки (будем потом аггрегировать per-query)
        df["flag_has_sku"]   = df["query_clean"].map(has_sku).astype("float32")
        df["flag_has_units"] = df["query_clean"].map(has_units).astype("float32")
        df["query_mean_idf"] = df["query_clean"].map(mean_idf_of_query).astype("float32")
        return df
    
    train_df = prepare_aux_block(train_df)
    test_df  = prepare_aux_block(test_df)
    
    # порог редкости запроса — верхний квартиль по train
    rare_T = float(np.quantile(train_df.drop_duplicates("query_id")["query_mean_idf"], 0.75))
    
    # per-query z для якорей
    for c in ["bigram_overlap_title", "lcs_title_ratio"]:
        train_df[c + "_z"] = z_by_query(train_df, c)
        test_df[c + "_z"]  = z_by_query(test_df,  c)
    
    # агрегаты флажков на уровне запроса (все строки одного запроса получают один и тот же w)
    q_agg = train_df.groupby("query_id").agg(
        q_has_sku=("flag_has_sku", "max"),
        q_has_units=("flag_has_units", "max"),
        q_mean_idf=("query_mean_idf", "mean"),
    ).reset_index()
    train_df = train_df.merge(q_agg, on="query_id", how="left")
    
    q_agg_test = test_df.groupby("query_id").agg(
        q_has_sku=("flag_has_sku", "max"),
        q_has_units=("flag_has_units", "max"),
        q_mean_idf=("query_mean_idf", "mean"),
    ).reset_index()
    test_df = test_df.merge(q_agg_test, on="query_id", how="left")
    
    # финальный вспомогательный скор (пер-запросно нормирован)
    train_df["aux_score_raw"] = np.maximum.reduce([
        train_df["bigram_overlap_title_z"].values,
        train_df["lcs_title_ratio_z"].values,
        train_df["sku_exact"].values.astype("float32")*50.0  # «бесконечный» якорь
    ]).astype("float32")
    test_df["aux_score_raw"] = np.maximum.reduce([
        test_df["bigram_overlap_title_z"].values,
        test_df["lcs_title_ratio_z"].values,
        test_df["sku_exact"].values.astype("float32")*50.0
    ]).astype("float32")
    
    # per-query z для aux_score_raw
    train_df["aux_score_z"] = z_by_query(train_df, "aux_score_raw")
    test_df["aux_score_z"]  = z_by_query(test_df,  "aux_score_raw")
    
    def build_title_pack(df):
        t = (
            df["product_title_clean"].fillna("") + " " +
            df["product_brand_clean"].fillna("") + " " +
            df["product_color_clean"].fillna("")
        ).str.strip()
        return t
    
    train_title_pack = build_title_pack(train_df)
    test_title_pack  = build_title_pack(test_df)
    
    ce_model_title = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", device=DEVICE)
    
    def ce_predict_pairs(model, left, right, batch_size=64):
        out = []
        for i in range(0, len(left), batch_size):
            pairs = list(zip(left[i:i+batch_size], right[i:i+batch_size]))
            with torch.no_grad():
                s = model.predict(pairs, batch_size=len(pairs), show_progress_bar=False)
            out.append(np.asarray(s))
        return np.concatenate(out)
    
    train_df["ce_title_score_raw"] = ce_predict_pairs(ce_model_title, train_df["query_clean"].tolist(), train_title_pack.tolist())
    test_df["ce_title_score_raw"]  = ce_predict_pairs(ce_model_title,  test_df["query_clean"].tolist(),  test_title_pack.tolist())
    
    # пер-запросная нормализация
    def add_qwise_z(df, col, qid_col="query_id"):
        g = df.groupby(qid_col)[col]
        z = (df[col] - g.transform("mean")) / g.transform("std").replace(0, 1.0)
        return z.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    
    train_df["ce_title_z"] = add_qwise_z(train_df, "ce_title_score_raw")
    test_df["ce_title_z"]  = add_qwise_z(test_df,  "ce_title_score_raw")
    
    base_feature_cols = [
        "query_len","title_len","desc_len","bullet_len",
        "overlap_q_title","overlap_q_title_ratio",
        "brand_match","color_match",
        "cosine_q_product",
    ]
    
    bert_feature_cols = [
        "bert_cosine","bert_l2","bert_absdiff_mean","bert_prod_mean",
    ]
    
    ce_feature_cols = [
        "ce_score_raw","ce_rank","ce_inv_rank","ce_zscore",'ce_title_z'
    ]
    
    feature_cols = base_feature_cols + bert_feature_cols + ce_feature_cols
    print("Всего фичей:", len(feature_cols))
    
    X_train = train_df[feature_cols].astype("float32").values
    X_test  = test_df[feature_cols].astype("float32").values
    
    y = train_df["relevance"].values.astype(float)
    groups = train_df["query_id"].values
    groups_test = test_df["query_id"].values
    
    print("Shapes:", X_train.shape, X_test.shape)
    
    
    def dcg_at_k(relevances, k=10):
        relevances = np.asarray(relevances)[:k]
        if relevances.size == 0:
            return 0.0
        discounts = 1.0 / np.log2(np.arange(2, relevances.size + 2))
        gains = (2 ** relevances - 1)
        return float(np.sum(gains * discounts))
    
    def ndcg_for_single_query(y_true, y_pred, k=10):
        order = np.argsort(-y_pred)
        y_true_sorted = np.asarray(y_true)[order]
        dcg = dcg_at_k(y_true_sorted, k=k)
    
        ideal = np.sort(np.asarray(y_true))[::-1]
        idcg = dcg_at_k(ideal, k=k)
        return 0.0 if idcg == 0 else dcg / idcg
    
    def ndcg_at_k(y_true, y_pred, query_ids, k=10):
        df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "qid": query_ids})
        scores = [ndcg_for_single_query(g.y_true.values, g.y_pred.values, k)
                  for _, g in df.groupby("qid", sort=False)]
        return float(np.mean(scores))
    
    def cv_catboost_ranker(X, y, groups, n_splits=20, k=10):
        gkf = GroupKFold(n_splits=n_splits)
        fold_scores = []
    
        use_gpu = (DEVICE == "cuda")
        for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups), 1):
            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]
            g_tr, g_va = groups[tr_idx], groups[va_idx]
    
            train_pool = Pool(X_tr, y_tr, group_id=g_tr)
            val_pool   = Pool(X_va, y_va, group_id=g_va)
    
            model = CatBoostRanker(
                loss_function="YetiRank",
                iterations=1000,
                learning_rate=0.01,
                depth=3,
                random_seed=SEED,
                verbose=False,
                task_type= "GPU" if use_gpu else "CPU",
                devices="0" if use_gpu else None,
            )
            model.fit(train_pool, eval_set=val_pool, verbose=False)
            y_va_pred = model.predict(val_pool)
            score = ndcg_at_k(y_va, y_va_pred, g_va, k=k)
            fold_scores.append(score)
            print(f"Fold {fold:02d}  NDCG@{k}: {score:.4f}")
    
        print(f"\nMean NDCG@{k} over {n_splits} folds: {np.mean(fold_scores):.4f}")
        return fold_scores
    
    cv_scores = cv_catboost_ranker(X_train, y, groups, n_splits=20, k=10)
    print("CV mean:", np.mean(cv_scores))
    
    use_gpu = (DEVICE == "cuda")
    train_pool_full = Pool(X_train, y, group_id=groups)
    
    final_model = CatBoostRanker(
        loss_function="YetiRank",
        iterations=1000,
        learning_rate=0.01,
        depth=3,
        random_seed=SEED,
        verbose=False,
        task_type="GPU" if use_gpu else "CPU",
        devices="0" if use_gpu else None,
    )
    final_model.fit(train_pool_full, verbose=False)
    
    test_pool = Pool(X_test, group_id=groups_test)
    test_pred = final_model.predict(test_pool)
    
    submission = pd.DataFrame({
         "id": test_df["id"],
         "prediction": test_pred
     })

    create_submission(submission)
    
    print("=" * 50)
    print("Выполнение завершено успешно!")
    print("=" * 50)
    
if __name__ == "__main__":
    main()
