# app.py
# PB-025 / Fintech Banking - Credit Scoring Demo (UI đầy đủ 4 chế độ)
# - Train LGBM ổn định trên Streamlit Cloud (tránh OOM)
# - Có upload/download model artifact
# - Giải thích: Feature importance + SHAP (tùy chọn, giới hạn mẫu)

import io
import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    mean_squared_error,
    r2_score,
)

import joblib

warnings.filterwarnings("ignore")

# ✅ MUST be the first Streamlit command
st.set_page_config(
    page_title="PB-025 / Fintech Banking - Credit Scoring Demo",
    layout="wide",
)

# =========================
# Data / Model Helpers
# =========================
def read_csv_uploaded(uploaded_file) -> pd.DataFrame:
    data = uploaded_file.getvalue()
    return pd.read_csv(io.BytesIO(data), low_memory=False)


def guess_target_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "Credit_Score", "credit_score", "target", "label", "y",
        "loan_status", "status", "default", "is_default"
    ]
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def detect_task_type(y: pd.Series) -> str:
    # numeric with many unique -> regression; else classification
    if pd.api.types.is_numeric_dtype(y):
        nunique = y.nunique(dropna=True)
        if nunique <= 20:
            return "classification"
        return "regression"
    return "classification"


def downcast_numeric(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    for c in numeric_cols:
        if pd.api.types.is_integer_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], downcast="integer")
        elif pd.api.types.is_float_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], downcast="float")
    return df


def prepare_features(
    df: pd.DataFrame,
    target_col: str,
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    if target_col not in df.columns:
        raise ValueError(f"Không tìm thấy cột target: {target_col}")

    y = df[target_col]
    X = df.drop(columns=[target_col])

    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in numeric_cols]

    # Fill numeric
    for c in numeric_cols:
        med = X[c].median() if X[c].notna().any() else 0
        X[c] = X[c].fillna(med)

    # Fill categorical + convert to category (LightGBM native categorical)
    for c in cat_cols:
        X[c] = X[c].astype("string").fillna("missing").astype("category")

    X = downcast_numeric(X, numeric_cols)
    return X, y, numeric_cols, cat_cols


@dataclass
class TrainResult:
    model: Any
    task_type: str
    target_col: str
    cat_cols: List[str]
    metrics: Dict[str, Any]
    label_encoder: Optional[LabelEncoder]


def train_lgbm(
    df: pd.DataFrame,
    target_col: str,
    test_size: float,
    params: Dict[str, Any],
    max_train_rows: int = 40000,
    random_state: int = 42,
) -> TrainResult:
    if len(df) > max_train_rows:
        df = df.sample(n=max_train_rows, random_state=random_state)

    X, y, numeric_cols, cat_cols = prepare_features(df, target_col)
    task_type = detect_task_type(y)

    le = None
    if task_type == "classification":
        le = LabelEncoder()
        y_clean = y.astype("string").fillna("missing")
        y_use = le.fit_transform(y_clean)
        num_class = len(le.classes_)
        stratify = y_use if num_class > 1 else None
    else:
        y_use = pd.to_numeric(y, errors="coerce").fillna(y.median() if y.notna().any() else 0)
        stratify = None

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y_use,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )

    common_params = dict(
        n_estimators=int(params["n_estimators"]),
        learning_rate=float(params["learning_rate"]),
        num_leaves=int(params["num_leaves"]),
        max_depth=int(params["max_depth"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        reg_lambda=float(params["reg_lambda"]),
        min_child_samples=int(params["min_child_samples"]),
        n_jobs=int(params["n_jobs"]),
        random_state=random_state,
        force_col_wise=True,   # giảm overhead
        verbose=-1
    )

    # Progress
    total = int(common_params["n_estimators"])
    progress = st.progress(0)
    status = st.empty()

    def cb_env(env):
        it = env.iteration + 1
        if total > 0:
            if it == 1 or it % max(1, total // 40) == 0 or it == total:
                progress.progress(min(1.0, it / total))
                status.write(f"Đang huấn luyện... {it}/{total}")

    callbacks = [
        lgb.callback.early_stopping(stopping_rounds=int(params["early_stopping_rounds"]), verbose=False),
        cb_env
    ]

    start = time.time()

    if task_type == "classification":
        num_class = len(np.unique(y_train))
        model = lgb.LGBMClassifier(
            **common_params,
            objective="multiclass" if num_class > 2 else "binary"
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="multi_logloss" if num_class > 2 else "binary_logloss",
            categorical_feature=cat_cols,
            callbacks=callbacks
        )
        y_pred = model.predict(X_valid)
        acc = accuracy_score(y_valid, y_pred)
        f1w = f1_score(y_valid, y_pred, average="weighted")
        report = classification_report(y_valid, y_pred, output_dict=True)

        metrics = {
            "rows_used": int(len(df)),
            "features": int(X.shape[1]),
            "cat_cols": int(len(cat_cols)),
            "accuracy": float(acc),
            "f1_weighted": float(f1w),
            "report": report
        }

    else:
        model = lgb.LGBMRegressor(
            **common_params,
            objective="regression"
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="rmse",
            categorical_feature=cat_cols,
            callbacks=callbacks
        )
        y_pred = model.predict(X_valid)
        mse = mean_squared_error(y_valid, y_pred)
        r2 = r2_score(y_valid, y_pred)
        metrics = {
            "rows_used": int(len(df)),
            "features": int(X.shape[1]),
            "cat_cols": int(len(cat_cols)),
            "mse": float(mse),
            "r2": float(r2),
        }

    elapsed = time.time() - start
    progress.progress(1.0)
    status.write(f"✅ Huấn luyện xong. Thời gian: {elapsed:.1f}s")

    return TrainResult(
        model=model,
        task_type=task_type,
        target_col=target_col,
        cat_cols=cat_cols,
        metrics=metrics,
        label_encoder=le
    )


def pack_artifact(result: TrainResult) -> bytes:
    blob = {
        "model": result.model,
        "task_type": result.task_type,
        "target_col": result.target_col,
        "cat_cols": result.cat_cols,
        "label_encoder": result.label_encoder,
        "metrics": result.metrics
    }
    bio = io.BytesIO()
    joblib.dump(blob, bio)
    return bio.getvalue()


def load_artifact(uploaded_file) -> TrainResult:
    data = uploaded_file.getvalue()
    blob = joblib.load(io.BytesIO(data))
    return TrainResult(
        model=blob["model"],
        task_type=blob["task_type"],
        target_col=blob["target_col"],
        cat_cols=blob.get("cat_cols", []),
        metrics=blob.get("metrics", {}),
        label_encoder=blob.get("label_encoder", None),
    )


def predict_df(result: TrainResult, df_input: pd.DataFrame) -> pd.DataFrame:
    X = df_input.copy()
    if result.target_col in X.columns:
        X = X.drop(columns=[result.target_col])

    # Convert cat cols
    for c in result.cat_cols:
        if c in X.columns:
            X[c] = X[c].astype("string").fillna("missing").astype("category")

    # Fill numeric missing
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]) and X[c].isna().any():
            X[c] = X[c].fillna(X[c].median() if X[c].notna().any() else 0)

    if result.task_type == "classification":
        pred_class = result.model.predict(X)
        out = df_input.copy()
        out["pred_class"] = pred_class

        # predict_proba (nếu có)
        if hasattr(result.model, "predict_proba"):
            proba = result.model.predict_proba(X)
            # nhãn gốc
            if result.label_encoder is not None:
                labels = list(result.label_encoder.classes_)
            else:
                labels = [f"class_{i}" for i in range(proba.shape[1])]
            for i, lab in enumerate(labels):
                out[f"proba_{lab}"] = proba[:, i]

            # label gốc
            if result.label_encoder is not None:
                out["pred_label"] = result.label_encoder.inverse_transform(pred_class.astype(int))

        return out

    # regression
    pred = result.model.predict(X)
    out = df_input.copy()
    out["pred"] = pred
    return out


def get_feature_importance(result: TrainResult) -> pd.DataFrame:
    booster = result.model.booster_
    names = booster.feature_name()
    gains = booster.feature_importance(importance_type="gain")
    df = pd.DataFrame({"feature": names, "gain": gains})
    df = df.sort_values("gain", ascending=False).reset_index(drop=True)
    return df


# =========================
# Session State
# =========================
if "train_result" not in st.session_state:
    st.session_state.train_result = None
if "is_training" not in st.session_state:
    st.session_state.is_training = False
if "last_scored_df" not in st.session_state:
    st.session_state.last_scored_df = None
if "cached_test_df" not in st.session_state:
    st.session_state.cached_test_df = None

# =========================
# UI Layout
# =========================
st.title("PB-025 / Fintech Banking - Credit Scoring Demo")

with st.sidebar:
    st.header("Chế độ")
    mode = st.radio(
        "Chọn chế độ",
        [
            "Tra nhanh (SDT/CCCD/ID)",
            "Chấm điểm chi tiết",
            "Giải thích & Đạo đức",
            "Quản trị (train & upload)",
        ],
        index=0,
    )

    st.divider()

    if st.session_state.train_result is None:
        st.info("Chưa có model – mô phỏng")
    else:
        tr: TrainResult = st.session_state.train_result
        st.success("Đã có model")
        st.caption(f"Task: {tr.task_type} | Target: {tr.target_col} | Cat cols: {len(tr.cat_cols)}")

# =========================
# Page 1: Quick Lookup
# =========================
if mode == "Tra nhanh (SDT/CCCD/ID)":
    st.subheader("Tra nhanh (SDT/CCCD/ID)")
    st.caption("Mô phỏng tra cứu nhanh. (Nếu có file test, bạn có thể chấm điểm ở tab Chấm điểm chi tiết.)")

    q = st.text_input("Nhập SĐT/CCCD/ID", value="")
    colA, colB = st.columns([1, 3])
    with colA:
        if st.button("Tra cứu"):
            if not q.strip():
                st.warning("Vui lòng nhập ID.")
            else:
                # Mô phỏng kết quả nhanh
                score = float(np.random.uniform(0, 1))
                st.metric("Điểm (mô phỏng)", f"{score:.3f}")
                st.write("Khuyến nghị:", "Duyệt" if score > 0.60 else "Cân nhắc" if score > 0.40 else "Từ chối")

    with colB:
        st.info(
            "Gợi ý nâng cấp: nếu bạn có cột ID trong dữ liệu test, mình có thể map ID → lấy đúng dòng dữ liệu → chấm điểm thật."
        )

# =========================
# Page 2: Detailed Scoring
# =========================
elif mode == "Chấm điểm chi tiết":
    st.subheader("Chấm điểm chi tiết")
    if st.session_state.train_result is None:
        st.warning("Chưa có model. Vào tab **Quản trị (train & upload)** để huấn luyện hoặc upload model.")
    else:
        tr: TrainResult = st.session_state.train_result

        up = st.file_uploader("Upload CSV để chấm điểm (có thể nhiều dòng)", type=["csv"], key="score_csv")
        if up is not None:
            df_test = read_csv_uploaded(up)
            st.session_state.cached_test_df = df_test

            st.write("Preview:", df_test.head())
            st.write("Shape:", df_test.shape)

            if st.button("Chấm điểm"):
                try:
                    with st.spinner("Đang chấm điểm..."):
                        out = predict_df(tr, df_test)
                        st.session_state.last_scored_df = out

                    st.success("✅ Chấm điểm xong")
                    st.dataframe(out.head(50), use_container_width=True)

                    # download
                    csv_bytes = out.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        "Tải kết quả (.csv)",
                        data=csv_bytes,
                        file_name="scored_output.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error("Lỗi chấm điểm:")
                    st.exception(e)

# =========================
# Page 3: Explain & Ethics
# =========================
elif mode == "Giải thích & Đạo đức":
    st.subheader("Giải thích & Đạo đức")
    if st.session_state.train_result is None:
        st.warning("Chưa có model. Vào tab **Quản trị (train & upload)** để huấn luyện hoặc upload model.")
    else:
        tr: TrainResult = st.session_state.train_result

        c1, c2 = st.columns([1.2, 1.0])

        with c1:
            st.markdown("### 1) Feature importance (Gain)")
            try:
                fi = get_feature_importance(tr).head(30)
                st.dataframe(fi, use_container_width=True)
                st.caption("Top 30 feature theo Gain của LightGBM.")
            except Exception as e:
                st.error("Không lấy được feature importance:")
                st.exception(e)

        with c2:
            st.markdown("### 2) Checklist Đạo đức / Tuân thủ")
            st.checkbox("Không dùng thuộc tính nhạy cảm (tôn giáo/chính trị/sức khỏe) làm đầu vào", value=True, disabled=True)
            st.checkbox("Có cơ chế giải thích (feature importance/SHAP)", value=True, disabled=True)
            st.checkbox("Giới hạn dữ liệu cá nhân, tuân thủ PDPL/NDOP (mô phỏng)", value=True, disabled=True)
            st.checkbox("Có log/audit cho quyết định (mô phỏng)", value=True, disabled=True)

        st.divider()

        st.markdown("### 3) SHAP (tùy chọn – giới hạn mẫu để tránh crash Cloud)")
        enable_shap = st.checkbox("Bật SHAP (có thể chậm)", value=False)
        shap_max_rows = st.selectbox("Giới hạn số dòng SHAP", [200, 500, 1000, 2000], index=1)

        if enable_shap:
            # cần có dữ liệu test đã upload hoặc dùng lại train? ở đây dùng cached_test_df nếu có
            if st.session_state.cached_test_df is None:
                st.info("Hãy upload file ở tab **Chấm điểm chi tiết** trước để có dữ liệu chạy SHAP.")
            else:
                df_for_shap = st.session_state.cached_test_df.copy()
                # lấy sample
                if len(df_for_shap) > shap_max_rows:
                    df_for_shap = df_for_shap.sample(n=shap_max_rows, random_state=42)

                try:
                    import shap  # nặng, chỉ import khi cần
                    with st.spinner("Đang tính SHAP..."):
                        # preprocess tương tự predict
                        X = df_for_shap.copy()
                        if tr.target_col in X.columns:
                            X = X.drop(columns=[tr.target_col])
                        for c in tr.cat_cols:
                            if c in X.columns:
                                X[c] = X[c].astype("string").fillna("missing").astype("category")
                        for c in X.columns:
                            if pd.api.types.is_numeric_dtype(X[c]) and X[c].isna().any():
                                X[c] = X[c].fillna(X[c].median() if X[c].notna().any() else 0)

                        # TreeExplainer
                        explainer = shap.TreeExplainer(tr.model)
                        shap_values = explainer.shap_values(X)

                    st.success("✅ SHAP xong (tóm tắt)")
                    st.caption("Do giới hạn UI, phần SHAP summary plot có thể nặng; ở đây hiển thị bảng mean(|SHAP|).")

                    # Tính mean abs shap
                    if isinstance(shap_values, list):
                        # multiclass: lấy trung bình theo class
                        abs_means = np.mean([np.mean(np.abs(sv), axis=0) for sv in shap_values], axis=0)
                    else:
                        abs_means = np.mean(np.abs(shap_values), axis=0)

                    shap_df = pd.DataFrame({"feature": X.columns, "mean_abs_shap": abs_means})
                    shap_df = shap_df.sort_values("mean_abs_shap", ascending=False).head(30)
                    st.dataframe(shap_df, use_container_width=True)

                except Exception as e:
                    st.error("SHAP bị lỗi (Cloud có thể thiếu RAM nếu file quá lớn):")
                    st.exception(e)

# =========================
# Page 4: Admin (Train & Upload)
# =========================
else:
    st.subheader("Quản trị (train & upload)")
    st.caption("Upload TRAIN CSV → chọn target → chỉnh hyperparameters → Huấn luyện. Hoặc upload model artifact đã lưu.")

    # --- Upload model artifact
    st.markdown("### A) Upload model (.joblib) đã lưu")
    art = st.file_uploader("Upload model artifact", type=["joblib"], key="artifact_upload")
    if art is not None:
        try:
            st.session_state.train_result = load_artifact(art)
            st.success("✅ Upload model thành công!")
            st.json(st.session_state.train_result.metrics)
        except Exception as e:
            st.error("Upload model bị lỗi:")
            st.exception(e)

    st.divider()

    # --- Train
    st.markdown("### B) Huấn luyện LightGBM (LGBM)")
    up = st.file_uploader("Upload TRAIN CSV", type=["csv"], key="train_csv")

    if up is not None:
        df_train = read_csv_uploaded(up)

        st.write("Shape:", df_train.shape)
        st.write("Preview:", df_train.head())

        guess = guess_target_column(df_train)

        with st.form("train_form"):
            c1, c2, c3 = st.columns([1.2, 1.2, 1.2])

            with c1:
                test_size = st.slider("Tỉ lệ test", 0.10, 0.40, 0.20, 0.05)
                target_col = st.selectbox(
                    "Chọn cột target (nhãn)",
                    options=list(df_train.columns),
                    index=list(df_train.columns).index(guess) if guess in df_train.columns else 0
                )
                max_train_rows = st.selectbox(
                    "Giới hạn số dòng train (để ổn định Cloud)",
                    [20000, 30000, 40000, 60000, 100000],
                    index=2
                )
                st.caption("Nếu Cloud chậm/lỗi: chọn 30k–40k.")

            with c2:
                st.caption("Gợi ý: n_estimators 400–1200, learning_rate 0.03–0.1")
                n_estimators = st.slider("n_estimators", 100, 2000, 600, 50)
                learning_rate = st.slider("learning_rate", 0.01, 0.20, 0.05, 0.01)
                num_leaves = st.slider("num_leaves", 15, 255, 63, 2)
                max_depth = st.slider("max_depth (-1 = không giới hạn)", -1, 20, -1, 1)

            with c3:
                subsample = st.slider("subsample", 0.50, 1.00, 0.90, 0.05)
                colsample_bytree = st.slider("colsample_bytree", 0.50, 1.00, 0.90, 0.05)
                min_child_samples = st.slider("min_child_samples", 5, 100, 20, 5)
                reg_lambda = st.slider("reg_lambda", 0.0, 5.0, 0.0, 0.1)

            st.divider()
            c4, c5 = st.columns([1, 2])
            with c4:
                n_jobs = st.selectbox("n_jobs", [1, 2, 4], index=0)
                early_stopping_rounds = st.selectbox("early_stopping_rounds", [20, 50, 100], index=1)
            with c5:
                st.caption("Nếu Streamlit Cloud chậm: để n_jobs=1, giảm n_estimators 300–600.")

            submitted = st.form_submit_button("Huấn luyện", disabled=st.session_state.is_training)

        if submitted and not st.session_state.is_training:
            st.session_state.is_training = True
            try:
                params = dict(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    num_leaves=num_leaves,
                    max_depth=max_depth,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    reg_lambda=reg_lambda,
                    min_child_samples=min_child_samples,
                    n_jobs=n_jobs,
                    early_stopping_rounds=early_stopping_rounds,
                )

                with st.spinner("Đang huấn luyện..."):
                    result = train_lgbm(
                        df=df_train,
                        target_col=target_col,
                        test_size=float(test_size),
                        params=params,
                        max_train_rows=int(max_train_rows),
                    )

                st.session_state.train_result = result
                st.success("✅ Huấn luyện thành công!")
                st.json(result.metrics)

                st.download_button(
                    "Tải model (.joblib)",
                    data=pack_artifact(result),
                    file_name="lgbm_model_artifact.joblib",
                    mime="application/octet-stream"
                )

            except Exception as e:
                st.error("Huấn luyện bị lỗi:")
                st.exception(e)
            finally:
                st.session_state.is_training = False
