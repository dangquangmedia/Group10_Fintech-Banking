def main():
# app.py
import io
import time
import json
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    mean_squared_error,
    r2_score,
)
import joblib

import lightgbm as lgb
import os, sys, traceback
import streamlit as st

st.write("✅ App started")   # để biết code đã chạy tới đây
st.write("Python:", sys.version)
st.write("PID:", os.getpid())

# Optional (SHAP may be heavy)
try:
    import shap
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False

# Silence noisy warnings (you can comment out if needed)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

st.set_page_config(page_title="Fintech Banking - Credit Scoring Demo", layout="wide")


# ---------------------------
# Utilities
# ---------------------------
def _read_csv_uploaded(uploaded_file) -> pd.DataFrame:
    """Read CSV from Streamlit UploadedFile safely."""
    if uploaded_file is None:
        raise ValueError("Chưa upload file CSV.")
    data = uploaded_file.getvalue()
    return pd.read_csv(io.BytesIO(data), low_memory=False)


def _guess_target_column(df: pd.DataFrame) -> Optional[str]:
    """Try to guess target column name."""
    candidates = [
        "target", "label", "y", "class", "loan_status", "status",
        "Credit_Score", "credit_score", "default", "is_default"
    ]
    cols = [c.lower() for c in df.columns]
    for cand in candidates:
        if cand.lower() in cols:
            return df.columns[cols.index(cand.lower())]
    return None


def _detect_task_type(y: pd.Series) -> str:
    """Return 'classification' or 'regression'."""
    # If numeric with many unique values -> regression
    if pd.api.types.is_numeric_dtype(y):
        nunique = y.nunique(dropna=True)
        # heuristic: small unique count looks like classification
        if nunique <= 20:
            return "classification"
        return "regression"
    # object/category -> classification
    return "classification"


def _split_features_target(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise ValueError(f"Không tìm thấy cột target: {target_col}")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def _build_preprocess_pipeline(X: pd.DataFrame) -> Tuple[ColumnTransformer, list, list]:
    """Build preprocessing: numeric impute + categorical impute+onehot."""
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop"
    )
    return preprocessor, numeric_cols, categorical_cols


@dataclass
class TrainResult:
    model: Any
    preprocessor: Any
    metrics: Dict[str, Any]
    task_type: str
    target_col: str
    feature_count_after: int


def _train_lgbm(
    df: pd.DataFrame,
    target_col: str,
    test_size: float,
    params: Dict[str, Any],
    enable_shap: bool = False,
    shap_max_samples: int = 800,
    random_state: int = 42,
) -> TrainResult:
    X, y = _split_features_target(df, target_col)
    task_type = _detect_task_type(y)

    # For classification: ensure y is clean labels
    if task_type == "classification":
        y = y.astype("category")

    # Split
    stratify = y if task_type == "classification" and y.nunique() > 1 else None
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    preprocessor, num_cols, cat_cols = _build_preprocess_pipeline(X_train)

    # Prepare model
    common = dict(
        n_estimators=int(params["n_estimators"]),
        learning_rate=float(params["learning_rate"]),
        num_leaves=int(params["num_leaves"]),
        max_depth=int(params["max_depth"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        reg_lambda=float(params["reg_lambda"]),
        min_child_samples=int(params["min_child_samples"]),
        random_state=random_state,
        n_jobs=int(params["n_jobs"]),
        force_col_wise=True,  # remove auto test overhead
        verbose=-1,
    )

    if task_type == "classification":
        model = lgb.LGBMClassifier(**common)
        eval_metric = "multi_logloss" if y_train.nunique() > 2 else "binary_logloss"
    else:
        model = lgb.LGBMRegressor(**common)
        eval_metric = "rmse"

    pipe = Pipeline(steps=[
        ("prep", preprocessor),
        ("lgbm", model),
    ])

    # Progress UI (best-effort)
    prog = st.progress(0)
    status = st.empty()

    start = time.time()

    def _cb(env):
        # env.iteration is 0-based
        it = env.iteration + 1
        total = int(params["n_estimators"])
        # update every ~2%
        if total > 0 and (it == 1 or it % max(1, total // 50) == 0 or it == total):
            prog.progress(min(1.0, it / total))
            status.write(f"Đang huấn luyện... {it}/{total}")

    callbacks = [lgb.callback.early_stopping(stopping_rounds=int(params["early_stopping"]), verbose=False),
                 _cb]

    # Fit
    pipe.fit(
        X_train, y_train,
        lgbm__eval_set=[(preprocessor.fit_transform(X_valid), y_valid)] if False else None
    )

    # NOTE:
    # LGBM sklearn wrapper doesn't accept eval_set in Pipeline cleanly without parameter routing.
    # We'll do early stopping by refit model directly on transformed data for reliability.

    # Refit properly with early stopping:
    X_train_t = preprocessor.fit_transform(X_train)
    X_valid_t = preprocessor.transform(X_valid)

    if task_type == "classification":
        model2 = lgb.LGBMClassifier(**common)
        model2.fit(
            X_train_t, y_train,
            eval_set=[(X_valid_t, y_valid)],
            eval_metric=eval_metric,
            callbacks=callbacks
        )
    else:
        model2 = lgb.LGBMRegressor(**common)
        model2.fit(
            X_train_t, y_train,
            eval_set=[(X_valid_t, y_valid)],
            eval_metric=eval_metric,
            callbacks=callbacks
        )

    elapsed = time.time() - start
    prog.progress(1.0)
    status.write(f"Huấn luyện xong. Thời gian: {elapsed:.1f}s")

    # Evaluate
    if task_type == "classification":
        y_pred = model2.predict(X_valid_t)
        acc = accuracy_score(y_valid, y_pred)
        f1 = f1_score(y_valid, y_pred, average="weighted")
        metrics = {
            "accuracy": float(acc),
            "f1_weighted": float(f1),
            "report": classification_report(y_valid, y_pred, output_dict=True),
        }
    else:
        y_pred = model2.predict(X_valid_t)
        mse = mean_squared_error(y_valid, y_pred)
        r2 = r2_score(y_valid, y_pred)
        metrics = {"mse": float(mse), "r2": float(r2)}

    # Build final pipeline object (preprocessor + trained model)
    final_pipe = Pipeline(steps=[("prep", preprocessor), ("lgbm", model2)])

    # Feature count after onehot
    feature_count_after = X_train_t.shape[1]

    # SHAP (optional) - compute later in UI if enabled
    if enable_shap and not _HAS_SHAP:
        metrics["shap_warning"] = "SHAP chưa sẵn sàng (thiếu package shap)."

    return TrainResult(
        model=final_pipe,
        preprocessor=preprocessor,
        metrics=metrics,
        task_type=task_type,
        target_col=target_col,
        feature_count_after=feature_count_after,
    )


def _download_bytes(obj) -> bytes:
    bio = io.BytesIO()
    joblib.dump(obj, bio)
    return bio.getvalue()


# ---------------------------
# Session state init
# ---------------------------
if "trained_model" not in st.session_state:
    st.session_state.trained_model = None
if "train_result" not in st.session_state:
    st.session_state.train_result = None
if "last_train_df_shape" not in st.session_state:
    st.session_state.last_train_df_shape = None


# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.title("Chế độ")
mode = st.sidebar.radio(
    "Chọn chế độ",
    ["Tra nhanh (SDT/CCCD/ID)", "Chấm điểm chi tiết", "Giải thích & Đạo đức", "Quản trị (train & upload)"],
    index=3
)

with st.sidebar:
    if st.session_state.trained_model is None:
        st.info("Chưa có model – mô phỏng")
    else:
        tr = st.session_state.train_result
        st.success("Đã có model")
        if tr:
            st.caption(f"Task: {tr.task_type} | Target: {tr.target_col} | #features(after): {tr.feature_count_after}")


# ---------------------------
# Main UI
# ---------------------------
st.title("PB-025 / Fintech Banking - Credit Scoring Demo")

if mode == "Tra nhanh (SDT/CCCD/ID)":
    st.subheader("Tra nhanh (mô phỏng)")
    col1, col2 = st.columns([2, 3])
    with col1:
        query_id = st.text_input("Nhập SĐT/CCCD/ID", value="")
        btn = st.button("Tra cứu")
    with col2:
        st.caption("Gợi ý: chế độ này mô phỏng, dùng để demo luồng. Nếu muốn tra thật, cần mapping ID → dữ liệu đặc trưng.")
        if btn:
            if not query_id.strip():
                st.warning("Vui lòng nhập ID.")
            else:
                score = np.random.uniform(0, 1)
                st.metric("Điểm (mô phỏng)", f"{score:.3f}")
                st.write("Khuyến nghị:", "Duyệt" if score > 0.6 else "Cân nhắc" if score > 0.4 else "Từ chối")

elif mode == "Chấm điểm chi tiết":
    st.subheader("Chấm điểm chi tiết (theo input)")
    if st.session_state.trained_model is None:
        st.warning("Chưa có model. Vào **Quản trị (train & upload)** để huấn luyện trước.")
    else:
        model = st.session_state.trained_model
        st.caption("Nhập dữ liệu 1 khách hàng (một dòng). Bạn có thể nhập tay hoặc upload CSV 1 dòng.")

        tab1, tab2 = st.tabs(["Nhập tay", "Upload 1 dòng CSV"])
        with tab1:
            st.info("Vì dataset mỗi nhóm khác nhau, mình làm generic: chọn cột và nhập giá trị theo tên cột.")
            cols = st.text_area("Danh sách cột (mỗi cột 1 dòng)", height=160, placeholder="age\nincome\ntenure\n...")
            if st.button("Tạo form nhập"):
                st.session_state._manual_cols = [c.strip() for c in cols.splitlines() if c.strip()]

            if "_manual_cols" in st.session_state and st.session_state._manual_cols:
                inputs = {}
                for c in st.session_state._manual_cols:
                    inputs[c] = st.text_input(f"{c}", value="")
                if st.button("Chấm điểm"):
                    df_one = pd.DataFrame([inputs])
                    try:
                        pred = model.predict(df_one)
                        st.write("Kết quả dự đoán:", pred[0])
                        if hasattr(model.named_steps["lgbm"], "predict_proba"):
                            proba = model.predict_proba(df_one)[0]
                            st.write("Xác suất:", proba)
                    except Exception as e:
                        st.error(f"Lỗi chấm điểm: {e}")

        with tab2:
            up = st.file_uploader("Upload CSV (1 dòng dữ liệu)", type=["csv"])
            if up is not None:
                df_one = _read_csv_uploaded(up)
                st.write("Preview:", df_one.head(3))
                if st.button("Chấm điểm từ CSV"):
                    try:
                        pred = model.predict(df_one)
                        st.write("Kết quả dự đoán:", pred)
                        if hasattr(model.named_steps["lgbm"], "predict_proba"):
                            proba = model.predict_proba(df_one)
                            st.write("Xác suất:", proba)
                    except Exception as e:
                        st.error(f"Lỗi chấm điểm: {e}")

elif mode == "Giải thích & Đạo đức":
    st.subheader("Giải thích & Đạo đức (demo)")
    st.write(
        """
- **Giải thích (Explainability):** dùng SHAP/feature importance để lý giải mô hình.
- **Công bằng (Fairness):** kiểm tra nhóm nhạy cảm (giới tính, vùng miền...) nếu có.
- **An toàn & minh bạch:** log, kiểm soát version model, dữ liệu đầu vào hợp lệ.
        """
    )
    if st.session_state.trained_model is None:
        st.warning("Chưa có model để giải thích. Vào **Quản trị (train & upload)** để huấn luyện trước.")
    else:
        tr = st.session_state.train_result
        model = st.session_state.trained_model

        c1, c2 = st.columns([1, 1])
        with c1:
            st.write("**Metrics:**")
            st.json(tr.metrics)

        with c2:
            st.write("**Feature importance (top 30):**")
            try:
                booster = model.named_steps["lgbm"]
                importances = booster.feature_importances_
                topk = min(30, len(importances))
                idx = np.argsort(importances)[::-1][:topk]
                df_imp = pd.DataFrame({"feature_idx": idx, "importance": importances[idx]})
                st.dataframe(df_imp, use_container_width=True)
                st.caption("Ghi chú: vì có OneHotEncoder, feature là index sau transform.")
            except Exception as e:
                st.error(f"Không lấy được importance: {e}")

        st.divider()
        st.write("### SHAP (tùy chọn)")
        if not _HAS_SHAP:
            st.info("Chưa cài SHAP hoặc SHAP lỗi import. Nếu muốn dùng, giữ `shap` trong requirements.")
        else:
            st.caption("SHAP có thể chậm. Nên dùng trên mẫu nhỏ.")
            up2 = st.file_uploader("Upload CSV để giải thích (tối đa vài nghìn dòng)", type=["csv"], key="shap_csv")
            max_samples = st.slider("Giới hạn mẫu SHAP", 50, 2000, 300, 50)
            if up2 is not None and st.button("Chạy SHAP"):
                df_explain = _read_csv_uploaded(up2)
                target_col = tr.target_col
                if target_col in df_explain.columns:
                    df_explain = df_explain.drop(columns=[target_col])
                df_explain = df_explain.head(max_samples)

                try:
                    X_t = model.named_steps["prep"].transform(df_explain)
                    explainer = shap.TreeExplainer(model.named_steps["lgbm"])
                    with st.spinner("Đang tính SHAP..."):
                        shap_values = explainer.shap_values(X_t)

                    st.success("Xong SHAP. (Hiển thị summary plot)")

                    import matplotlib.pyplot as plt
                    fig = plt.figure()
                    # For multiclass, shap_values is list
                    if isinstance(shap_values, list):
                        shap.summary_plot(shap_values[0], X_t, show=False)
                    else:
                        shap.summary_plot(shap_values, X_t, show=False)
                    st.pyplot(fig, clear_figure=True)
                except Exception as e:
                    st.error(f"Lỗi SHAP: {e}")

else:
    # Admin / Train
    st.subheader("Quản trị (train & upload)")
    st.caption("Upload dữ liệu train CSV → chọn target → chỉnh hyperparameters → Huấn luyện")

    up = st.file_uploader("Upload TRAIN CSV", type=["csv"], key="train_csv")
    if up is not None:
        df_train = _read_csv_uploaded(up)
        st.write("Preview dữ liệu:", df_train.head())
        st.write(f"Shape: {df_train.shape}")

        guess = _guess_target_column(df_train)

        # Form to avoid rerun-training loop
        with st.form("train_form"):
            c1, c2, c3 = st.columns([1.2, 1.2, 1.2])

            with c1:
                test_size = st.slider("Tỉ lệ test", 0.10, 0.40, 0.20, 0.05)
                target_col = st.selectbox(
                    "Chọn cột target (nhãn)",
                    options=list(df_train.columns),
                    index=list(df_train.columns).index(guess) if guess in df_train.columns else 0
                )
                algo = st.selectbox("Chọn thuật toán", ["LightGBM (LGBM)"], index=0)
                enable_shap = st.checkbox("Bật SHAP (giới hạn mẫu, có thể chậm)", value=False)

            with c2:
                st.caption("Gợi ý: n_estimators 400–1200, learning_rate 0.03–0.1")
                n_estimators = st.slider("n_estimators", 100, 2000, 600, 50)
                learning_rate = st.slider("learning_rate", 0.01, 0.20, 0.05, 0.01)
                min_child_samples = st.slider("min_child_samples", 5, 100, 20, 5)
                reg_lambda = st.slider("reg_lambda", 0.0, 5.0, 0.0, 0.1)

            with c3:
                num_leaves = st.slider("num_leaves", 15, 255, 63, 2)
                max_depth = st.slider("max_depth (-1 = không giới hạn)", -1, 20, -1, 1)
                subsample = st.slider("subsample", 0.50, 1.00, 0.90, 0.05)
                colsample_bytree = st.slider("colsample_bytree", 0.50, 1.00, 0.90, 0.05)

            st.divider()
            c4, c5 = st.columns([1, 3])
            with c4:
                n_jobs = st.selectbox("n_jobs", [1, 2, 4], index=1)
                early_stopping = st.selectbox("early_stopping_rounds", [20, 50, 100], index=1)
            with c5:
                st.caption("Mẹo: nếu Streamlit Cloud chậm, để n_jobs=1 hoặc 2 để ổn định.")

            submitted = st.form_submit_button("Huấn luyện")

        if submitted:
            if target_col not in df_train.columns:
                st.error("Target column không hợp lệ.")
            else:
                with st.spinner("Đang huấn luyện..."):
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
                        early_stopping=early_stopping,
                    )
                    try:
                        result = _train_lgbm(
                            df=df_train,
                            target_col=target_col,
                            test_size=float(test_size),
                            params=params,
                            enable_shap=enable_shap,
                        )
                        st.session_state.trained_model = result.model
                        st.session_state.train_result = result
                        st.session_state.last_train_df_shape = df_train.shape

                        st.success("Huấn luyện thành công ✅")
                        st.write("Metrics:", result.metrics)

                        # Download model
                        model_bytes = _download_bytes(result.model)
                        st.download_button(
                            "Tải model (.joblib)",
                            data=model_bytes,
                            file_name="lgbm_pipeline.joblib",
                            mime="application/octet-stream"
                        )
                    except Exception as e:
                        st.error(f"Huấn luyện lỗi: {e}")

    st.divider()
    st.write("### Trạng thái model")
    if st.session_state.trained_model is None:
        st.info("Chưa có model.")
    else:
        tr = st.session_state.train_result
        st.success("Đã có model trong session")
        st.json({
            "task_type": tr.task_type,
            "target_col": tr.target_col,
            "feature_count_after": tr.feature_count_after,
            "train_df_shape": st.session_state.last_train_df_shape,
        })
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("App crashed:")
        st.exception(e)
        print("CRASH:", traceback.format_exc())
        raise
