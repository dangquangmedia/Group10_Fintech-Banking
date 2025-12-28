# app.py
# ✅ IMPORTANT: st.set_page_config MUST be the first Streamlit command.

import io
import time
import warnings
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Fintech Banking - Credit Scoring Demo",
    layout="wide",
)

# -----------------------------
# Helpers
# -----------------------------
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
    # If numeric with many unique -> regression, else classification
    if pd.api.types.is_numeric_dtype(y):
        nunique = y.nunique(dropna=True)
        if nunique <= 20:
            return "classification"
        return "regression"
    return "classification"


def downcast_numeric(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    # Reduce memory footprint (helpful on Streamlit Cloud)
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

    # Identify cols
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in numeric_cols]

    # Basic cleaning
    # numeric: fillna median
    for c in numeric_cols:
        med = X[c].median() if X[c].notna().any() else 0
        X[c] = X[c].fillna(med)

    # categorical: fillna + convert to string then category
    for c in cat_cols:
        X[c] = X[c].astype("string").fillna("missing").astype("category")

    # reduce memory
    X = downcast_numeric(X, numeric_cols)

    return X, y, numeric_cols, cat_cols


@dataclass
class TrainResult:
    model: Any
    task_type: str
    target_col: str
    metrics: Dict[str, Any]
    cat_cols: List[str]
    label_encoder: Optional[LabelEncoder]


def train_lgbm(
    df: pd.DataFrame,
    target_col: str,
    test_size: float,
    params: Dict[str, Any],
    max_train_rows: int = 40000,
    random_state: int = 42,
) -> TrainResult:
    # sample to keep Cloud stable
    if len(df) > max_train_rows:
        df = df.sample(n=max_train_rows, random_state=random_state)

    X, y, numeric_cols, cat_cols = prepare_features(df, target_col)
    task_type = detect_task_type(y)

    le = None
    if task_type == "classification":
        # encode labels for LGBMClassifier
        le = LabelEncoder()
        y_clean = y.astype("string").fillna("missing")
        y_enc = le.fit_transform(y_clean)
        y_use = y_enc
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

    # LightGBM can take pandas DataFrame with category dtypes
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
        force_col_wise=True,
        verbose=-1
    )

    progress = st.progress(0)
    status = st.empty()

    total = int(common_params["n_estimators"])

    def cb_env(env):
        it = env.iteration + 1
        if total > 0:
            if it == 1 or it % max(1, total // 50) == 0 or it == total:
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
        f1 = f1_score(y_valid, y_pred, average="weighted")
        report = classification_report(y_valid, y_pred, output_dict=True)

        metrics = {
            "rows_used": int(len(df)),
            "features": int(X.shape[1]),
            "cat_cols": len(cat_cols),
            "accuracy": float(acc),
            "f1_weighted": float(f1),
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
            "cat_cols": len(cat_cols),
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
        metrics=metrics,
        cat_cols=cat_cols,
        label_encoder=le
    )


def pack_artifact(result: TrainResult) -> bytes:
    blob = {
        "model": result.model,
        "task_type": result.task_type,
        "target_col": result.target_col,
        "cat_cols": result.cat_cols,
        "label_encoder": result.label_encoder,
    }
    bio = io.BytesIO()
    joblib.dump(blob, bio)
    return bio.getvalue()


def predict_one(result: TrainResult, df_input: pd.DataFrame) -> Any:
    # Align preprocessing: convert cat cols to category + fill missing
    X = df_input.copy()
    # If user accidentally includes target col, drop it
    if result.target_col in X.columns:
        X = X.drop(columns=[result.target_col])

    # Ensure columns exist
    for c in result.cat_cols:
        if c in X.columns:
            X[c] = X[c].astype("string").fillna("missing").astype("category")

    # Fill numeric missing quickly
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            if X[c].isna().any():
                X[c] = X[c].fillna(X[c].median() if X[c].notna().any() else 0)

    pred = result.model.predict(X)
    if result.task_type == "classification" and result.label_encoder is not None:
        # convert numeric classes back to label strings
        pred_label = result.label_encoder.inverse_transform(pred.astype(int))
        return pred, pred_label
    return pred, None


# -----------------------------
# Session init
# -----------------------------
if "train_result" not in st.session_state:
    st.session_state.train_result = None
if "is_training" not in st.session_state:
    st.session_state.is_training = False


# -----------------------------
# UI
# -----------------------------
st.title("PB-025 / Fintech Banking - Credit Scoring Demo")

with st.sidebar:
    st.header("Chế độ")
    mode = st.radio(
        "Chọn chế độ",
        ["Tra nhanh (mô phỏng)", "Chấm điểm (upload/test)", "Quản trị (train)"],
        index=2
    )

    if st.session_state.train_result is None:
        st.info("Chưa có model – mô phỏng")
    else:
        tr = st.session_state.train_result
        st.success("Đã có model")
        st.caption(f"Task: {tr.task_type} | Target: {tr.target_col} | Cat cols: {len(tr.cat_cols)}")


if mode == "Tra nhanh (mô phỏng)":
    st.subheader("Tra nhanh (mô phỏng)")
    q = st.text_input("Nhập SĐT/CCCD/ID", value="")
    if st.button("Tra cứu"):
        if not q.strip():
            st.warning("Vui lòng nhập ID.")
        else:
            score = np.random.uniform(0, 1)
            st.metric("Điểm (mô phỏng)", f"{score:.3f}")
            st.write("Khuyến nghị:", "Duyệt" if score > 0.6 else "Cân nhắc" if score > 0.4 else "Từ chối")

elif mode == "Chấm điểm (upload/test)":
    st.subheader("Chấm điểm (upload/test)")
    if st.session_state.train_result is None:
        st.warning("Chưa có model. Vào **Quản trị (train)** để huấn luyện trước.")
    else:
        tr: TrainResult = st.session_state.train_result

        up = st.file_uploader("Upload CSV để chấm điểm (có thể nhiều dòng)", type=["csv"])
        if up is not None:
            df_test = read_csv_uploaded(up)
            st.write("Preview:", df_test.head())
            if st.button("Chấm điểm"):
                try:
                    pred, pred_label = predict_one(tr, df_test)
                    if pred_label is not None:
                        out = df_test.copy()
                        out["pred_class"] = pred
                        out["pred_label"] = pred_label
                        st.dataframe(out.head(50), use_container_width=True)
                    else:
                        out = df_test.copy()
                        out["pred"] = pred
                        st.dataframe(out.head(50), use_container_width=True)
                except Exception as e:
                    st.error("Lỗi chấm điểm:")
                    st.exception(e)

else:
    st.subheader("Quản trị (train)")
    st.caption("Upload TRAIN CSV → chọn target → chỉnh hyperparameters → Huấn luyện")

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
                max_train_rows = st.selectbox("Giới hạn số dòng train (để ổn định Cloud)", [20000, 30000, 40000, 60000, 100000], index=2)
                st.caption("Khuyến nghị: 30k–40k để demo ổn định.")

            with c2:
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
                n_jobs = st.selectbox("n_jobs", [1, 2, 4], index=0)  # default 1 for Cloud stability
                early_stopping_rounds = st.selectbox("early_stopping_rounds", [20, 50, 100], index=1)
            with c5:
                st.caption("Nếu Cloud hay lỗi: để n_jobs=1 và giảm n_estimators xuống 300–600.")

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
