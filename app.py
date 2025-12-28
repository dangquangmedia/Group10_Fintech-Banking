# app.py
# PB-025 / Fintech Banking - Credit Scoring Demo
# - Train: Logistic Regression + LightGBM
# - Report: AUC, KS, Brier, ECE + ROC / PR / Calibration plots
# - Score: Upload CSV -> predict risk prob + credit score + download results
#
# Notes for Streamlit Cloud:
# - st.session_state keeps model + UI after training
# - Use sampling to avoid heavy training on large datasets

from __future__ import annotations

import io
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    average_precision_score,
    precision_recall_curve,
    brier_score_loss,
    precision_score,
    recall_score,
    f1_score,
)

from sklearn.calibration import calibration_curve

import lightgbm as lgb
import matplotlib.pyplot as plt
import joblib


# ----------------------------
# Page config (MUST be first Streamlit call)
# ----------------------------
st.set_page_config(
    page_title="PB-025 / Fintech Banking - Credit Scoring Demo",
    page_icon="📊",
    layout="wide",
)

# ----------------------------
# Styling
# ----------------------------
CSS = """
<style>
/* Layout */
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1200px; }
h1, h2, h3 { letter-spacing: -0.02em; }
.small-muted { color: rgba(49,51,63,0.65); font-size: 0.92rem; }

/* Cards */
.card {
  border: 1px solid rgba(49,51,63,0.10);
  border-radius: 14px;
  padding: 16px 16px;
  background: white;
  box-shadow: 0 6px 18px rgba(0,0,0,0.04);
}
.card-title { font-weight: 700; font-size: 1.0rem; margin-bottom: 8px; }
hr.soft { border: 0; height: 1px; background: rgba(49,51,63,0.10); margin: 10px 0; }

/* Sidebar */
section[data-testid="stSidebar"] { border-right: 1px solid rgba(49,51,63,0.10); }

/* Buttons */
div.stButton > button {
  border-radius: 10px;
  padding: 0.5rem 1rem;
}

/* Metrics row */
.metric-row { display:flex; gap:10px; flex-wrap:wrap; }
.metric-box {
  border: 1px solid rgba(49,51,63,0.10);
  border-radius: 14px;
  padding: 12px 14px;
  min-width: 180px;
  background: white;
}
.metric-name { color: rgba(49,51,63,0.70); font-size: 0.85rem; }
.metric-val { font-weight: 800; font-size: 1.35rem; margin-top: 2px; }
.metric-sub { color: rgba(49,51,63,0.60); font-size: 0.82rem; margin-top: 2px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------
# Data structures
# ----------------------------
@dataclass
class TrainArtifacts:
    preprocessor: ColumnTransformer
    feature_names: List[str]
    logistic: LogisticRegression
    lgbm: lgb.LGBMClassifier
    task: str  # "classification"
    target_col: str
    bad_labels: List[str]
    id_hint: Optional[str] = None


@dataclass
class ModelMetrics:
    auc: float
    ap: float
    ks: float
    ks_thr: float
    youden_thr: float
    precision_at_ks: float
    recall_at_ks: float
    f1_at_ks: float
    brier: float
    ece: float


# ----------------------------
# Session state init
# ----------------------------
if "artifacts" not in st.session_state:
    st.session_state.artifacts = None  # type: ignore

if "last_train_report" not in st.session_state:
    st.session_state.last_train_report = None  # type: ignore

if "last_scored_df" not in st.session_state:
    st.session_state.last_scored_df = None  # type: ignore


# ----------------------------
# Utilities
# ----------------------------
def read_csv_safely(uploaded) -> pd.DataFrame:
    # low_memory=False reduces mixed dtype chunk inference issues
    return pd.read_csv(uploaded, low_memory=False)


def guess_id_column(cols: List[str]) -> Optional[str]:
    cset = [c.lower() for c in cols]
    candidates = ["id", "member_id", "customer_id", "cust_id", "ssn", "cccd", "sdt", "phone"]
    for cand in candidates:
        if cand in cset:
            return cols[cset.index(cand)]
    # fuzzy contains
    for i, c in enumerate(cset):
        if any(k in c for k in ["cccd", "phone", "sdt", "customer", "member", "client"]):
            return cols[i]
    return None


def to_binary_labels(y_raw: pd.Series, bad_labels: List[str]) -> np.ndarray:
    # Convert y into 0/1 with "bad" meaning 1
    y_str = y_raw.astype(str).fillna("")
    bad_set = set([str(x) for x in bad_labels])
    return y_str.isin(bad_set).astype(int).to_numpy()


def build_preprocessor(df: pd.DataFrame, target_col: str) -> Tuple[ColumnTransformer, List[str], List[str], List[str]]:
    X = df.drop(columns=[target_col])
    # Identify categorical vs numeric
    cat_cols = []
    num_cols = []

    for c in X.columns:
        if pd.api.types.is_bool_dtype(X[c]):
            num_cols.append(c)
        elif pd.api.types.is_numeric_dtype(X[c]):
            num_cols.append(c)
        else:
            # object/category => categorical
            cat_cols.append(c)

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    # feature names after fit (need fit first). We'll return col lists too.
    return preprocessor, num_cols, cat_cols, list(X.columns)


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    try:
        names = preprocessor.get_feature_names_out()
        return [str(x) for x in names]
    except Exception:
        # fallback
        return []


def ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    # Returns (ks, threshold_at_ks)
    # Compute CDF difference between positives and negatives
    order = np.argsort(y_prob)
    y_true_sorted = y_true[order]
    y_prob_sorted = y_prob[order]

    pos = (y_true_sorted == 1).astype(int)
    neg = (y_true_sorted == 0).astype(int)

    pos_cum = np.cumsum(pos) / max(pos.sum(), 1)
    neg_cum = np.cumsum(neg) / max(neg.sum(), 1)

    diff = np.abs(pos_cum - neg_cum)
    idx = int(np.argmax(diff))
    ks = float(diff[idx])
    thr = float(y_prob_sorted[idx])
    return ks, thr


def youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    idx = int(np.argmax(j))
    return float(thr[idx])


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = y_true.astype(int)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    ece = 0.0
    n = len(y_prob)

    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        conf = float(np.mean(y_prob[mask]))
        acc = float(np.mean(y_true[mask]))
        ece += (np.sum(mask) / n) * abs(acc - conf)

    return float(ece)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> ModelMetrics:
    auc = float(roc_auc_score(y_true, y_prob))
    ap = float(average_precision_score(y_true, y_prob))
    ks, ks_thr = ks_statistic(y_true, y_prob)
    y_thr_youden = youden_threshold(y_true, y_prob)

    y_pred_ks = (y_prob >= ks_thr).astype(int)
    precision = float(precision_score(y_true, y_pred_ks, zero_division=0))
    recall = float(recall_score(y_true, y_pred_ks, zero_division=0))
    f1 = float(f1_score(y_true, y_pred_ks, zero_division=0))

    brier = float(brier_score_loss(y_true, y_prob))
    ece = float(expected_calibration_error(y_true, y_prob, n_bins=10))

    return ModelMetrics(
        auc=auc,
        ap=ap,
        ks=float(ks),
        ks_thr=float(ks_thr),
        youden_thr=float(y_thr_youden),
        precision_at_ks=precision,
        recall_at_ks=recall,
        f1_at_ks=f1,
        brier=brier,
        ece=ece,
    )


def fig_roc_curves(y_true: np.ndarray, probs: Dict[str, np.ndarray]) -> plt.Figure:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # diagonal
    ax.plot([0, 1], [0, 1], linestyle="--")
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(y_true, p)
        auc = roc_auc_score(y_true, p)
        ax.plot(fpr, tpr, label=f"{name} AUC={auc:.4f}")
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def fig_pr_curves(y_true: np.ndarray, probs: Dict[str, np.ndarray]) -> plt.Figure:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for name, p in probs.items():
        precision, recall, _ = precision_recall_curve(y_true, p)
        ap = average_precision_score(y_true, p)
        ax.plot(recall, precision, label=f"{name} AP={ap:.4f}")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def fig_calibration(y_true: np.ndarray, probs: Dict[str, np.ndarray]) -> plt.Figure:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot([0, 1], [0, 1], linestyle="--")
    for name, p in probs.items():
        frac_pos, mean_pred = calibration_curve(y_true, p, n_bins=10, strategy="uniform")
        ax.plot(mean_pred, frac_pos, marker="o", label=name)
    ax.set_title("Calibration Plot")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def render_metric_cards(title: str, m: ModelMetrics):
    st.markdown(f'<div class="card"><div class="card-title">{title}</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-row">', unsafe_allow_html=True)

    def box(name, val, sub=""):
        st.markdown(
            f"""
            <div class="metric-box">
              <div class="metric-name">{name}</div>
              <div class="metric-val">{val}</div>
              <div class="metric-sub">{sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    box("AUC (ROC)", f"{m.auc:.4f}")
    box("KS", f"{m.ks:.4f}", f"thr={m.ks_thr:.4f}")
    box("Brier score", f"{m.brier:.4f}")
    box("ECE", f"{m.ece:.4f}")
    box("Avg Precision (AP)", f"{m.ap:.4f}")
    box("F1 @ KS thr", f"{m.f1_at_ks:.4f}", f"P={m.precision_at_ks:.3f} / R={m.recall_at_ks:.3f}")

    st.markdown("</div></div>", unsafe_allow_html=True)


def prob_to_credit_score(prob_bad: np.ndarray) -> np.ndarray:
    # Map risk probability -> a "score" range like 300..850
    # score higher = safer
    score = 300 + 550 * (1.0 - prob_bad)
    return np.clip(score, 300, 850)


def risk_bucket(prob_bad: np.ndarray) -> List[str]:
    # Simple bucketing
    out = []
    for p in prob_bad:
        if p < 0.15:
            out.append("Low risk")
        elif p < 0.35:
            out.append("Medium risk")
        else:
            out.append("High risk")
    return out


def export_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


# ----------------------------
# Header
# ----------------------------
st.title("PB-025 / Fintech Banking - Credit Scoring Demo")
st.markdown(
    '<div class="small-muted">Demo gồm 3 chế độ: Tra nhanh (mô phỏng) • Chấm điểm (upload/test) • Quản trị (train). '
    'Báo cáo hiển thị các chỉ số & biểu đồ thực nghiệm theo slide nhóm (AUC/KS/Brier/ECE + ROC/PR/Calibration).</div>',
    unsafe_allow_html=True,
)

# ----------------------------
# Sidebar: Mode selector
# ----------------------------
with st.sidebar:
    st.header("Chế độ")
    mode = st.radio(
        "Chọn chế độ",
        ["Tra nhanh (mô phỏng)", "Chấm điểm (upload/test)", "Quản trị (train)"],
        index=0,
    )

    st.markdown("---")

    artifacts: Optional[TrainArtifacts] = st.session_state.artifacts
    if artifacts is None:
        st.warning("Chưa có model. Hãy vào **Quản trị (train)** để huấn luyện.")
    else:
        st.success("Đã có model")
        st.caption(f"Task: {artifacts.task} | Target: {artifacts.target_col}")
        st.caption(f"Features: {len(artifacts.feature_names) if artifacts.feature_names else 'N/A'}")
        if artifacts.bad_labels:
            st.caption(f"Bad labels: {', '.join(artifacts.bad_labels[:6])}{'...' if len(artifacts.bad_labels)>6 else ''}")

    st.markdown("---")
    st.caption("Gợi ý: Streamlit Cloud yếu → dùng **Sampling** khi train.")


# ----------------------------
# MODE 1: Quick lookup (mock)
# ----------------------------
if mode == "Tra nhanh (mô phỏng)":
    st.subheader("Tra nhanh (mô phỏng)")
    colA, colB = st.columns([2, 1], gap="large")

    with colA:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Nhập SĐT/CCCD/ID</div>', unsafe_allow_html=True)
        q = st.text_input("Nhập SĐT/CCCD/ID", placeholder="Ví dụ: 090xxxxxxx / 0790xxxxxxx / CUS_00123", label_visibility="collapsed")
        btn = st.button("Tra cứu", type="primary")

        st.markdown('<hr class="soft"/>', unsafe_allow_html=True)

        if btn:
            # If we have last scored df and an ID column, try lookup; else simulate
            last_df: Optional[pd.DataFrame] = st.session_state.last_scored_df
            id_col = None
            if last_df is not None:
                id_col = guess_id_column(list(last_df.columns))
            if last_df is not None and id_col is not None and q.strip() != "":
                hit = last_df[last_df[id_col].astype(str) == q.strip()]
                if len(hit) == 0:
                    st.info("Không tìm thấy trong file chấm điểm gần nhất. (Đang mô phỏng kết quả.)")
                else:
                    row = hit.iloc[0].to_dict()
                    st.success(f"Tìm thấy: {id_col} = {q.strip()}")
                    st.json({k: row[k] for k in list(row.keys())[:18]})
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.stop()

            # simulate deterministically
            seed = abs(hash(q.strip())) % (10**6)
            rng = np.random.default_rng(seed)
            prob_bad = float(np.clip(rng.normal(0.22, 0.12), 0.01, 0.95))
            score = float(prob_to_credit_score(np.array([prob_bad]))[0])
            bucket = risk_bucket(np.array([prob_bad]))[0]

            st.success("Kết quả mô phỏng")
            st.write(f"- Xác suất rủi ro (bad/default): **{prob_bad:.3f}**")
            st.write(f"- Credit score (300–850): **{score:.0f}**")
            st.write(f"- Nhóm rủi ro: **{bucket}**")

        st.markdown("</div>", unsafe_allow_html=True)

    with colB:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Giải thích</div>', unsafe_allow_html=True)
        st.write("• Chế độ này dùng để demo trải nghiệm “tra cứu nhanh”.")
        st.write("• Nếu bạn đã **Chấm điểm** một file CSV trước đó, app sẽ thử tra cứu theo ID trong file đó.")
        st.write("• Nếu không tìm thấy, app sẽ **mô phỏng** kết quả (deterministic theo ID).")
        st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------
# MODE 2: Score CSV (upload/test)
# ----------------------------
elif mode == "Chấm điểm (upload/test)":
    st.subheader("Chấm điểm (upload/test)")

    artifacts = st.session_state.artifacts
    if artifacts is None:
        st.error("Chưa có model. Vào **Quản trị (train)** để huấn luyện trước.")
        st.stop()

    up = st.file_uploader("Upload CSV để chấm điểm (có thể nhiều dòng)", type=["csv"])
    if up is None:
        st.info("Tải lên file CSV để chấm điểm.")
        st.stop()

    df = read_csv_safely(up)
    st.caption(f"Shape: {df.shape[0]:,} dòng × {df.shape[1]:,} cột")
    st.dataframe(df.head(30), use_container_width=True)

    col1, col2, col3 = st.columns([1.2, 1.2, 1.0], gap="large")
    with col1:
        model_choice = st.selectbox("Chọn model", ["LightGBM", "Logistic Regression"], index=0)
    with col2:
        id_col_guess = guess_id_column(list(df.columns))
        id_col = st.selectbox("Cột định danh (tuỳ chọn)", ["(không có)"] + list(df.columns), index=(1 + list(df.columns).index(id_col_guess)) if id_col_guess in df.columns else 0)
        id_col = None if id_col == "(không có)" else id_col
    with col3:
        # optional label column exists? evaluate
        label_cols = ["(không có)"] + [c for c in df.columns]
        label_col = st.selectbox("Cột nhãn (tuỳ chọn để đánh giá)", label_cols, index=0)
        label_col = None if label_col == "(không có)" else label_col

    st.markdown("---")
    run = st.button("Chấm điểm", type="primary")

    if not run:
        st.stop()

    # Build X
    df_work = df.copy()
    y_true = None
    if label_col is not None:
        # Interpret label using training bad_labels mapping if same style
        y_true = to_binary_labels(df_work[label_col], artifacts.bad_labels)
        df_work = df_work.drop(columns=[label_col])

    # Ensure the feature columns are aligned: keep only columns seen in training (before preprocess)
    # We can't perfectly know raw columns lists from preprocessor easily, so use intersection:
    raw_cols = []
    try:
        # preprocessor transformers store columns
        for name, trans, cols in artifacts.preprocessor.transformers_:
            if isinstance(cols, list):
                raw_cols.extend(cols)
    except Exception:
        raw_cols = list(df_work.columns)

    keep = [c for c in raw_cols if c in df_work.columns]
    if len(keep) == 0:
        st.error("File chấm điểm không khớp cột so với lúc train (không có cột giao nhau).")
        st.stop()

    X = df_work[keep].copy()

    with st.spinner("Đang transform & predict..."):
        X_t = artifacts.preprocessor.transform(X)

        if model_choice == "LightGBM":
            prob_bad = artifacts.lgbm.predict_proba(X_t)[:, 1]
        else:
            prob_bad = artifacts.logistic.predict_proba(X_t)[:, 1]

        score = prob_to_credit_score(prob_bad)
        bucket = risk_bucket(prob_bad)

        out = df.copy()
        out["prob_bad"] = prob_bad
        out["credit_score_300_850"] = score.round(0).astype(int)
        out["risk_bucket"] = bucket

    # Save last scored df for quick lookup mode
    st.session_state.last_scored_df = out

    # Summary cards
    st.markdown("### Kết quả chấm điểm")
    cA, cB = st.columns([1.6, 1.0], gap="large")

    with cA:
        st.dataframe(out.head(200), use_container_width=True)

        st.download_button(
            "Tải kết quả CSV",
            data=export_csv_bytes(out),
            file_name="scored_results.csv",
            mime="text/csv",
        )

    with cB:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Tóm tắt</div>', unsafe_allow_html=True)
        st.write(f"Model: **{model_choice}**")
        st.write(f"Số dòng: **{len(out):,}**")
        st.write(f"Prob_bad trung bình: **{float(np.mean(prob_bad)):.3f}**")
        st.write(f"Credit score TB: **{float(np.mean(score)):.0f}**")

        # Distribution chart
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(score, bins=30)
        ax.set_title("Distribution of Credit Score (300–850)")
        ax.set_xlabel("score")
        ax.set_ylabel("count")
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # If label exists, show evaluation
    if y_true is not None:
        st.markdown("---")
        st.markdown("### Đánh giá trên file (vì có nhãn)")
        m = compute_metrics(y_true, prob_bad)

        render_metric_cards(f"Metrics ({model_choice})", m)

        probs_for_plot = {model_choice: prob_bad}
        st.pyplot(fig_roc_curves(y_true, probs_for_plot), clear_figure=True)
        st.pyplot(fig_pr_curves(y_true, probs_for_plot), clear_figure=True)
        st.pyplot(fig_calibration(y_true, probs_for_plot), clear_figure=True)


# ----------------------------
# MODE 3: Train
# ----------------------------
else:
    st.subheader("Quản trị (train)")

    up = st.file_uploader("Upload TRAIN CSV", type=["csv"])
    if up is None:
        st.info("Tải lên file train CSV để huấn luyện.")
        st.stop()

    df_train = read_csv_safely(up)
    st.caption(f"Shape: {df_train.shape[0]:,} dòng × {df_train.shape[1]:,} cột")
    st.dataframe(df_train.head(30), use_container_width=True)

    # Choose target
    cols = list(df_train.columns)
    default_target = "Credit_Score" if "Credit_Score" in cols else cols[-1]
    target_col = st.selectbox("Chọn cột target (nhãn)", cols, index=cols.index(default_target) if default_target in cols else 0)

    # Bad labels selection
    y_unique = sorted([str(x) for x in df_train[target_col].dropna().unique()])[:200]
    st.caption(f"Số nhãn (unique) xem trước: {min(len(y_unique),200)}")
    bad_labels = st.multiselect(
        "Chọn nhãn được xem là RỦI RO / DEFAULT (bad = 1). Ví dụ: '1', 'bad', 'Charged Off'...",
        options=y_unique,
        default=[x for x in y_unique if x.lower() in ["1", "bad", "default", "charged off", "charged_off", "true"]],
    )

    st.markdown("---")
    colL, colR = st.columns([1.2, 1.0], gap="large")

    with colL:
        st.markdown("#### Thiết lập train/test")
        test_size = st.slider("Tỉ lệ test", 0.1, 0.4, 0.2, 0.05)
        random_state = st.number_input("random_state", min_value=0, max_value=9999, value=42, step=1)

        use_sampling = st.checkbox("Bật sampling để train nhanh (khuyên dùng trên Streamlit Cloud)", value=True)
        sample_n = st.number_input("Số dòng sample (nếu bật)", min_value=5000, max_value=int(min(300000, len(df_train))), value=int(min(100000, len(df_train))), step=5000)

    with colR:
        st.markdown("#### Hyperparameters (LightGBM)")
        n_estimators = st.slider("n_estimators", 100, 2000, 600, 50)
        learning_rate = st.slider("learning_rate", 0.01, 0.2, 0.05, 0.01)
        num_leaves = st.slider("num_leaves", 15, 255, 63, 2)
        max_depth = st.slider("max_depth (-1 = không giới hạn)", -1, 20, -1, 1)
        subsample = st.slider("subsample", 0.5, 1.0, 0.9, 0.05)
        colsample_bytree = st.slider("colsample_bytree", 0.5, 1.0, 0.9, 0.05)
        min_child_samples = st.slider("min_child_samples", 5, 100, 20, 1)
        reg_lambda = st.slider("reg_lambda", 0.0, 5.0, 0.0, 0.1)
        n_jobs = st.selectbox("n_jobs (Cloud chậm → để 1-2)", [1, 2, 4], index=1)
        early_stopping_rounds = st.selectbox("early_stopping_rounds", [20, 50, 100, 200], index=1)

    st.markdown("---")
    run_train = st.button("Huấn luyện", type="primary")

    if not run_train:
        # show last report if exists
        if st.session_state.last_train_report is not None:
            st.markdown("### Báo cáo lần huấn luyện gần nhất")
            rep = st.session_state.last_train_report
            render_metric_cards("Logistic Regression (baseline)", rep["logistic_metrics"])
            render_metric_cards("LightGBM (mở rộng)", rep["lgbm_metrics"])
            st.pyplot(rep["fig_roc"], clear_figure=True)
            st.pyplot(rep["fig_pr"], clear_figure=True)
            st.pyplot(rep["fig_cal"], clear_figure=True)
        st.stop()

    # Training pipeline
    with st.spinner("Đang chuẩn bị dữ liệu..."):
        dfw = df_train.copy()

        if use_sampling and len(dfw) > sample_n:
            dfw = dfw.sample(n=int(sample_n), random_state=int(random_state))

        # build y
        if len(bad_labels) == 0:
            st.error("Bạn chưa chọn nhãn 'bad'. Hãy chọn ít nhất 1 giá trị để map bad=1.")
            st.stop()

        y = to_binary_labels(dfw[target_col], bad_labels)
        X_df = dfw.drop(columns=[target_col]).copy()

        # Preprocessor
        preprocessor, num_cols, cat_cols, _ = build_preprocessor(dfw, target_col)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_df,
            y,
            test_size=float(test_size),
            random_state=int(random_state),
            stratify=y if len(np.unique(y)) == 2 else None,
        )

    with st.spinner("Đang fit preprocessor..."):
        X_train_t = preprocessor.fit_transform(X_train)
        X_test_t = preprocessor.transform(X_test)
        feat_names = get_feature_names(preprocessor)

    # Logistic Regression
    with st.spinner("Đang train Logistic Regression (baseline)..."):
        # StandardScaler for sparse: with_mean=False
        scaler = StandardScaler(with_mean=False)
        X_train_lr = scaler.fit_transform(X_train_t)
        X_test_lr = scaler.transform(X_test_t)

        lr = LogisticRegression(
            max_iter=600,
            solver="saga",
            n_jobs=int(n_jobs),
            class_weight="balanced",
        )
        lr.fit(X_train_lr, y_train)
        prob_lr = lr.predict_proba(X_test_lr)[:, 1]
        m_lr = compute_metrics(y_test, prob_lr)

    # LightGBM
    with st.spinner("Đang train LightGBM (mở rộng)..."):
        # scale_pos_weight for imbalance
        pos = max(int(np.sum(y_train == 1)), 1)
        neg = max(int(np.sum(y_train == 0)), 1)
        spw = neg / pos

        lgbm = lgb.LGBMClassifier(
            n_estimators=int(n_estimators),
            learning_rate=float(learning_rate),
            num_leaves=int(num_leaves),
            max_depth=int(max_depth),
            subsample=float(subsample),
            colsample_bytree=float(colsample_bytree),
            min_child_samples=int(min_child_samples),
            reg_lambda=float(reg_lambda),
            n_jobs=int(n_jobs),
            objective="binary",
            scale_pos_weight=float(spw),
            random_state=int(random_state),
        )

        # early stopping
        lgbm.fit(
            X_train_t,
            y_train,
            eval_set=[(X_test_t, y_test)],
            eval_metric="auc",
            callbacks=[
                lgb.early_stopping(stopping_rounds=int(early_stopping_rounds), verbose=False)
            ],
        )
        prob_lgb = lgbm.predict_proba(X_test_t)[:, 1]
        m_lgb = compute_metrics(y_test, prob_lgb)

    # Save artifacts in session state (keep UI after train)
    artifacts = TrainArtifacts(
        preprocessor=preprocessor,
        feature_names=feat_names,
        logistic=lr,
        lgbm=lgbm,
        task="classification",
        target_col=str(target_col),
        bad_labels=[str(x) for x in bad_labels],
        id_hint=guess_id_column(list(df_train.columns)),
    )
    st.session_state.artifacts = artifacts

    # Figures
    probs = {"Logistic": prob_lr, "LightGBM": prob_lgb}
    fig_roc = fig_roc_curves(y_test, probs)
    fig_pr = fig_pr_curves(y_test, probs)
    fig_cal = fig_calibration(y_test, probs)

    st.session_state.last_train_report = {
        "logistic_metrics": m_lr,
        "lgbm_metrics": m_lgb,
        "fig_roc": fig_roc,
        "fig_pr": fig_pr,
        "fig_cal": fig_cal,
    }

    # Render report
    st.success("Huấn luyện xong ✅ (Model đã được lưu trong session, UI không bị mất)")
    st.markdown("### Báo cáo thực nghiệm")

    render_metric_cards("Logistic Regression (baseline)", m_lr)
    render_metric_cards("LightGBM (mở rộng)", m_lgb)

    st.pyplot(fig_roc, clear_figure=True)
    st.pyplot(fig_pr, clear_figure=True)
    st.pyplot(fig_cal, clear_figure=True)

    st.markdown("---")
    st.markdown("### Xuất model (tuỳ chọn)")
    c1, c2 = st.columns([1, 1], gap="large")

    with c1:
        # Package for download
        buf = io.BytesIO()
        joblib.dump(
            {
                "preprocessor": artifacts.preprocessor,
                "scaler_for_lr": scaler,
                "logistic": artifacts.logistic,
                "lgbm": artifacts.lgbm,
                "target_col": artifacts.target_col,
                "bad_labels": artifacts.bad_labels,
                "feature_names": artifacts.feature_names,
            },
            buf,
        )
        st.download_button(
            "Tải model bundle (.joblib)",
            data=buf.getvalue(),
            file_name="pb025_models_bundle.joblib",
            mime="application/octet-stream",
        )

    with c2:
        st.write("Nếu app bị restart trên Streamlit Cloud, session_state mất.")
        st.write("Khi đó bạn có thể upload lại model bundle và load (nếu cần mình thêm tab Load model).")
