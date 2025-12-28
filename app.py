# app.py
# PB-025 / Fintech Banking - Credit Scoring Demo (CIC-style UI)
# - Train: Logistic Regression + LightGBM
# - Persist: auto-save model+report to ./artifacts (joblib + json)
# - Score: upload CSV -> prob_bad + score(300-850) + risk bucket + percentile rank
# - Charts: ROC / PR / Calibration + CIC gauge + distribution
# - SHAP: optional, limited samples (safe for Streamlit Cloud)

from __future__ import annotations

import io
import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt

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
import joblib

# Optional SHAP (may be heavy)
try:
    import shap  # type: ignore
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False


# ----------------------------
# Page config (must be first)
# ----------------------------
st.set_page_config(
    page_title="PB-025 / Fintech Banking - Credit Scoring Demo",
    page_icon="📊",
    layout="wide",
)

# ----------------------------
# Paths for persistence
# ----------------------------
ART_DIR = "artifacts"
os.makedirs(ART_DIR, exist_ok=True)

MODEL_BUNDLE_PATH = os.path.join(ART_DIR, "pb025_models_bundle.joblib")
REPORT_PATH = os.path.join(ART_DIR, "pb025_last_report.json")
SCORE_REF_PATH = os.path.join(ART_DIR, "pb025_score_ref.npy")  # store reference score distribution


# ----------------------------
# UI styling
# ----------------------------
CSS = """
<style>
.block-container { padding-top: 1.3rem; padding-bottom: 2rem; max-width: 1250px; }
h1, h2, h3 { letter-spacing: -0.02em; }
.small-muted { color: rgba(49,51,63,0.65); font-size: 0.92rem; }

.card {
  border: 1px solid rgba(49,51,63,0.10);
  border-radius: 14px;
  padding: 16px 16px;
  background: white;
  box-shadow: 0 6px 18px rgba(0,0,0,0.04);
}
.card-title { font-weight: 800; font-size: 1.02rem; margin-bottom: 6px; }
hr.soft { border: 0; height: 1px; background: rgba(49,51,63,0.10); margin: 10px 0; }

.metric-row { display:flex; gap:10px; flex-wrap:wrap; }
.metric-box {
  border: 1px solid rgba(49,51,63,0.10);
  border-radius: 14px;
  padding: 12px 14px;
  min-width: 180px;
  background: white;
}
.metric-name { color: rgba(49,51,63,0.70); font-size: 0.85rem; }
.metric-val { font-weight: 900; font-size: 1.35rem; margin-top: 2px; }
.metric-sub { color: rgba(49,51,63,0.60); font-size: 0.82rem; margin-top: 2px; }

.badge {
  display:inline-block; padding: 2px 10px; border-radius: 999px;
  border: 1px solid rgba(49,51,63,0.15);
  font-size: 0.82rem;
  margin-right: 6px;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------
# Data structures
# ----------------------------
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


@dataclass
class TrainArtifacts:
    task: str
    target_col: str
    bad_labels: List[str]
    primary_model: str  # "LightGBM" or "Logistic"
    feature_count: int
    id_hint: Optional[str] = None


# ----------------------------
# Session init
# ----------------------------
if "bundle" not in st.session_state:
    st.session_state.bundle = None  # dict holding actual models/objects

if "artifacts" not in st.session_state:
    st.session_state.artifacts = None  # TrainArtifacts

if "last_report" not in st.session_state:
    st.session_state.last_report = None  # report dict (metrics + serialized)

if "last_scored_df" not in st.session_state:
    st.session_state.last_scored_df = None


# ----------------------------
# Helpers
# ----------------------------
def read_csv_safely(uploaded) -> pd.DataFrame:
    return pd.read_csv(uploaded, low_memory=False)


def guess_id_column(cols: List[str]) -> Optional[str]:
    cset = [c.lower() for c in cols]
    candidates = ["id", "member_id", "customer_id", "cust_id", "cccd", "sdt", "phone"]
    for cand in candidates:
        if cand in cset:
            return cols[cset.index(cand)]
    for i, c in enumerate(cset):
        if any(k in c for k in ["cccd", "phone", "sdt", "customer", "member", "client"]):
            return cols[i]
    return None


def to_binary_labels(y_raw: pd.Series, bad_labels: List[str]) -> np.ndarray:
    y_str = y_raw.astype(str).fillna("")
    bad_set = set([str(x) for x in bad_labels])
    return y_str.isin(bad_set).astype(int).to_numpy()


def build_preprocessor(df: pd.DataFrame, target_col: str, min_freq: int = 10) -> Tuple[ColumnTransformer, List[str], List[str]]:
    X = df.drop(columns=[target_col])

    cat_cols, num_cols = [], []
    for c in X.columns:
        if pd.api.types.is_bool_dtype(X[c]) or pd.api.types.is_numeric_dtype(X[c]):
            num_cols.append(c)
        else:
            cat_cols.append(c)

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    # IMPORTANT: reduce exploding one-hot features
    # handle_unknown="infrequent_if_exist" + min_frequency helps keep feature count manageable
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=min_freq,
            sparse_output=True
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    return preprocessor, num_cols, cat_cols


def ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    order = np.argsort(y_prob)
    y_true_sorted = y_true[order]
    y_prob_sorted = y_prob[order]

    pos = (y_true_sorted == 1).astype(int)
    neg = (y_true_sorted == 0).astype(int)

    pos_cum = np.cumsum(pos) / max(pos.sum(), 1)
    neg_cum = np.cumsum(neg) / max(neg.sum(), 1)

    diff = np.abs(pos_cum - neg_cum)
    idx = int(np.argmax(diff))
    return float(diff[idx]), float(y_prob_sorted[idx])


def youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = y_true.astype(int)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    ece, n = 0.0, len(y_prob)

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
        auc=auc, ap=ap, ks=ks, ks_thr=ks_thr, youden_thr=y_thr_youden,
        precision_at_ks=precision, recall_at_ks=recall, f1_at_ks=f1,
        brier=brier, ece=ece
    )


def fig_roc_curves(y_true: np.ndarray, probs: Dict[str, np.ndarray]) -> plt.Figure:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
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
    ax.set_title("Precision–Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def fig_calibration(y_true: np.ndarray, probs: Dict[str, np.ndarray]) -> plt.Figure:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
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
    box("Brier", f"{m.brier:.4f}")
    box("ECE", f"{m.ece:.4f}")
    box("AP", f"{m.ap:.4f}")
    box("F1 @ KS", f"{m.f1_at_ks:.4f}", f"P={m.precision_at_ks:.3f} / R={m.recall_at_ks:.3f}")

    st.markdown("</div></div>", unsafe_allow_html=True)


def prob_to_credit_score(prob_bad: np.ndarray) -> np.ndarray:
    score = 300 + 550 * (1.0 - prob_bad)
    return np.clip(score, 300, 850)


def cic_bucket(score: float) -> Tuple[str, str]:
    # CIC-like tiers (you can adjust thresholds)
    if score >= 750:
        return "A - Rất tốt", "#16a34a"
    if score >= 680:
        return "B - Tốt", "#22c55e"
    if score >= 600:
        return "C - Trung bình", "#f59e0b"
    if score >= 520:
        return "D - Yếu", "#f97316"
    return "E - Rủi ro cao", "#ef4444"


def fig_gauge(score: float) -> plt.Figure:
    # Semi gauge (300..850)
    vmin, vmax = 300, 850
    value = float(np.clip(score, vmin, vmax))
    pct = (value - vmin) / (vmax - vmin)

    fig = plt.figure(figsize=(5.2, 2.8))
    ax = fig.add_subplot(111)

    # Draw arcs zones
    zones = [
        (300, 520, "#ef4444"),
        (520, 600, "#f97316"),
        (600, 680, "#f59e0b"),
        (680, 750, "#22c55e"),
        (750, 850, "#16a34a"),
    ]

    for a, b, color in zones:
        start = (a - vmin) / (vmax - vmin) * np.pi
        end = (b - vmin) / (vmax - vmin) * np.pi
        theta = np.linspace(start, end, 200)
        ax.plot(np.cos(theta), np.sin(theta), linewidth=18, color=color, solid_capstyle="round")

    # Needle
    theta_v = pct * np.pi
    ax.plot([0, np.cos(theta_v)], [0, np.sin(theta_v)], linewidth=3, color="black")

    ax.text(0, -0.15, f"Score: {value:.0f}", ha="center", va="center", fontsize=15, fontweight="bold")
    label, _ = cic_bucket(value)
    ax.text(0, -0.30, label, ha="center", va="center", fontsize=11)

    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    return fig


def percentile_rank(score: np.ndarray, ref: np.ndarray) -> np.ndarray:
    # percentile (0..100) vs reference distribution
    ref_sorted = np.sort(ref)
    ranks = np.searchsorted(ref_sorted, score, side="right") / len(ref_sorted) * 100.0
    return np.clip(ranks, 0, 100)


def save_bundle(bundle: dict, artifacts: TrainArtifacts, report: dict, score_ref: np.ndarray):
    joblib.dump(bundle, MODEL_BUNDLE_PATH)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump({"artifacts": asdict(artifacts), "report": report}, f, ensure_ascii=False, indent=2)
    np.save(SCORE_REF_PATH, score_ref)


def load_bundle_if_exists() -> bool:
    if os.path.exists(MODEL_BUNDLE_PATH) and os.path.exists(REPORT_PATH) and os.path.exists(SCORE_REF_PATH):
        try:
            bundle = joblib.load(MODEL_BUNDLE_PATH)
            with open(REPORT_PATH, "r", encoding="utf-8") as f:
                obj = json.load(f)
            artifacts = TrainArtifacts(**obj["artifacts"])
            report = obj["report"]
            score_ref = np.load(SCORE_REF_PATH)

            st.session_state.bundle = bundle
            st.session_state.artifacts = artifacts
            st.session_state.last_report = report
            st.session_state.score_ref = score_ref
            return True
        except Exception:
            return False
    return False


# Auto load on startup
if "score_ref" not in st.session_state:
    st.session_state.score_ref = None

if st.session_state.bundle is None:
    load_bundle_if_exists()


# ----------------------------
# Header
# ----------------------------
st.title("PB-025 / Fintech Banking - Credit Scoring Demo")
st.markdown(
    '<div class="small-muted">3 chế độ: Tra nhanh (mô phỏng) • Chấm điểm (upload/test) • Quản trị (train). '
    'Báo cáo: AUC/KS/Brier/ECE + ROC/PR/Calibration. UI chấm điểm theo phong cách CIC (score + gauge + tier).</div>',
    unsafe_allow_html=True,
)

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.header("Chế độ")
    mode = st.radio("Chọn chế độ", ["Tra nhanh (mô phỏng)", "Chấm điểm (upload/test)", "Quản trị (train)"], index=1)

    st.markdown("---")
    artifacts: Optional[TrainArtifacts] = st.session_state.artifacts
    if artifacts is None:
        st.warning("Chưa có model. Vào **Quản trị (train)** để huấn luyện.")
    else:
        st.success("Đã có model")
        st.markdown(f'<span class="badge">Primary: <b>{artifacts.primary_model}</b></span>', unsafe_allow_html=True)
        st.markdown(f'<span class="badge">Target: <b>{artifacts.target_col}</b></span>', unsafe_allow_html=True)
        st.markdown(f'<span class="badge">Task: <b>{artifacts.task}</b></span>', unsafe_allow_html=True)
        st.caption(f"Features (after encode): {artifacts.feature_count:,}")
        if artifacts.bad_labels:
            st.caption(f"Bad labels: {', '.join(artifacts.bad_labels[:6])}{'...' if len(artifacts.bad_labels)>6 else ''}")

    st.markdown("---")
    st.caption("Gợi ý: One-hot dễ bùng feature ⇒ dùng min_frequency để giảm. Train xong sẽ tự lưu vào artifacts/.")


# ----------------------------
# MODE: Quick lookup
# ----------------------------
if mode == "Tra nhanh (mô phỏng)":
    st.subheader("Tra nhanh (mô phỏng)")
    colA, colB = st.columns([2, 1], gap="large")

    with colA:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Nhập SĐT/CCCD/ID</div>', unsafe_allow_html=True)
        q = st.text_input("Nhập", placeholder="Ví dụ: 090xxxxxxx / 0790xxxxxxx / CUS_00123", label_visibility="collapsed")
        btn = st.button("Tra cứu", type="primary")
        st.markdown('<hr class="soft"/>', unsafe_allow_html=True)

        if btn:
            last_df: Optional[pd.DataFrame] = st.session_state.last_scored_df
            id_col = guess_id_column(list(last_df.columns)) if last_df is not None else None

            if last_df is not None and id_col is not None and q.strip():
                hit = last_df[last_df[id_col].astype(str) == q.strip()]
                if len(hit) > 0:
                    row = hit.iloc[0]
                    score = float(row["credit_score_300_850"])
                    st.pyplot(fig_gauge(score), clear_figure=True)
                    st.write("**Thông tin:**")
                    st.json(row.to_dict())
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.stop()

            # simulate
            seed = abs(hash(q.strip())) % (10**6)
            rng = np.random.default_rng(seed)
            prob_bad = float(np.clip(rng.normal(0.22, 0.12), 0.01, 0.95))
            score = float(prob_to_credit_score(np.array([prob_bad]))[0])
            st.pyplot(fig_gauge(score), clear_figure=True)
            st.write(f"- Xác suất rủi ro (bad): **{prob_bad:.3f}**")
            st.write(f"- Credit score: **{score:.0f}**")
        st.markdown("</div>", unsafe_allow_html=True)

    with colB:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Ghi chú</div>', unsafe_allow_html=True)
        st.write("• Nếu đã chấm điểm 1 file trước đó, app sẽ tra cứu theo ID.")
        st.write("• Nếu không có dữ liệu, app mô phỏng kết quả theo ID.")
        st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------
# MODE: Scoring
# ----------------------------
elif mode == "Chấm điểm (upload/test)":
    st.subheader("Chấm điểm (upload/test)")

    artifacts = st.session_state.artifacts
    bundle = st.session_state.bundle
    score_ref = st.session_state.score_ref

    if artifacts is None or bundle is None:
        st.error("Chưa có model. Vào **Quản trị (train)** để huấn luyện.")
        st.stop()

    up = st.file_uploader("Upload CSV để chấm điểm (nhiều dòng)", type=["csv"])
    if up is None:
        st.info("Tải lên file CSV để chấm điểm.")
        st.stop()

    df = read_csv_safely(up)
    st.caption(f"Shape: {df.shape[0]:,} dòng × {df.shape[1]:,} cột")
    st.dataframe(df.head(30), use_container_width=True)

    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.0, 1.0], gap="large")
    with c1:
        model_choice = st.selectbox(
            "Chọn model",
            ["LightGBM", "Logistic"],
            index=0 if artifacts.primary_model == "LightGBM" else 1
        )
    with c2:
        id_col_guess = guess_id_column(list(df.columns))
        id_col = st.selectbox("Cột định danh (tuỳ chọn)", ["(không có)"] + list(df.columns),
                              index=(1 + list(df.columns).index(id_col_guess)) if id_col_guess in df.columns else 0)
        id_col = None if id_col == "(không có)" else id_col
    with c3:
        enable_shap = st.checkbox("Bật SHAP (có thể chậm)", value=False, disabled=not _HAS_SHAP)
    with c4:
        shap_n = st.number_input("Giới hạn mẫu SHAP", min_value=50, max_value=500, value=200, step=50)

    st.markdown("---")
    run = st.button("Chấm điểm", type="primary")

    if not run:
        st.stop()

    preprocessor = bundle["preprocessor"]
    scaler_for_lr = bundle["scaler_for_lr"]
    lr = bundle["logistic"]
    lgbm = bundle["lgbm"]

    # Align columns: use columns stored in transformers
    raw_cols = []
    try:
        for _, _, cols in preprocessor.transformers_:
            if isinstance(cols, list):
                raw_cols.extend(cols)
    except Exception:
        raw_cols = list(df.columns)

    keep = [c for c in raw_cols if c in df.columns]
    if len(keep) == 0:
        st.error("File chấm điểm không khớp cột so với lúc train (không có cột giao nhau).")
        st.stop()

    X = df[keep].copy()

    with st.spinner("Đang transform & predict..."):
        Xt = preprocessor.transform(X)

        if model_choice == "LightGBM":
            prob_bad = lgbm.predict_proba(Xt)[:, 1]
        else:
            Xt_lr = scaler_for_lr.transform(Xt)
            prob_bad = lr.predict_proba(Xt_lr)[:, 1]

        score = prob_to_credit_score(prob_bad)
        tier = [cic_bucket(s)[0] for s in score]

        # percentile rank
        if score_ref is not None and len(score_ref) > 10:
            pr = percentile_rank(score, score_ref)
        else:
            pr = np.full_like(score, np.nan, dtype=float)

        out = df.copy()
        out["prob_bad"] = prob_bad
        out["credit_score_300_850"] = score.round(0).astype(int)
        out["cic_tier"] = tier
        out["rank_percentile"] = np.round(pr, 1)

    st.session_state.last_scored_df = out

    st.markdown("### Kết quả chấm điểm (CIC-style)")
    left, right = st.columns([1.55, 1.0], gap="large")

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Tóm tắt & mô phỏng</div>', unsafe_allow_html=True)

        # show first row gauge
        s0 = float(out["credit_score_300_850"].iloc[0])
        st.pyplot(fig_gauge(s0), clear_figure=True)

        st.write(f"Model: **{model_choice}**")
        st.write(f"Số dòng: **{len(out):,}**")
        st.write(f"Prob_bad TB: **{float(np.mean(prob_bad)):.3f}**")
        st.write(f"Score TB: **{float(np.mean(score)):.0f}**")

        # distribution
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(score, bins=30, color="#93c5fd")
        ax.set_title("Phân phối Credit Score (300–850)")
        ax.set_xlabel("score")
        ax.set_ylabel("count")
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with left:
        st.dataframe(out.head(300), use_container_width=True)
        st.download_button(
            "Tải kết quả CSV",
            data=out.to_csv(index=False).encode("utf-8-sig"),
            file_name="scored_results.csv",
            mime="text/csv",
        )

    # SHAP (optional)
    if enable_shap:
        if not _HAS_SHAP:
            st.warning("SHAP chưa sẵn sàng trong môi trường.")
        else:
            st.markdown("---")
            st.markdown("### SHAP (giải thích mô hình) – giới hạn mẫu để tránh sập")

            n = int(min(shap_n, len(X)))
            sample_idx = np.random.default_rng(42).choice(len(X), size=n, replace=False)
            Xs = X.iloc[sample_idx]
            Xst = preprocessor.transform(Xs)

            if model_choice != "LightGBM":
                st.info("SHAP demo chỉ bật tốt nhất cho LightGBM (TreeExplainer).")
            else:
                with st.spinner("Đang tính SHAP (TreeExplainer)..."):
                    explainer = shap.TreeExplainer(lgbm)
                    # Convert tiny sample to dense to avoid sparse issues
                    X_dense = Xst.toarray() if hasattr(Xst, "toarray") else np.asarray(Xst)
                    shap_values = explainer.shap_values(X_dense)

                # Plot summary (bar)
                st.caption("Top features ảnh hưởng (bar plot)")
                fig = plt.figure()
                shap.summary_plot(shap_values, X_dense, show=False, plot_type="bar")
                st.pyplot(fig, clear_figure=True)


# ----------------------------
# MODE: Admin / Train
# ----------------------------
else:
    st.subheader("Quản trị (train)")

    up = st.file_uploader("Upload TRAIN CSV", type=["csv"])
    if up is None:
        st.info("Tải lên file train CSV để huấn luyện.")
        # if existing report, show it
        if st.session_state.last_report is not None and st.session_state.artifacts is not None:
            st.markdown("### Báo cáo gần nhất (đã lưu)")
            rep = st.session_state.last_report
            render_metric_cards("Logistic Regression (baseline)", ModelMetrics(**rep["logistic_metrics"]))
            render_metric_cards("LightGBM (mở rộng)", ModelMetrics(**rep["lgbm_metrics"]))
        st.stop()

    df_train = read_csv_safely(up)
    st.caption(f"Shape: {df_train.shape[0]:,} dòng × {df_train.shape[1]:,} cột")
    st.dataframe(df_train.head(20), use_container_width=True)

    cols = list(df_train.columns)
    default_target = "Credit_Score" if "Credit_Score" in cols else cols[-1]
    target_col = st.selectbox("Chọn cột target (nhãn)", cols, index=cols.index(default_target) if default_target in cols else 0)

    # bad labels
    y_unique = sorted([str(x) for x in df_train[target_col].dropna().unique()])[:300]
    bad_labels = st.multiselect(
        "Chọn nhãn được xem là RỦI RO / DEFAULT (bad=1). Ví dụ: Poor / bad / 1 ...",
        options=y_unique,
        default=[x for x in y_unique if x.lower() in ["poor", "bad", "default", "1", "charged off", "charged_off", "true"]],
    )

    st.markdown("---")
    L, R = st.columns([1.2, 1.0], gap="large")

    with L:
        st.markdown("#### Thiết lập train")
        primary_model = st.selectbox("Primary model (mặc định để chấm điểm)", ["LightGBM", "Logistic"], index=0)
        test_size = st.slider("Tỉ lệ test", 0.1, 0.4, 0.2, 0.05)
        random_state = st.number_input("random_state", min_value=0, max_value=9999, value=42, step=1)

        use_sampling = st.checkbox("Bật sampling để train nhanh (khuyên dùng trên Streamlit Cloud)", value=True)
        sample_n = st.number_input("Số dòng sample", min_value=5000, max_value=int(min(300000, len(df_train))),
                                   value=int(min(100000, len(df_train))), step=5000)

        min_frequency = st.number_input("min_frequency (giảm bùng One-hot)", min_value=2, max_value=200, value=10, step=1)

    with R:
        st.markdown("#### Hyperparameters (LightGBM)")
        n_estimators = st.slider("n_estimators", 100, 2000, 600, 50)
        learning_rate = st.slider("learning_rate", 0.01, 0.2, 0.05, 0.01)
        num_leaves = st.slider("num_leaves", 15, 255, 63, 2)
        max_depth = st.slider("max_depth (-1 = không giới hạn)", -1, 20, -1, 1)
        subsample = st.slider("subsample", 0.5, 1.0, 0.9, 0.05)
        colsample_bytree = st.slider("colsample_bytree", 0.5, 1.0, 0.9, 0.05)
        min_child_samples = st.slider("min_child_samples", 5, 100, 20, 1)
        reg_lambda = st.slider("reg_lambda", 0.0, 5.0, 0.0, 0.1)
        n_jobs = st.selectbox("n_jobs (Cloud chậm → 1-2)", [1, 2, 4], index=1)
        early_stopping_rounds = st.selectbox("early_stopping_rounds", [20, 50, 100, 200], index=1)

    st.markdown("---")
    run_train = st.button("Huấn luyện", type="primary")

    if not run_train:
        # show persisted report if exists
        if st.session_state.last_report is not None and st.session_state.artifacts is not None:
            st.markdown("### Báo cáo gần nhất (đã lưu)")
            rep = st.session_state.last_report
            render_metric_cards("Logistic Regression (baseline)", ModelMetrics(**rep["logistic_metrics"]))
            render_metric_cards("LightGBM (mở rộng)", ModelMetrics(**rep["lgbm_metrics"]))
        st.stop()

    if len(bad_labels) == 0:
        st.error("Bạn chưa chọn nhãn 'bad'. Hãy chọn ít nhất 1 giá trị để map bad=1.")
        st.stop()

    # Prepare data
    dfw = df_train.copy()
    if use_sampling and len(dfw) > sample_n:
        dfw = dfw.sample(n=int(sample_n), random_state=int(random_state))

    y = to_binary_labels(dfw[target_col], bad_labels)
    X_df = dfw.drop(columns=[target_col]).copy()

    with st.spinner("Đang fit preprocessor..."):
        preprocessor, _, _ = build_preprocessor(dfw, target_col, min_freq=int(min_frequency))
        X_train, X_test, y_train, y_test = train_test_split(
            X_df,
            y,
            test_size=float(test_size),
            random_state=int(random_state),
            stratify=y if len(np.unique(y)) == 2 else None,
        )
        Xt_train = preprocessor.fit_transform(X_train)
        Xt_test = preprocessor.transform(X_test)

        # estimate feature count
        try:
            feat_count = int(Xt_train.shape[1])
        except Exception:
            feat_count = 0

    # Logistic
    with st.spinner("Train Logistic Regression..."):
        scaler = StandardScaler(with_mean=False)
        Xt_train_lr = scaler.fit_transform(Xt_train)
        Xt_test_lr = scaler.transform(Xt_test)

        lr = LogisticRegression(
            max_iter=700,
            solver="saga",
            n_jobs=int(n_jobs),
            class_weight="balanced",
        )
        lr.fit(Xt_train_lr, y_train)
        prob_lr = lr.predict_proba(Xt_test_lr)[:, 1]
        m_lr = compute_metrics(y_test, prob_lr)

    # LightGBM
    with st.spinner("Train LightGBM..."):
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

        lgbm.fit(
            Xt_train,
            y_train,
            eval_set=[(Xt_test, y_test)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(stopping_rounds=int(early_stopping_rounds), verbose=False)],
        )
        prob_lgb = lgbm.predict_proba(Xt_test)[:, 1]
        m_lgb = compute_metrics(y_test, prob_lgb)

    # Reference distribution for ranking (use test scores from primary model)
    if primary_model == "LightGBM":
        score_ref = prob_to_credit_score(prob_lgb)
    else:
        score_ref = prob_to_credit_score(prob_lr)

    # Build report + figures
    probs = {"Logistic": prob_lr, "LightGBM": prob_lgb}
    fig_roc = fig_roc_curves(y_test, probs)
    fig_pr = fig_pr_curves(y_test, probs)
    fig_cal = fig_calibration(y_test, probs)

    report = {
        "logistic_metrics": asdict(m_lr),
        "lgbm_metrics": asdict(m_lgb),
        "note": "Saved metrics only. Plots re-generated on-demand in UI.",
    }

    # Persist everything
    artifacts = TrainArtifacts(
        task="classification",
        target_col=str(target_col),
        bad_labels=[str(x) for x in bad_labels],
        primary_model=str(primary_model),
        feature_count=int(feat_count),
        id_hint=guess_id_column(list(df_train.columns)),
    )

    bundle = {
        "preprocessor": preprocessor,
        "scaler_for_lr": scaler,
        "logistic": lr,
        "lgbm": lgbm,
        "target_col": artifacts.target_col,
        "bad_labels": artifacts.bad_labels,
        "primary_model": artifacts.primary_model,
    }

    save_bundle(bundle, artifacts, report, score_ref)

    # Put into session
    st.session_state.bundle = bundle
    st.session_state.artifacts = artifacts
    st.session_state.last_report = report
    st.session_state.score_ref = score_ref

    st.success("Huấn luyện xong ✅ (đã lưu model + report vào artifacts/, chuyển tab không mất)")

    st.markdown("### Báo cáo thực nghiệm")
    render_metric_cards("Logistic Regression (baseline)", m_lr)
    render_metric_cards("LightGBM (mở rộng)", m_lgb)

    st.pyplot(fig_roc, clear_figure=True)
    st.pyplot(fig_pr, clear_figure=True)
    st.pyplot(fig_cal, clear_figure=True)

    st.markdown("---")
    st.markdown("### Tải model bundle (tuỳ chọn)")
    buf = io.BytesIO()
    joblib.dump(bundle, buf)
    st.download_button(
        "Tải model bundle (.joblib)",
        data=buf.getvalue(),
        file_name="pb025_models_bundle.joblib",
        mime="application/octet-stream",
    )
