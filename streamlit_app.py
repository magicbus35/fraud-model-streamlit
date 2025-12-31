import datetime
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ---- 기본 설정 ----
st.set_page_config(page_title="이상거래 탐지 모델 리포트 (PaySim)", layout="wide")


# ---- 데이터/모델 요약 ----
SUMMARY = {
    "updated_at": "2025-12-18 16:22",
    "dataset": {
        "total_rows": 6_362_620,
        "fraud_rows": 8_213,
        "train_rows": 6_259_047,
        "train_fraud": 6_613,
        "test_rows": 103_573,
        "test_fraud": 1_600,
    },
    "type_counts_train": {
        "PAYMENT": 2_116_354,
        "CASH_OUT": 2_204_075,
        "CASH_IN": 1_375_225,
        "TRANSFER": 522_784,
        "DEBIT": 40_609,
    },
    "type_counts_test": {
        "PAYMENT": 35_141,
        "CASH_OUT": 33_425,
        "CASH_IN": 24_059,
        "TRANSFER": 10_125,
        "DEBIT": 823,
    },
    "model": {
        "name": "XGBoost (full feat, scale_pos_weight=2)",
        "f1": 0.4855,
        "acc": 0.9701,
        "f1_best": 0.7980,
        "best_thr": 0.972,
        "features": 18,
    },
}


# ---- 헤더 ----
st.title("이상거래 탐지 모델 리포트 (PaySim / 위치·계정 비식별)")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("F1 (기본 임계값=0.5)", f"{SUMMARY['model']['f1']*100:.2f}%")
with col2:
    st.metric("Accuracy", f"{SUMMARY['model']['acc']*100:.2f}%")
with col3:
    st.metric(
        "Best F1 / 임계값",
        f"{SUMMARY['model']['f1_best']*100:.2f}% @ {SUMMARY['model']['best_thr']:.3f}",
    )
with col4:
    st.metric("업데이트", SUMMARY["updated_at"])

st.write("---")


# ---- Part A. 데이터 ----
st.subheader("Part A: 데이터 분석")
pipeline_cols = st.columns([1.2, 1])
with pipeline_cols[0]:
    st.markdown("### 1) 데이터 파이프라인")
    steps = [
        ("원본", SUMMARY["dataset"]["total_rows"], "PaySim 전체"),
        ("Train", SUMMARY["dataset"]["train_rows"], "step ≤ 600"),
        ("Test", SUMMARY["dataset"]["test_rows"], "step > 600"),
    ]
    for name, cnt, desc in steps:
        st.write(
            f"**{name}**: {cnt:,}건 \n"
            f"<span style='color:gray'>{desc}</span>",
            unsafe_allow_html=True,
        )

with pipeline_cols[1]:
    st.markdown("### 2) 거래 유형 분포 (Train)")
    type_df = pd.DataFrame(
        [{"type": k, "count": v} for k, v in SUMMARY["type_counts_train"].items()]
    )
    chart = (
        alt.Chart(type_df)
        .mark_bar(cornerRadius=4)
        .encode(
            x=alt.X("type:N", title="거래 유형", axis=alt.Axis(labelAngle=90)),
            y=alt.Y("count:Q", title="건수", axis=alt.Axis(titleAngle=0)),
        )
        .properties(height=220)
    )
    st.altair_chart(chart, use_container_width=True)

st.write("---")

# ---- 분할/불균형 ----
split_cols = st.columns([1, 1])
with split_cols[0]:
    st.markdown("### 3) 시간 기반 Train/Test Split")
    train_pct = SUMMARY["dataset"]["train_rows"] / SUMMARY["dataset"]["total_rows"] * 100
    test_pct = 100 - train_pct
    split_df = pd.DataFrame({"set": ["Train", "Test"], "pct": [train_pct, test_pct]})
    split_chart = (
        alt.Chart(split_df)
        .mark_bar()
        .encode(
            x=alt.X("set:N", title=None),
            y=alt.Y("pct:Q", title="비율 (%)"),
            color=alt.Color("set", scale=alt.Scale(range=["#2ca02c", "#d62728"])),
            text=alt.Text("pct:Q", format=".1f"),
        )
        .properties(height=200)
    )
    st.altair_chart(split_chart, use_container_width=True)
    st.caption("step ≤ 600 → Train, 이후 → Test (미래 데이터 유출 방지)")

with split_cols[1]:
    st.markdown("### 4) 이상거래 비율")
    frac = SUMMARY["dataset"]["fraud_rows"] / SUMMARY["dataset"]["total_rows"] * 100
    st.write(
        f"- 전체 이상거래 비율: **{frac:.3f}%** "
        f"(이상거래 {SUMMARY['dataset']['fraud_rows']:,} / 총 {SUMMARY['dataset']['total_rows']:,})"
    )
    st.write(
        f"- Train 이상거래: {SUMMARY['dataset']['train_fraud']:,} | "
        f"Test 이상거래: {SUMMARY['dataset']['test_fraud']:,}"
    )
    st.write("- 처리: 이상거래 SMOTE 미사용, `class_weight`/`scale_pos_weight`로 보정")

st.write("---")


# ---- Part B. 모델 성능 ----
st.subheader("Part B: 모델 성능")
perf_cols = st.columns(4)
with perf_cols[0]:
    st.metric("모델", SUMMARY["model"]["name"])
with perf_cols[1]:
    st.metric("Features", SUMMARY["model"]["features"])
with perf_cols[2]:
    st.metric("Best F1", f"{SUMMARY['model']['f1_best']*100:.2f}%")
with perf_cols[3]:
    st.metric("Best Threshold", f"{SUMMARY['model']['best_thr']:.3f}")

st.markdown("#### 임계값 스캔 (0.9~1.0 구간 촘촘 탐색)")
thr_points = pd.DataFrame(
    {
        "threshold": [0.90, 0.93, 0.95, 0.97, 0.972, 0.98],
        "f1": [0.70, 0.73, 0.76, 0.78, 0.798, 0.77],
    }
)
thr_chart = (
    alt.Chart(thr_points)
    .mark_line(point=True)
    .encode(x=alt.X("threshold:Q"), y=alt.Y("f1:Q", title="F1"))
    .properties(height=240)
)
st.altair_chart(thr_chart, use_container_width=True)
st.caption("임계값 0.972 부근에서 F1≈0.798 (Precision/Recall 균형 기반)")


@st.cache_resource
def load_model():
    return joblib.load(Path("models") / "paysim_generic_no_flag_featplus.joblib")


def get_feature_importance(top_k: int = 15) -> pd.DataFrame:
    clf = load_model()
    prep = clf.named_steps["prep"]
    cat_names = prep.named_transformers_["cat"].get_feature_names_out(["type", "amount_bin"])
    num_names = prep.named_transformers_["num"].feature_names_in_
    feat_names = np.concatenate([cat_names, num_names])
    importances = clf.named_steps["model"].feature_importances_
    order = np.argsort(importances)[::-1]
    return pd.DataFrame(
        {
            "feature": feat_names[order][:top_k],
            "importance": importances[order][:top_k],
        }
    )


st.write("---")

# ---- Part C. 실험 로그 요약 ----
st.subheader("Part C: 실험 요약")
exp_data = pd.DataFrame(
    [
        {"실험": "기본(균형 샘플)", "F1": 0.12, "메모": "step/type/amount 최소 피처"},
        {"실험": "피처 확장", "F1": 0.40, "메모": "유형변화율, 금액 Z 추가"},
        {"실험": "이상거래 가중 + class_weight", "F1": 0.4855, "메모": "feat+ (spw≈2)"},
        {"실험": "feat++/시간", "F1": 0.4736, "메모": "feat++/time (spw≈2)"},
        {"실험": "임계값 튜닝", "F1": 0.7980, "메모": "thr≈0.972"},
    ]
)
st.table(exp_data.style.format({"F1": "{:.3f}"}))

st.info(
    "현재 스트림릿은 PaySim 기반 이상거래·계정 비식별 버전입니다. "
    "기존 IBM/MCC 소비모델 대시보드는 `streamlit_app_backup_paysim.py`에 백업되어 있습니다."
)


# ---- Part D. 피처/의사결정 요약 ----
st.write("---")
st.subheader("Part D: 피처/의사결정 요약")

feat_cols = st.columns([1, 1])
with feat_cols[0]:
    st.markdown("### 5) 피처 구성 (18개)")
    st.markdown(
        "- 시간(6): step, day, hour, dow, hour_sin, hour_cos\n"
        "- 금액(7): amount_log, amount_log_z3/5/10, amount_log_iqr_z, amount_bin, amount_rank_pct\n"
        "- 유형 변화율(4): type_change_rate2/3/5/10\n"
        "- 유형 one-hot: type\n"
    )
    st.code(
        "step, day, hour, dow, hour_sin, hour_cos,\n"
        "amount_log, amount_log_z3, amount_log_z5, amount_log_z10, amount_log_iqr_z, amount_bin, amount_rank_pct,\n"
        "type_change_rate2, type_change_rate3, type_change_rate5, type_change_rate10, type(one-hot)"
    )
    st.caption("최종 모델 기준 피처 수는 one-hot 포함 28개입니다.")
with feat_cols[1]:
    st.markdown("### 6) 성능 향상 히스토리")
    hist_df = pd.DataFrame(
        {
            "stage": ["기본", "피처확장", "이상거래가중(feat+)", "feat++/시간", "임계값튜닝"],
            "f1": [0.12, 0.40, 0.4855, 0.47, 0.798],
        }
    )
    hist_chart = (
        alt.Chart(hist_df)
        .mark_line(point=True)
        .encode(x="stage:N", y=alt.Y("f1:Q", title="F1"))
        .properties(height=240)
    )
    st.altair_chart(hist_chart, use_container_width=True)


imp_cols = st.columns([1, 1])
with imp_cols[0]:
    st.markdown("### 7) 피처 중요도 (실측)")
    fi = get_feature_importance(12)
    fi_chart = (
        alt.Chart(fi)
        .mark_bar()
        .encode(
            x=alt.X("importance:Q", title="중요도"),
            y=alt.Y("feature:N", sort="-x", title=None),
        )
        .properties(height=260)
    )
    st.altair_chart(fi_chart, use_container_width=True)

with imp_cols[1]:
    st.markdown("### 8) 의사결정 규칙 요약")
    st.caption("상위 중요 피처를 활용한 간단 플로우 (설명용)")
    st.graphviz_chart(
        """
        digraph G {
          node [shape=box, style=filled, color=lightyellow];
          start [label="거래 입력", shape=box, style=filled, color=lightskyblue];
          t1 [label="type = PAYMENT?", shape=diamond];
          t2 [label="type_change_rate2 > 0.3?", shape=diamond];
          t3 [label="amount_log_z3 > 1.5?", shape=diamond];
          f1 [label="정상", shape=box, style=filled, color=lightgreen];
          f2 [label="이상거래 의심", shape=box, style=filled, color=tomato];
          f3 [label="이상거래 의심", shape=box, style=filled, color=tomato];
          start -> t1;
          t1 -> f1 [label="Yes"];
          t1 -> t2 [label="No"];
          t2 -> f2 [label="Yes"];
          t2 -> t3 [label="No"];
          t3 -> f3 [label="> 1.5"];
          t3 -> f1 [label="<= 1.5"];
        }
        """
    )

# ---- 인사이트 추가 ----
st.markdown("### 9) 피처/규칙 인사이트")
st.write(
    "- **거래 유형**: type_PAYMENT, CASH_IN, TRANSFER가 전체 중요도의 대부분을 차지해 유형 자체가 강력한 신호임.\n"
    "- **유형 변화율**: type_change_rate2/3/5/10이 상위권 → 짧은 구간에서 거래 유형이 급변하면 이상거래 위험도 상승.\n"
    "- **금액 이상치**: amount_log_z3·IQR Z가 중요 → 최근 분포 대비 이례적으로 큰 금액이 핵심 단서.\n"
    "- **시간 패턴**: hour가 기여 → 비정상 시간대 거래가 위험 신호.\n"
    "- **의사결정 흐름**: PAYMENT면 정상으로, PAYMENT가 아니면서 유형 변화율이 높거나 금액 Z가 크면 이상거래 의심으로 분기."
)


# ---- 다운로드 / 참고 ----
st.write("---")
st.markdown("#### 참고 / 다운로드")
backup_path = Path(__file__).with_name("streamlit_app_backup_paysim.py")
st.write(f"- 이전 대시보드 백업: `{backup_path.name}`")
st.write(
    "- 테스트 결과 CSV: `paysim_trials_results_v4.csv`, `paysim_trials_results_v5.csv` (최신)"
)
st.write(
    "- 최신 모델: `models/paysim_generic_no_flag_featplus.joblib` "
    "(feat 확장, best_thr≈0.972)"
)

st.caption(f"⏱️ 생성 시각: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
