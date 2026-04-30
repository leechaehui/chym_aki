"""
================================================================================
app.py  —  AKI CDSS Streamlit
================================================================================
환자 등록번호(Stay ID) 입력 → DB 자동 조회 → 필드 자동 채움 → 예측
================================================================================
실행법: streamlit run app.py
================================================================================
"""

import sys, os, warnings
warnings.filterwarnings("ignore")

import streamlit as st
import plotly.graph_objects as go

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "xgb_model"))

st.set_page_config(
    page_title="CDSS | AKI 예측",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #f0f4f8; }
    .main .block-container { padding: 2rem 3rem; max-width: 1100px; }

    section[data-testid="stSidebar"] {
        background: #1F3365 !important;
        min-width: 210px !important; max-width: 210px !important;
    }
    section[data-testid="stSidebar"] > div { padding: 0 !important; }
    section[data-testid="stSidebar"] * { color: white !important; }
    section[data-testid="stSidebar"] .stButton > button {
        background: transparent !important; border: none !important;
        color: white !important; text-align: left !important;
        width: 100% !important; padding: 0.55rem 1.2rem !important;
        font-size: 0.93rem !important; border-radius: 0 !important;
        margin: 0 !important; box-shadow: none !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(255,255,255,0.12) !important;
    }
    section[data-testid="stSidebar"] .stButton > button:focus {
        box-shadow: none !important;
    }

    .card {
        background: white; border-radius: 12px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        margin-bottom: 1.2rem;
    }
    .gauge-card {
        background: white; border-radius: 12px;
        padding: 1.5rem; text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }
    .autofill-badge {
        display: inline-block; background: #dbeafe; color: #1d4ed8;
        border-radius: 6px; padding: 2px 8px; font-size: 0.75em;
        font-weight: 600; margin-left: 0.5rem;
    }
    .badge-high { display:inline-block; background:#fee2e2; color:#dc2626;
                  border-radius:20px; padding:4px 14px; font-weight:700; font-size:0.9em; }
    .badge-warn { display:inline-block; background:#fff7ed; color:#c2410c;
                  border-radius:20px; padding:4px 14px; font-weight:700; font-size:0.9em; }
    .badge-ok   { display:inline-block; background:#f0fdf4; color:#16a34a;
                  border-radius:20px; padding:4px 14px; font-weight:700; font-size:0.9em; }
    .bar-bg { background:#f3f4f6; border-radius:4px; height:10px; margin-top:4px; }
</style>
""", unsafe_allow_html=True)

# ── 세션 초기화 ───────────────────────────────────────────────────────────────
for k, v in [("page","환자 데이터 입력"), ("patient",None), ("db_info",None)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── 백엔드 로드 ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_backend():
    mods = {}
    try:
        from db import execute_query; mods["db"] = execute_query
    except Exception: pass
    for key, fn, mod in [
        ("scr03","build_drug_management_screen_response","scr03_drug_management"),
        ("scr06","build_ai_risk_score_screen_response",  "scr06_ai_risk_score"),
    ]:
        try:
            import importlib
            m = importlib.import_module(mod)
            mods[key] = getattr(m, fn)
        except Exception: pass
    try:
        from xgb_model.inference import aki_engine
        mods["xgb"] = aki_engine if aki_engine.is_ready else None
    except Exception:
        mods["xgb"] = None
    return mods

M = load_backend()

# ── DB에서 환자 정보 조회 ──────────────────────────────────────────────────────
def fetch_patient_info(stay_id: int) -> dict | None:
    if "db" not in M:
        return None
    try:
        rows = M["db"]("""
            SELECT
                stay_id, subject_id, age, gender,
                cr_min   AS baseline_cr,
                cr_max   AS current_cr,
                bun_max  AS bun,
                egfr_ckdepi AS egfr,
                COALESCE(current_map, isch_map_mean) AS map_val,
                first_careunit
            FROM aki_project.cdss_master_features
            WHERE stay_id = :sid
        """, {"sid": stay_id})
        if not rows:
            return None
        r = rows[0]
        return {
            "stay_id":    int(r["stay_id"]),
            "subject_id": str(r.get("subject_id") or stay_id),
            "age":        int(r.get("age") or 65),
            "gender":     str(r.get("gender") or "M"),
            "baseline_cr":float(r.get("baseline_cr") or 1.0),
            "current_cr": float(r.get("current_cr")  or 1.0),
            "bun":        float(r.get("bun")          or 20),
            "egfr":       float(r.get("egfr")         or 60),
            "urine_6h":   200.0,
            "map":        float(r.get("map_val")       or 75),
            "careunit":   str(r.get("first_careunit")  or "ICU"),
        }
    except Exception as e:
        st.error(f"DB 조회 오류: {e}")
        return None

# ────────────────────────────────────────────────────────────────────────────
# 사이드바
# ────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1.4rem 1.2rem 0.2rem;display:flex;align-items:center;gap:0.5rem">
        <span style="font-size:1.3rem;font-weight:900">CDSS</span>
        <span style="background:#ef4444;color:white;font-size:0.62rem;
                     font-weight:700;padding:2px 6px;border-radius:4px">AKI</span>
    </div>
    <div style="height:1px;background:rgba(255,255,255,0.18);margin:0.3rem 0.8rem 0.6rem"></div>
    <div style="font-size:0.68rem;color:rgba(255,255,255,0.45);
                padding:0 1.2rem 0.4rem;letter-spacing:0.08em">MENU</div>
    """, unsafe_allow_html=True)

    for label, icon in [("환자 데이터 입력","📋"),("검사 결과 (예측)","🔬"),("처방 관리","💊")]:
        prefix = "▶ " if st.session_state.page == label else "   "
        if st.button(f"{prefix}{icon}  {label}", key=f"nav_{label}",
                     use_container_width=True):
            st.session_state.page = label
            st.rerun()

    st.markdown("""
    <div style="height:1px;background:rgba(255,255,255,0.18);margin:1.2rem 0.8rem 0.8rem"></div>
    <div style="padding:0 1.2rem;display:flex;align-items:center;gap:0.7rem">
        <div style="width:34px;height:34px;border-radius:50%;background:#3b82f6;flex-shrink:0;
                    display:flex;align-items:center;justify-content:center;
                    font-weight:700;font-size:0.9rem">김</div>
        <div>
            <div style="font-size:0.88rem;font-weight:600">김철수 선생님</div>
            <div style="font-size:0.72rem;opacity:0.55">신장내과 전문의</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 시스템 상태 표시 ───────────────────────────────────────────────────
    st.divider()
    st.markdown('<div style="font-size:0.68rem;color:rgba(255,255,255,0.45);padding:0 1.2rem 0.4rem;letter-spacing:0.08em">🔧 시스템 상태</div>', unsafe_allow_html=True)

    status_cols = st.columns(3)
    with status_cols[0]:
        xgb_status = "✅" if M.get("xgb") else "❌"
        st.caption(f"{xgb_status} XGBoost")
    with status_cols[1]:
        scr06_status = "✅" if M.get("scr06") else "❌"
        st.caption(f"{scr06_status} SCR-06")
    with status_cols[2]:
        db_status = "✅" if M.get("db") else "❌"
        st.caption(f"{db_status} Database")


# ────────────────────────────────────────────────────────────────────────────
# 페이지 1: 환자 데이터 입력
# ────────────────────────────────────────────────────────────────────────────
def page_input():
    st.markdown("## 환자 데이터 입력")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**환자 등록번호 (Stay ID) 입력**")
    st.caption("Stay ID를 입력하면 DB에서 임상 수치를 자동으로 불러옵니다.")

    sid_input = st.text_input(
        "Stay ID",
        placeholder="예: 30000153",
        label_visibility="collapsed",
        value=str(st.session_state.db_info["stay_id"])
              if st.session_state.db_info else ""
    )
    search_btn = st.button("🔍 조회", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if search_btn:
        if not sid_input.strip().isdigit():
            st.error("숫자로 된 Stay ID를 입력해주세요.")
            st.stop()
        info = fetch_patient_info(int(sid_input.strip()))
        if info is None:
            st.error(f"Stay ID {sid_input} 에 해당하는 환자를 찾을 수 없습니다.")
            st.stop()
        st.session_state.db_info = info
        st.rerun()

    info = st.session_state.db_info
    if info is None:
        st.info("👆 Stay ID를 입력하고 조회 버튼을 눌러주세요.")
        return

    g_label = "남성" if info["gender"] == "M" else "여성"
    st.markdown(f"""
    <div style="background:#1F3365;color:white;border-radius:10px;
                padding:0.8rem 1.2rem;margin-bottom:1rem">
        <b>👤 Stay ID: {info['stay_id']}</b>
        &nbsp;|&nbsp; Subject ID: {info['subject_id']}
        &nbsp;|&nbsp; {info['age']}세 / {g_label}
        &nbsp;|&nbsp; {info['careunit']}
        <span style="float:right;font-size:0.78em;opacity:0.7;margin-top:2px">
            ✅ DB에서 자동 입력됨
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**필수 임상 데이터** &nbsp;"
                '<span class="autofill-badge">DB 자동 입력</span>'
                '<span style="font-size:0.82em;color:#6b7280;margin-left:0.5rem">'
                "필요 시 수정 가능합니다</span>",
                unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        baseline_cr = st.number_input(
            "기저 크레아티닌 (Baseline Cr) mg/dL",
            value=round(info["baseline_cr"], 2), step=0.1, format="%.2f"
        )
    with c2:
        current_cr = st.number_input(
            "현재 크레아티닌 (Current Cr) mg/dL",
            value=round(info["current_cr"], 2), step=0.1, format="%.2f"
        )

    c3, c4 = st.columns(2)
    with c3:
        urine = st.number_input(
            "최근 소변량 (Urine Output - 6H) mL",
            value=int(info["urine_6h"]) if info["urine_6h"] else 200, step=10
        )
    with c4:
        map_v = st.number_input(
            "MAP (평균동맥압) mmHg",
            value=int(info["map"]) if info["map"] else 75, step=1
        )

    c5, c6 = st.columns(2)
    with c5:
        bun = st.number_input(
            "BUN (혈중요소질소) mg/dL",
            value=int(info["bun"]) if info["bun"] else 20, step=1
        )
    with c6:
        egfr = st.number_input(
            "eGFR (추정사구체여과율) mL/min",
            value=int(info["egfr"]) if info["egfr"] else 60, step=1
        )

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("→ 48시간 내 AKI 예측하기", type="primary", use_container_width=True):
        st.session_state.patient = {
            "mode":        "db",
            "stay_id":     info["stay_id"],
            "subject_id":  info["subject_id"],
            "age":         info["age"],
            "gender":      info["gender"],
            "careunit":    info["careunit"],
            "baseline_cr": baseline_cr,
            "current_cr":  current_cr,
            "urine_6h":    urine,
            "map":         map_v,
            "bun":         bun,
            "egfr":        egfr,
        }
        st.session_state.page = "검사 결과 (예측)"
        st.rerun()


# ────────────────────────────────────────────────────────────────────────────
# 페이지 2: 검사 결과 (예측)
# ────────────────────────────────────────────────────────────────────────────
def calc_risk_from_values(p: dict) -> tuple[int, list]:
    score, factors = 0, []
    cr_ratio = p["current_cr"] / max(p["baseline_cr"], 0.1)
    if cr_ratio >= 3.0:
        score += 55; factors.append((f"현재 크레아티닌 급등 (기저 대비 {cr_ratio:.1f}배)", 55, "#ef4444"))
    elif cr_ratio >= 2.0:
        score += 40; factors.append((f"현재 크레아티닌 상승 (기저 대비 {cr_ratio:.1f}배)", 40, "#ef4444"))
    elif cr_ratio >= 1.5:
        score += 25; factors.append((f"크레아티닌 경계 상승 ({cr_ratio:.1f}배)", 25, "#f97316"))
    if p["urine_6h"] < 100:
        score += 25; factors.append(("소변량 감소 (핍뇨 상태)", 25, "#f97316"))
    elif p["urine_6h"] < 200:
        score += 10; factors.append(("소변량 감소 경향", 10, "#f97316"))
    if p["map"] < 65:
        score += 15; factors.append(("MAP 저하 (신관류압 부족)", 15, "#f97316"))
    if p["egfr"] < 30:
        score += 15; factors.append(("eGFR 심각 저하 (<30)", 15, "#ef4444"))
    elif p["egfr"] < 60:
        score += 8;  factors.append(("eGFR 저하 (<60)", 8, "#f97316"))
    if p["bun"] > 30:
        score += 10; factors.append(("BUN 상승 (>30)", 10, "#f97316"))
    if p["age"] >= 65:
        score += 5;  factors.append(("고령 (65세 이상)", 5, "#f59e0b"))
    return min(score, 100), factors


def risk_gauge(value: int, color: str) -> go.Figure:
    fig = go.Figure(go.Pie(
        values=[value, 100 - value], hole=0.72,
        marker_colors=[color, "#f3f4f6"],
        textinfo="none", hoverinfo="skip", sort=False,
    ))
    fig.update_layout(
        showlegend=False, margin=dict(t=20, b=20, l=20, r=20),
        height=230, paper_bgcolor="white",
        annotations=[
            {"text": f"<b>{value}%</b>",
             "font": {"size": 40, "color": color, "family": "Arial Black"},
             "showarrow": False, "x": 0.5, "y": 0.54},
            {"text": "48시간 내 발생 확률",
             "font": {"size": 11, "color": "#9ca3af"},
             "showarrow": False, "x": 0.5, "y": 0.36},
        ]
    )
    return fig


def page_result():
    st.markdown("## 검사 결과 (예측)")
    p = st.session_state.patient

    if not p or "stay_id" not in p:
        st.session_state.patient = None
        st.session_state.db_info = None
        st.warning("환자 정보가 초기화되었습니다. 다시 조회해주세요.")
        if st.button("← 데이터 입력으로"):
            st.session_state.page = "환자 데이터 입력"; st.rerun()
        return

    stay_id = p["stay_id"]
    g_label = "남성" if p["gender"] == "M" else "여성"

    st.markdown(f"""
    <div class="card">
        <div style="display:flex;gap:2rem;flex-wrap:wrap;align-items:center">
            <div>
                <div style="font-size:0.75em;color:#6b7280">환자 정보 (ID/나이/성별)</div>
                <div style="font-size:1.05em;font-weight:700;color:#1F3365">
                    {p['subject_id']} ({p['age']}세 / {g_label})
                </div>
            </div>
            <div>
                <div style="font-size:0.75em;color:#6b7280">크레아티닌 (기저/현재)</div>
                <div style="font-size:1.05em;font-weight:700;color:#ef4444">
                    {p['baseline_cr']} → {p['current_cr']}
                </div>
            </div>
            <div>
                <div style="font-size:0.75em;color:#6b7280">소변량 / MAP</div>
                <div style="font-size:1.05em;font-weight:700;color:#1F3365">
                    {p['urine_6h']}mL / {p['map']}mmHg
                </div>
            </div>
            <div>
                <div style="font-size:0.75em;color:#6b7280">BUN / eGFR</div>
                <div style="font-size:1.05em;font-weight:700;color:#1F3365">
                    {p['bun']} / {p['egfr']}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 모델 예측 - 상세 추적 추가 ────────────────────────────────────
    risk_score = 0
    factors = []
    source_label = "규칙 기반"
    prediction_log = []

    # XGBoost 시도
    if M.get("xgb"):
        try:
            risk_score = M["xgb"].predict_single(stay_id).aki_probability_pct
            source_label = "🤖 XGBoost AI"
            prediction_log.append("✅ XGBoost 예측 성공")
        except Exception as e:
            prediction_log.append(f"❌ XGBoost 실패: {str(e)[:50]}")
    else:
        prediction_log.append("⏭️ XGBoost 모듈 미로드")

    # SCR-06 폴백
    if risk_score == 0 and "scr06" in M:
        try:
            d = M["scr06"](stay_id)
            risk_score = d.risk_display.displayed_value
            factors = [
                (r.factor_name, r.contribution,
                 "#ef4444" if r.is_exceeded else "#d1d5db")
                for r in d.risk_factor_table
            ]
            source_label = "📋 규칙 기반 (SCR-06)"
            prediction_log.append("✅ SCR-06 폴백 성공")
        except Exception as e:
            prediction_log.append(f"❌ SCR-06 실패: {str(e)[:50]}")
    elif "scr06" not in M and risk_score == 0:
        prediction_log.append("⏭️ SCR-06 모듈 미로드")

    # 최종 폴백 - 임상 수치 기반
    if risk_score == 0:
        risk_score, factors = calc_risk_from_values(p)
        source_label = "📊 임상 수치 기반"
        prediction_log.append("✅ 임상 수치 계산 완료")

    # ── 위험도 판정 ────────────────────────────────────────────────────
    if risk_score >= 70:
        gc = "#ef4444"; stage = "고위험군 (Stage 2 예상)"
        badge = f'<span class="badge-high">🔺 {stage}</span>'
    elif risk_score >= 40:
        gc = "#f97316"; stage = "중위험군 (Stage 1 예상)"
        badge = f'<span class="badge-warn">⚠ {stage}</span>'
    else:
        gc = "#22c55e"; stage = "저위험군"
        badge = f'<span class="badge-ok">✅ {stage}</span>'

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="gauge-card">', unsafe_allow_html=True)
        st.markdown("**AKI 발생 예측 결과**")
        st.plotly_chart(risk_gauge(risk_score, gc),
                        use_container_width=True, config={"displayModeBar": False})
        st.markdown(badge, unsafe_allow_html=True)

        # 📌 예측 출처를 명확하게 표시
        if "XGBoost" in source_label:
            st.success(f"✅ **{source_label}** — AI 모델이 예측을 실행했습니다.")
        elif "SCR-06" in source_label:
            st.warning(f"⚠️ **{source_label}** — 규칙 기반 알고리즘으로 계산됨")
        else:
            st.info(f"📊 **{source_label}** — 입력 값 기반 계산")

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**AI 예측 주요 원인 분석**")
        top = sorted(factors, key=lambda x: -x[1])[:5]
        for name, contrib, color in top:
            bar = min(int(contrib * 1.5), 100)
            st.markdown(f"""
            <div style="margin-bottom:0.8rem">
                <div style="display:flex;justify-content:space-between;font-size:0.88em">
                    <span>{name}</span>
                    <span style="color:{color};font-weight:700">+ {contrib}%</span>
                </div>
                <div class="bar-bg">
                    <div style="width:{bar}%;background:{color};
                                height:10px;border-radius:4px"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        if not top:
            st.info("기여 요인 데이터 없음")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 📌 예측 과정 상세 보기
    with st.expander("🔍 예측 과정 상세 보기", expanded=False):
        st.caption("**모델 실행 순서 및 결과:**")
        for log_msg in prediction_log:
            st.caption(log_msg)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("→ 처방 관리로 이동", use_container_width=True):
        st.session_state.page = "처방 관리"; st.rerun()


# ────────────────────────────────────────────────────────────────────────────
# 페이지 3: 처방 관리  ✅ 동적 추천 + HTML 깨짐 수정
# ────────────────────────────────────────────────────────────────────────────
def page_prescription():
    st.markdown("## 처방 관리")
    p = st.session_state.patient

    if not p or "stay_id" not in p:
        st.warning("먼저 환자 데이터를 입력해주세요.")
        if st.button("← 데이터 입력으로"):
            st.session_state.page = "환자 데이터 입력"; st.rerun()
        return

    if "scr03" not in M:
        st.error("SCR-03 모듈 로드 실패")
        return

    try:
        data      = M["scr03"](p["stay_id"])
        stop_list = [rx for rx in data.prescriptions if rx.is_nephrotoxic]
        ai        = data.ai_alert
        triggered = [a for a in data.combination_alerts if a.is_triggered]

        # ── 중단 권고 약물 ─────────────────────────────────────────────
        st.markdown(
            '<div style="background:#fef2f2;border-left:4px solid #ef4444;'
            'border-radius:10px;padding:1.2rem 1.4rem;margin-bottom:1rem">',
            unsafe_allow_html=True
        )
        st.markdown("""
        <div style="display:flex;align-items:center;gap:0.7rem;margin-bottom:0.8rem">
            <div style="width:36px;height:36px;background:#ef4444;border-radius:50%;flex-shrink:0;
                        display:flex;align-items:center;justify-content:center;font-size:1.1rem">🔴</div>
            <div>
                <b style="color:#dc2626">투약 중단 권고 약물 (신독성 우려)</b><br>
                <span style="font-size:0.85em;color:#6b7280">
                    환자의 신기능이 악화되고 있어 아래 약물의 중단 또는 주의가 필요합니다.
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if stop_list:
            for rx in stop_list:
                st.markdown(
                    f'<div style="font-size:0.92em;color:#374151;margin:0.4rem 0">'
                    f'• <b>{rx.drug_name}</b> ({rx.route}) — {rx.note}</div>',
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                '<div style="font-size:0.92em;color:#6b7280;margin:0.4rem 0">'
                '현재 중단 권고 신독성 약물 없음</div>',
                unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

        # ── 추천 처방 — 환자 상태 기반 동적 생성 ─────────────────────
        recommend_items = []

        if stop_list:
            recommend_items.append("크리스탈로이드 수액 (Normal Saline 0.9%) — 시간당 달기")

        if ai.burden_score >= 2 or ai.overall_risk == "높음":
            recommend_items.append("신장내과(Nephrology) 긴급 협진 의뢰")

        if ai.drug_risk_score >= 2:
            recommend_items.append("신독성 약물 용량 조정 또는 대체 항생제 검토")

        if any(a.combo_type == "vanco_piptazo" for a in triggered):
            recommend_items.append("Vancomycin + Pip/Tazo 병용 중단 — 대체 항생제 처방 권고")

        if ai.overall_risk == "보통":
            recommend_items.append("4~6시간 내 신기능 재검 (Cr, BUN, 소변량)")

        if not recommend_items:
            recommend_items.append("정기적인 신기능 모니터링 유지 (Cr, BUN 매일 체크)")
            recommend_items.append("충분한 수분 섭취 유지")

        st.markdown(
            '<div style="background:#f0fdf4;border-left:4px solid #22c55e;'
            'border-radius:10px;padding:1.2rem 1.4rem;margin-bottom:1rem">',
            unsafe_allow_html=True
        )
        st.markdown("""
        <div style="display:flex;align-items:center;gap:0.7rem;margin-bottom:0.8rem">
            <div style="width:36px;height:36px;background:#22c55e;border-radius:50%;flex-shrink:0;
                        display:flex;align-items:center;justify-content:center;font-size:1.1rem">✅</div>
            <div>
                <b style="color:#16a34a">추천 처치 (필요 처방)</b><br>
                <span style="font-size:0.85em;color:#6b7280">
                    현재 환자 상태를 기반으로 아래 처방이 필요할 수 있습니다.
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        for item in recommend_items:
            st.markdown(
                f'<div style="font-size:0.92em;color:#374151;margin:0.4rem 0">• {item}</div>',
                unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

        # ── AI 신독성 모니터링 배너 ────────────────────────────────────
        st.divider()
        lv    = ai.risk_level.level
        color = {"high": "#ef4444", "warning": "#f97316"}.get(lv, "#22c55e")
        bg    = {"high": "#fef2f2", "warning": "#fff7ed"}.get(lv, "#f0fdf4")
        st.markdown(
            f'<div style="background:{bg};border-left:5px solid {color};'
            f'padding:0.8rem 1rem;border-radius:8px">'
            f'<b>🤖 AI 신독성 모니터링 — 위험도: {ai.overall_risk} | '
            f'부담점수: {ai.burden_score}점 | '
            f'약물위험점수: {ai.drug_risk_score}점</b></div>',
            unsafe_allow_html=True
        )
        st.caption(f"💡 {ai.recommendation}")
        st.caption(ai.detail_message)

        # ── 위험 약물 조합 경고 ────────────────────────────────────────
        if triggered:
            st.divider()
            st.markdown("**⚠️ 위험 약물 조합 감지**")
            for a in triggered:
                sc = "#ef4444" if a.severity == "high" else "#f97316"
                st.markdown(
                    f'<div style="background:#fef2f2;border-left:3px solid {sc};'
                    f'padding:0.5rem 0.8rem;border-radius:6px;margin-bottom:0.3rem;'
                    f'font-size:0.9em">🚨 {a.description}</div>',
                    unsafe_allow_html=True
                )

    except Exception as e:
        st.error(f"처방 데이터 오류: {e}")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("← 검사 결과로 돌아가기"):
        st.session_state.page = "검사 결과 (예측)"; st.rerun()


# ── 라우터 ────────────────────────────────────────────────────────────────────
page = st.session_state.page
if   page == "환자 데이터 입력": page_input()
elif page == "검사 결과 (예측)": page_result()
elif page == "처방 관리":         page_prescription()