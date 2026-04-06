"""
SolanaSense — Home
==================
Slack-inspired dashboard landing page.
"""

from pathlib import Path
import base64

import streamlit as st

st.set_page_config(
    page_title="SolanaSense — Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Static assets ────────────────────────────────────────────────────────
STATIC_DIR = Path(__file__).resolve().parent / "static"
LOGO_PATH = STATIC_DIR / "logo.png"
BG_PATH = STATIC_DIR / "background.png"


@st.cache_resource
def _load_logo_b64():
    if LOGO_PATH.exists():
        return base64.b64encode(LOGO_PATH.read_bytes()).decode()
    return ""


@st.cache_resource
def _load_bg_b64():
    if BG_PATH.exists():
        return base64.b64encode(BG_PATH.read_bytes()).decode()
    return ""


# ── Solanaceae fun facts ─────────────────────────────────────────────────
SOLANA_FACTS = [
    "🍅 Tomatoes were once considered poisonous in Europe and called 'poison apples' until the 1800s.",
    "🥔 Potatoes were the first vegetable grown in space — aboard the Space Shuttle Columbia in 1995.",
    "🌶️ The Solanaceae family contains over 2,700 species spread across 98 genera.",
    "🍆 Eggplants are technically berries — not vegetables — in botanical classification.",
    "🧬 Tomatoes share about 60% of their DNA with humans.",
    "🥔 The average person eats roughly 33 kg of potatoes per year worldwide.",
    "🌿 Nicotiana tabacum (tobacco) is a member of the Solanaceae family.",
    "🍅 There are over 10,000 known varieties of tomatoes grown around the world.",
    "🌶️ Capsaicin — the compound that makes chillies hot — is produced as a defence against fungal infections.",
    "🥔 Peru is home to over 3,000 potato varieties, the highest diversity on Earth.",
    "🍅 The world's heaviest tomato weighed 4.896 kg — grown in Washington, USA in 2020.",
    "🌿 Deadly nightshade (Atropa belladonna) belongs to the same family as tomatoes and peppers.",
    "🍆 China produces over 60% of the world's eggplant supply.",
    "🌶️ The Carolina Reaper held the Guinness record at 2.2 million Scoville Heat Units.",
    "🥔 Potatoes are 80% water — making them surprisingly hydrating.",
    "🧬 Late blight (Phytophthora infestans) caused the Irish Potato Famine of 1845–1852.",
    "🍅 Lycopene, the red pigment in tomatoes, is a powerful antioxidant linked to heart health.",
    "🌿 Solanaceae plants produce tropane alkaloids — used in medicines like atropine.",
    "🥔 Sweet potatoes are NOT in the Solanaceae family — they're Convolvulaceae (morning glory family).",
    "🌶️ Birds are immune to capsaicin — they help spread chilli seeds across ecosystems.",
]

# ── CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
*:not([class*="material"]):not([data-testid="stIconMaterial"]) { font-family: 'Poppins', sans-serif !important; }

/* ---- Hide Streamlit chrome ---- */
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent !important; pointer-events: none !important; }
[data-testid="collapsedControl"] { z-index: 10001 !important; pointer-events: auto !important; }

/* ---- Background with green overlay ---- */
.stApp {
    background: linear-gradient(170deg,
        rgba(232,245,233,0.82) 0%, rgba(200,230,201,0.78) 50%, rgba(165,214,167,0.82) 100%),
        url("data:image/png;base64,{bg_b64}") center/cover fixed no-repeat !important;
}
[data-testid="stAppViewContainer"], [data-testid="stMain"] { background: transparent !important; }
.block-container { padding-top: 1.5rem !important; max-width: 1100px !important; }

/* ---- Brand header ---- */
.ss-brand-header {
    display: flex; align-items: center; justify-content: center;
    gap: 12px; margin-bottom: 0.2rem;
}
.ss-brand-header img { height: 40px; }
.ss-brand-header span {
    font-size: 1.3rem; font-weight: 700; color: #1B4332 !important;
}

/* ---- Nav links row ---- */
.nav-links {
    display: flex; justify-content: center; gap: 0.3rem;
    margin-bottom: 0.5rem;
}
.nav-links [data-testid="stPageLink-NavLink"] {
    background: rgba(46,125,50,0.10) !important;
    border: 1px solid #C8E6C9 !important;
    border-radius: 20px !important;
    padding: 0.25rem 1rem !important;
}
.nav-links [data-testid="stPageLink-NavLink"] p,
.nav-links [data-testid="stPageLink-NavLink"] span {
    color: #1B4332 !important; font-weight: 500 !important; font-size: 0.85rem !important;
    white-space: nowrap !important; overflow: visible !important; text-overflow: unset !important;
}
.nav-links [data-testid="stPageLink-NavLink"]:hover {
    background: rgba(46,125,50,0.20) !important;
    border-color: #2E7D32 !important;
}

/* ---- Clickable brand ---- */
.brand-link [data-testid="stPageLink-NavLink"] {
    background: transparent !important; border: none !important;
    padding: 0 !important; justify-content: center;
}
.brand-link [data-testid="stPageLink-NavLink"] p,
.brand-link [data-testid="stPageLink-NavLink"] span {
    font-size: 1.4rem !important; font-weight: 800 !important;
    color: #1B4332 !important;
}

/* ---- Sidebar styling ---- */
section[data-testid="stSidebar"] { background: #f5faf6 !important; }
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] { display: none; }

/* ---- Hero ---- */
.hero-home {
    text-align: center;
    padding: 3rem 0 2.5rem 0;
    animation: fadeSlideUp 0.6s ease-out both;
}
.hero-home h1 {
    font-size: 3.6rem !important; font-weight: 800 !important;
    color: #1B4332 !important; line-height: 1.1 !important;
    margin: 0 !important; letter-spacing: -1px;
}
.hero-home .hero-sub {
    font-size: 1.2rem; color: #555 !important;
    max-width: 620px; margin: 1.2rem auto 0 auto; line-height: 1.6;
}

/* ---- Feature cards ---- */
.feature-card {
    background: #fff; border: 1px solid #e8e8e8; border-radius: 16px;
    padding: 2rem 1.5rem; text-align: center;
    transition: all 0.3s ease;
    box-shadow: 0 2px 8px rgba(0,0,0,0.03);
    height: 100%;
}
.feature-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.08);
    border-color: #2E7D32;
}
.feature-icon { font-size: 2.5rem; margin-bottom: 1rem; }
.feature-card h3 {
    font-size: 1.05rem !important; font-weight: 700 !important;
    color: #1B4332 !important; margin: 0 0 0.5rem 0 !important;
}
.feature-card p {
    font-size: 0.9rem; color: #555 !important; line-height: 1.5; margin: 0;
}

/* ---- Stats ---- */
.stat-card { text-align: center; padding: 1.5rem 0.5rem; }
.stat-number { font-size: 2.8rem; font-weight: 800; color: #2E7D32 !important; }
.stat-label { font-size: 0.9rem; color: #666 !important; margin-top: 0.2rem; }

/* ---- Disease chips ---- */
.disease-chip {
    display: inline-block; background: #E8F5E9; color: #1B5E20 !important;
    padding: 8px 16px; border-radius: 20px; margin: 4px;
    font-size: 0.85rem; font-weight: 500; border: 1px solid #C8E6C9;
}

/* ---- CTA button area ---- */
.cta-area { text-align: center; padding: 0.5rem 0 0 0; }

/* ---- Section header ---- */
.section-header {
    text-align: center; font-size: 1.6rem !important; font-weight: 700 !important;
    color: #1B4332 !important; margin: 0 0 1rem 0 !important;
}

/* ---- Ticker ---- */
@keyframes tickerScroll { 0% { transform: translateX(0); } 100% { transform: translateX(-50%); } }
.fact-ticker-wrap {
    position: fixed; bottom: 0; left: 0; right: 0; z-index: 9998;
    background: rgba(27,67,50,0.90); backdrop-filter: blur(8px);
    overflow: hidden; height: 36px; border-top: 2px solid #2E7D32;
}
.fact-ticker-track {
    display: inline-block; white-space: nowrap;
    animation: tickerScroll 90s linear infinite;
}
.fact-ticker-track span {
    display: inline-block; padding: 8px 0;
    font-size: 0.85rem; font-weight: 500; color: #C8E6C9 !important;
}

/* ---- Animations ---- */
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(24px); }
    to   { opacity: 1; transform: translateY(0); }
}
.anim-up { animation: fadeSlideUp 0.6s ease-out both; }
.anim-up-d1 { animation: fadeSlideUp 0.6s ease-out 0.1s both; }
.anim-up-d2 { animation: fadeSlideUp 0.6s ease-out 0.2s both; }
.anim-up-d3 { animation: fadeSlideUp 0.6s ease-out 0.3s both; }
</style>
""".replace("{bg_b64}", _load_bg_b64()), unsafe_allow_html=True)

# ── Hero Section ─────────────────────────────────────────────────────────
logo_b64 = _load_logo_b64()
logo_html = (
    f'<img src="data:image/png;base64,{logo_b64}" style="height:100px;object-fit:contain;" />'
    if logo_b64 else '<div style="font-size:4rem;">\U0001f33f</div>'
)

st.markdown(f"""
<div class="hero-home">
    <div style="text-align:center;">
        {logo_html.replace('height:100px', 'height:70px')}
    </div>
    <div style="text-align:center;font-size:2.2rem;font-weight:800;color:#1B4332 !important;">
        SolanaSense
    </div>
</div>
""", unsafe_allow_html=True)

# ── Nav links ────────────────────────────────────────────────────────────
nav_left, nav_c1, nav_c2, nav_c3, nav_right = st.columns([1, 1, 1, 1, 1])
with nav_c1:
    st.page_link("pages/1_diagnose.py", label="🔬 Diagnose", use_container_width=True)
with nav_c2:
    st.page_link("pages/2_glossary.py", label="📖 Glossary", use_container_width=True)
with nav_c3:
    st.page_link("pages/3_references.py", label="📚 References", use_container_width=True)

st.markdown(f"""
<div class="hero-home" style="padding-top:1rem;">
    <h1>Detect plant diseases<br>in seconds</h1>
    <p class="hero-sub">
        Upload a leaf photo and get instant AI-powered diagnosis with severity metrics,
        treatment recommendations, and root-cause analysis for Solanaceae crops.
    </p>
</div>
""", unsafe_allow_html=True)

# ── CTA ──────────────────────────────────────────────────────────────────
col_l, col_c, col_r = st.columns([2, 1, 2])
with col_c:
    st.page_link("pages/1_diagnose.py", label="🔬  Start Diagnosing  →",
                 use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Feature Cards ────────────────────────────────────────────────────────
features = [
    ("🔬", "AI-Powered Detection",
     "SVM classifier trained on 109 engineered features extracted from each leaf image."),
    ("🎯", "10 Disease Classes",
     "Covers bacterial spot, early & late blight, septoria, target spot, and healthy — across pepper, potato, and tomato."),
    ("📊", "Visual Analysis",
     "Health scores, severity percentages, confidence bars, and before/after disease overlay comparisons."),
    ("💡", "Actionable Insights",
     "Treatment recommendations and root-cause analysis sourced from university extension services."),
]

cols = st.columns(4)
for i, (col, (icon, title, desc)) in enumerate(zip(cols, features)):
    delay = f"anim-up-d{i}" if i < 4 else "anim-up"
    with col:
        st.markdown(f"""
        <div class="feature-card {delay}">
            <div class="feature-icon">{icon}</div>
            <h3>{title}</h3>
            <p>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# ── Stats Row ────────────────────────────────────────────────────────────
st.markdown('<h2 class="section-header">By the Numbers</h2>', unsafe_allow_html=True)
stat_cols = st.columns(4)
stats = [("10", "Disease Classes"), ("109", "Features Extracted"),
         ("3", "Crop Species"), ("7", "Pipeline Stages")]
for col, (num, label) in zip(stat_cols, stats):
    with col:
        st.markdown(f"""
        <div class="stat-card anim-up-d2">
            <div class="stat-number">{num}</div>
            <div class="stat-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Supported Diseases ───────────────────────────────────────────────────
st.markdown('<h2 class="section-header">Supported Diseases</h2>', unsafe_allow_html=True)
diseases = [
    "Pepper — Bacterial Spot", "Pepper — Healthy",
    "Potato — Early Blight", "Potato — Late Blight", "Potato — Healthy",
    "Tomato — Bacterial Spot", "Tomato — Early Blight", "Tomato — Late Blight",
    "Tomato — Septoria Leaf Spot", "Tomato — Target Spot",
]
chips_html = "".join(f'<span class="disease-chip">{d}</span>' for d in diseases)
st.markdown(
    f'<div style="text-align:center;padding:0.5rem 0 4rem 0">{chips_html}</div>',
    unsafe_allow_html=True,
)

# ── Fact Ticker ──────────────────────────────────────────────────────────
separator = "&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;"
facts_html = separator.join(f"🌿 {f}" for f in SOLANA_FACTS)
full_text = f"{facts_html}{separator}{facts_html}"
st.markdown(f"""
<div class="fact-ticker-wrap">
    <div class="fact-ticker-track">
        <span>{full_text}</span>
    </div>
</div>
""", unsafe_allow_html=True)
