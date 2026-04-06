"""
SolanaSense — Diagnose
======================
Image upload, preprocessing pipeline, and AI-powered disease classification.
"""

from pathlib import Path
import base64
import json
import sys
import time

import cv2
import joblib
import numpy as np
import streamlit as st
from PIL import Image

# ── Project imports ──────────────────────────────────────────────────────
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent / "src"))

from plantdisease.data.preprocess.pipeline import PreprocessingPipeline
from plantdisease.features import extract_features_from_pipeline_result

st.set_page_config(
    page_title="SolanaSense — Diagnose",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Paths ────────────────────────────────────────────────────────────────
MODEL_PATH = Path("models/exports/svm.pkl")
CLASS_NAMES_PATH = Path("models/exports/class_names.json")
FEATURE_COLUMNS_PATH = Path("models/exports/feature_columns.json")

# ── Disease knowledge base ───────────────────────────────────────────────
DISEASE_INFO = {
    "Pepper,_bell___Bacterial_spot": {
        "description": "Bacterial leaf spot caused by *Xanthomonas* bacteria. Characterised by small, dark, water-soaked lesions on leaves.",
        "features_seen": "Dark necrotic spots with irregular edges and slight chlorotic halos detected on the leaf surface.",
        "recommendations": [
            "Remove and destroy infected leaves immediately",
            "Apply copper-based bactericide as preventive spray",
            "Avoid overhead watering — use drip irrigation",
            "Rotate crops with non-Solanaceae plants for 2–3 years",
        ],
        "severity_note": "Bacterial spot spreads rapidly in warm, humid conditions.",
        "root_causes": [
            "Poor soil drainage and waterlogging promoting bacterial growth",
            "Excess moisture on foliage from overhead irrigation",
            "Weakened plant immunity due to nutrient deficiency (calcium, potassium)",
            "Contaminated seeds or transplants introducing the pathogen",
        ],
        "root_insight": "Xanthomonas bacteria enter through stomata and wounds — persistent leaf wetness and root stress are the primary systemic enablers.",
    },
    "Pepper,_bell___healthy": {
        "description": "The leaf appears healthy with no visible signs of disease or stress.",
        "features_seen": "Uniform green colouration with consistent texture. No necrotic or chlorotic regions detected.",
        "recommendations": [
            "Continue current care routine",
            "Monitor regularly for early signs of disease",
            "Maintain proper spacing for air circulation",
            "Ensure balanced fertilisation schedule",
        ],
        "severity_note": "No action required — plant is in good health.",
        "root_causes": ["No root stress detected — plant appears systemically healthy"],
        "root_insight": "Healthy foliage indicates adequate root function, nutrient uptake, and water balance.",
    },
    "Potato___Early_blight": {
        "description": "Early blight caused by *Alternaria solani*. Produces characteristic concentric ring ('target') lesions on older leaves first.",
        "features_seen": "Brown concentric ring patterns detected with surrounding chlorosis. Lesions show distinct target-like morphology.",
        "recommendations": [
            "Remove lower infected leaves to slow spread",
            "Apply chlorothalonil or mancozeb fungicide",
            "Improve air circulation between plants",
            "Avoid wetting foliage during irrigation",
            "Rotate crops — avoid planting Solanaceae in the same bed",
        ],
        "severity_note": "Early blight progresses from lower to upper leaves. Early intervention is critical.",
        "root_causes": [
            "Water stress and irregular watering cycles weakening plant defences",
            "Nitrogen imbalance — excess N promotes lush but susceptible growth",
            "Root damage from compacted soil reducing nutrient uptake",
            "Fungal spores overwintering in soil and infected debris",
        ],
        "root_insight": "Alternaria solani thrives when plants are stressed by inconsistent watering and poor root health. Soil-borne inoculum splashes onto lower leaves first.",
    },
    "Potato___Late_blight": {
        "description": "Late blight caused by *Phytophthora infestans*. Highly destructive — causes rapid, water-soaked lesions that turn dark and spread quickly.",
        "features_seen": "Large water-soaked dark lesions with pale green to brown margins. High texture variance indicating active tissue destruction.",
        "recommendations": [
            "⚠️ Act immediately — late blight spreads very rapidly",
            "Remove and destroy all infected plant material",
            "Apply systemic fungicide (e.g., metalaxyl-based)",
            "Ensure good drainage to reduce humidity",
            "Monitor neighbouring plants closely for spread",
        ],
        "severity_note": "Late blight is extremely aggressive and can devastate crops within days. Immediate treatment is essential.",
        "root_causes": [
            "Cool, wet conditions creating ideal pathogen environment",
            "Infected seed tubers introducing Phytophthora into the soil",
            "Poor field drainage keeping soil saturated around roots",
            "Dense planting reducing air flow and trapping humidity",
        ],
        "root_insight": "Phytophthora infestans is an oomycete (water mould) — root-zone waterlogging and infected tubers are the primary systemic entry points.",
    },
    "Potato___healthy": {
        "description": "The potato leaf appears healthy with no visible disease symptoms.",
        "features_seen": "Uniform green colouration with consistent texture across the leaf. No abnormal colour patterns detected.",
        "recommendations": [
            "Continue current care routine",
            "Monitor for early blight signs on lower leaves",
            "Maintain proper hilling to protect tubers",
            "Ensure consistent watering schedule",
        ],
        "severity_note": "No action required — plant is in good health.",
        "root_causes": ["No root stress detected — plant appears systemically healthy"],
        "root_insight": "Healthy foliage indicates adequate root function, nutrient uptake, and water balance.",
    },
    "Tomato___Bacterial_spot": {
        "description": "Bacterial spot caused by *Xanthomonas* species. Creates small, dark, greasy-looking spots on leaves.",
        "features_seen": "Multiple small dark spots detected with water-soaked appearance. Brown necrotic tissue with irregular distribution.",
        "recommendations": [
            "Remove infected leaves and debris",
            "Apply copper-based spray (preventive, not curative)",
            "Avoid working with plants when they are wet",
            "Increase plant spacing for better air flow",
            "Use disease-free seeds and transplants",
        ],
        "severity_note": "Bacterial diseases cannot be cured once established — focus on slowing spread.",
        "root_causes": [
            "Poor soil drainage and persistent waterlogging",
            "Excess moisture from overhead / sprinkler irrigation",
            "Reduced plant resistance due to root stress or nutrient imbalance",
            "Contaminated tools or hands spreading bacteria between plants",
        ],
        "root_insight": "Prolonged leaf wetness and root-zone stress are the primary systemic enablers of bacterial foliar diseases in tomatoes.",
    },
    "Tomato___Early_blight": {
        "description": "Early blight caused by *Alternaria solani*. Produces dark, concentric-ring lesions starting on lower, older leaves.",
        "features_seen": "Brown target-shaped lesions with concentric ring patterns. Chlorotic margins surrounding necrotic centres detected.",
        "recommendations": [
            "Prune lower leaves that touch the soil",
            "Apply fungicide (chlorothalonil or copper-based)",
            "Mulch around plants to prevent soil splash",
            "Water at the base of plants, not overhead",
            "Rotate crops annually",
        ],
        "severity_note": "Early blight is common and manageable with proper fungicide timing.",
        "root_causes": [
            "Water stress and irregular watering weakening defences",
            "Nutrient imbalance — especially nitrogen excess or potassium deficit",
            "Root damage from compacted or poorly drained soil",
            "Infected plant debris left in soil from prior seasons",
        ],
        "root_insight": "Alternaria solani spores survive in soil and splash onto lower foliage. Stressed roots reduce the plant's systemic resistance.",
    },
    "Tomato___Late_blight": {
        "description": "Late blight caused by *Phytophthora infestans*. Causes large, irregular, water-soaked lesions that quickly turn brown-black.",
        "features_seen": "Large irregular dark patches with rapid tissue degradation. High colour variance indicating severe infection progression.",
        "recommendations": [
            "⚠️ Urgent: Remove and destroy all infected material",
            "Apply systemic fungicide immediately",
            "Do NOT compost infected plant material",
            "Check nearby tomato and potato plants for spread",
            "Improve greenhouse ventilation if applicable",
        ],
        "severity_note": "Late blight is a crop emergency — delay can cause total loss within 1–2 weeks.",
        "root_causes": [
            "Waterlogged soil providing ideal oomycete habitat",
            "Infected transplants or volunteer plants in the field",
            "Cool, rainy weather overwhelming plant defences",
            "Dense canopy trapping humidity at the leaf surface",
        ],
        "root_insight": "Phytophthora infestans spreads via airborne sporangia but thrives systemically when roots sit in saturated soil.",
    },
    "Tomato___Septoria_leaf_spot": {
        "description": "Septoria leaf spot caused by *Septoria lycopersici*. Creates many small, circular spots with dark borders and light centres.",
        "features_seen": "Numerous small circular lesions with grey-white centres and dark brown margins. Scattered distribution across leaf surface.",
        "recommendations": [
            "Remove and destroy affected lower leaves",
            "Apply fungicide (mancozeb or chlorothalonil)",
            "Avoid overhead irrigation",
            "Stake plants to keep foliage off the ground",
            "Clean up plant debris at end of season",
        ],
        "severity_note": "Septoria rarely kills plants but severely reduces yield through defoliation.",
        "root_causes": [
            "Persistent leaf wetness from rain or overhead watering",
            "Infected crop debris harbouring pycnidia in the soil",
            "Poor air circulation in dense plantings",
            "Weakened plant immunity from nutrient or water stress",
        ],
        "root_insight": "Septoria lycopersici pycnidiospores splash from soil to lower leaves — the root zone and irrigation method are key systemic factors.",
    },
    "Tomato___Target_Spot": {
        "description": "Target spot caused by *Corynespora cassiicola*. Produces circular to irregular brown spots with concentric zonation.",
        "features_seen": "Brown spots with target-like concentric rings and diffuse margins. Moderate texture irregularity across affected areas.",
        "recommendations": [
            "Remove heavily infected leaves",
            "Apply broad-spectrum fungicide",
            "Improve plant spacing and air circulation",
            "Reduce foliage wetness duration",
            "Consider resistant varieties for next planting",
        ],
        "severity_note": "Target spot is moderately aggressive — consistent fungicide application controls spread.",
        "root_causes": [
            "High humidity and prolonged leaf wetness",
            "Nutrient-stressed plants with reduced systemic resistance",
            "Dense canopy limiting air flow and light penetration",
            "Fungal inoculum persisting in crop residue",
        ],
        "root_insight": "Corynespora cassiicola is an opportunistic fungus that exploits plants weakened by root stress and environmental factors.",
    },
}

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

# ── Static asset loaders ───────────────────────────────────────────────
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
BG_IMAGE_PATH = STATIC_DIR / "background.png"
LOGO_PATH = STATIC_DIR / "logo.png"


@st.cache_resource
def _load_logo_b64():
    if LOGO_PATH.exists():
        return base64.b64encode(LOGO_PATH.read_bytes()).decode()
    return ""


@st.cache_resource
def _load_bg_base64() -> str:
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        p = BG_IMAGE_PATH.with_suffix(ext)
        if p.exists():
            return base64.b64encode(p.read_bytes()).decode()
    return ""


# ── CSS ──────────────────────────────────────────────────────────────────
def _build_css(bg_b64: str) -> str:
    if bg_b64:
        bg_rule = (
            f'background: linear-gradient(170deg,'
            f'rgba(232,245,233,0.82) 0%,rgba(200,230,201,0.78) 50%,rgba(165,214,167,0.82) 100%),'
            f'url("data:image/jpeg;base64,{bg_b64}") center/cover fixed no-repeat;'
        )
    else:
        bg_rule = 'background: linear-gradient(170deg, #E8F5E9 0%, #C8E6C9 50%, #A5D6A7 100%);'

    txt = '#1B4332'
    txt_sub = '#555'
    card_bg = 'rgba(255,255,255,0.9)'
    card_bg_85 = 'rgba(255,255,255,0.85)'
    gauge_bg = '#fff'
    gauge_border = '#C8E6C9'
    bar_bg = '#C8E6C9'
    rec_bg = '#FFFDE7'
    root_bg = '#E8EAF6'
    warn_bg = '#FFF3E0'
    ticker_bg = 'rgba(27,67,50,0.90)'
    brand_green = '#2E7D32'
    conf_name_clr = '#1B4332'
    conf_pct_clr = '#2E7D32'

    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
*:not([class*="material"]):not([data-testid="stIconMaterial"]) {{ font-family: 'Poppins', sans-serif !important; }}
[data-testid="collapsedControl"] span {{ font-family: 'Material Symbols Rounded' !important; }}

/* ---- Hide Streamlit chrome ---- */
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {{ display: none !important; }}
header[data-testid="stHeader"] {{ background: transparent !important; pointer-events: none !important; }}
[data-testid="collapsedControl"] {{ z-index: 10001 !important; pointer-events: auto !important; }}

/* ---- Brand header ---- */
.ss-brand-header {{
    display: flex; align-items: center; justify-content: center; gap: 12px;
    padding: 0.7rem 0 0.2rem 0;
}}
.ss-brand-header img {{ height: 48px; }}
.ss-brand-header span {{
    font-size: 1.7rem !important; font-weight: 800 !important;
    color: #1B4332 !important;
}}

/* ---- Nav links row ---- */
.nav-links [data-testid="stPageLink-NavLink"] {{
    background: rgba(46,125,50,0.10) !important;
    border: 1px solid #C8E6C9 !important;
    border-radius: 20px !important;
    padding: 0.25rem 1rem !important;
}}
.nav-links [data-testid="stPageLink-NavLink"] p,
.nav-links [data-testid="stPageLink-NavLink"] span {{
    color: #1B4332 !important; font-weight: 500 !important; font-size: 0.85rem !important;
    white-space: nowrap !important; overflow: visible !important; text-overflow: unset !important;
}}
.nav-links [data-testid="stPageLink-NavLink"]:hover {{
    background: rgba(46,125,50,0.20) !important;
    border-color: #2E7D32 !important;
}}

/* ---- Clickable brand ---- */
.brand-link [data-testid="stPageLink-NavLink"] {{
    background: transparent !important; border: none !important;
    padding: 0 !important; justify-content: center;
}}
.brand-link [data-testid="stPageLink-NavLink"] p,
.brand-link [data-testid="stPageLink-NavLink"] span {{
    font-size: 1.4rem !important; font-weight: 800 !important;
    color: #1B4332 !important;
}}

/* ---- Sidebar nav ---- */
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] {{ display: none; }}

/* ---- Page background ---- */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {{
    {bg_rule}
    color: {txt};
}}
.block-container {{ padding-top: 1.5rem; padding-bottom: 3.5rem; }}
h1, h2, h3, p, label, div, span {{ color: {txt} !important; }}

/* ---- Compact sidebar ---- */
section[data-testid="stSidebar"] {{
    padding-top: 0.8rem;
    width: 260px !important;
}}
section[data-testid="stSidebar"] .block-container {{
    padding-top: 0.5rem; padding-bottom: 0.5rem;
}}
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div {{
    margin-bottom: -0.2rem;
}}
section[data-testid="stSidebar"] h2 {{
    font-size: 1.1rem; margin-bottom: 0.3rem;
}}

/* ---- News-style fact ticker ---- */
@keyframes tickerScroll {{
    0%   {{ transform: translateX(0); }}
    100% {{ transform: translateX(-50%); }}
}}
.fact-ticker-wrap {{
    position: fixed; bottom: 0; left: 0; right: 0; z-index: 9998;
    background: {ticker_bg};
    backdrop-filter: blur(8px);
    overflow: hidden; height: 36px;
    border-top: 2px solid {brand_green};
}}
.fact-ticker-track {{
    display: inline-block; white-space: nowrap;
    animation: tickerScroll 90s linear infinite;
}}
.fact-ticker-track span {{
    display: inline-block;
    padding: 8px 0; font-size: 0.85rem; font-weight: 500;
    color: #C8E6C9 !important;
}}

/* ---- Entrance animations ---- */
@keyframes fadeSlideUp {{
    from {{ opacity: 0; transform: translateY(24px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes fadeIn {{
    from {{ opacity: 0; }}
    to   {{ opacity: 1; }}
}}
@keyframes pulseGlow {{
    0%, 100% {{ box-shadow: 0 0 8px rgba(46,125,50,0.15); }}
    50%      {{ box-shadow: 0 0 22px rgba(46,125,50,0.35); }}
}}
@keyframes scanLine {{
    0%   {{ top: 0%; }}
    50%  {{ top: 92%; }}
    100% {{ top: 0%; }}
}}

/* ---- Confidence bar ---- */
.conf-bar-wrap {{ margin-bottom: 6px; }}
.conf-bar-header {{
    display: flex; justify-content: space-between; align-items: baseline;
    margin-bottom: 2px;
}}
.conf-bar-name {{
    font-size: 0.82rem; font-weight: 600; color: {conf_name_clr} !important;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}}
.conf-bar-pct {{
    font-size: 0.78rem; font-weight: 600; color: {conf_pct_clr} !important;
    margin-left: 8px; flex-shrink: 0;
}}
.conf-bar-bg {{
    background: {bar_bg}; border-radius: 6px; height: 14px; overflow: hidden;
}}
.conf-bar-fill {{
    height: 100%; border-radius: 6px;
    animation: barGrow 0.8s ease-out both;
}}
@keyframes barGrow {{ from {{ width: 0% !important; }} }}

/* ---- Health gauge ---- */
.gauge-container {{
    background: {gauge_bg}; border-radius: 12px; padding: 16px;
    border: 2px solid {gauge_border}; text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    animation: fadeSlideUp 0.6s ease-out both, pulseGlow 3s ease-in-out infinite;
}}
.gauge-value {{
    font-size: 2.2rem; font-weight: 700; margin: 4px 0;
    text-shadow: 0 1px 3px rgba(0,0,0,0.1);
}}
.gauge-label {{ font-size: 0.95rem; color: {txt_sub} !important; }}
.gauge-bar {{
    height: 10px; border-radius: 5px; margin-top: 10px;
    background: linear-gradient(90deg, #E53935 0%, #FDD835 40%, #43A047 100%);
}}
.gauge-indicator {{
    width: 14px; height: 14px; border-radius: 50%;
    background: #1B4332; border: 2px solid #fff;
    margin-top: -12px; box-shadow: 0 1px 4px rgba(0,0,0,0.3);
}}

/* ---- Step / rec / root / warning cards ---- */
.step-card {{
    background: {card_bg_85}; border-radius: 10px;
    padding: 12px 16px; margin-bottom: 8px;
    border-left: 4px solid {brand_green};
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    animation: fadeSlideUp 0.5s ease-out both;
}}
.step-card h4 {{ margin: 0 0 4px 0; font-size: 0.95rem; }}
.step-card p  {{ margin: 0; font-size: 0.85rem; color: {txt_sub} !important; }}

.rec-card {{
    background: {rec_bg}; border-radius: 10px; padding: 14px 18px;
    border-left: 4px solid #F9A825; margin-top: 6px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    animation: fadeSlideUp 0.55s ease-out both;
}}
.rec-card li {{ margin-bottom: 4px; font-size: 0.9rem; }}

.root-card {{
    background: {root_bg}; border-radius: 10px; padding: 14px 18px;
    border-left: 4px solid #5C6BC0; margin-top: 6px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    animation: fadeSlideUp 0.55s ease-out both;
}}
.root-card li {{ margin-bottom: 4px; font-size: 0.9rem; }}

.warning-banner {{
    background: {warn_bg}; border: 2px solid #FF9800; border-radius: 10px;
    padding: 12px 18px; margin: 8px 0; text-align: center;
    animation: fadeIn 0.5s ease-out both;
}}
.warning-banner p {{ color: #E65100 !important; font-weight: 600; margin: 0; }}

/* ---- Scanning overlay ---- */
.scan-container {{
    position: relative; border-radius: 12px; overflow: hidden;
    border: 3px solid {brand_green};
    box-shadow: 0 0 20px rgba(46,125,50,0.25);
}}
.scan-container img {{ display: block; width: 100%; }}
.scan-line {{
    position: absolute; left: 0; width: 100%; height: 4px;
    background: linear-gradient(90deg, transparent 0%, #43A047 30%, #A5D6A7 50%, #43A047 70%, transparent 100%);
    box-shadow: 0 0 12px rgba(76,175,80,0.6), 0 0 30px rgba(76,175,80,0.3);
    animation: scanLine 2.2s ease-in-out infinite;
    z-index: 10;
}}
.scan-corners::before, .scan-corners::after {{
    content: ''; position: absolute; width: 30px; height: 30px;
    border-color: #2E7D32; border-style: solid; z-index: 5;
}}
.scan-corners::before {{
    top: 6px; left: 6px; border-width: 3px 0 0 3px; border-radius: 4px 0 0 0;
}}
.scan-corners::after {{
    bottom: 6px; right: 6px; border-width: 0 3px 3px 0; border-radius: 0 0 4px 0;
}}

/* ---- Result card ---- */
.result-card {{
    background: {card_bg}; border-radius: 14px;
    padding: 20px; margin: 12px 0;
    box-shadow: 0 4px 16px rgba(0,0,0,0.08);
    animation: fadeSlideUp 0.5s ease-out both;
}}
</style>
"""


# ═════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_artifacts():
    for path, name in [
        (MODEL_PATH, "Model"), (CLASS_NAMES_PATH, "Class names"),
        (FEATURE_COLUMNS_PATH, "Feature columns"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path.resolve()}")

    model = joblib.load(MODEL_PATH)
    with open(CLASS_NAMES_PATH) as f:
        class_names = json.load(f)
    with open(FEATURE_COLUMNS_PATH) as f:
        feature_columns = json.load(f)
    pipeline = PreprocessingPipeline()
    return model, class_names, feature_columns, pipeline


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    if img is None:
        return None
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def format_label(label: str) -> str:
    return label.replace("___", " — ").replace(",_", ", ").replace("_", " ")


def is_healthy(label: str) -> bool:
    return "healthy" in label.lower()


def health_score(severity_pct: float, confidence: float, healthy: bool) -> int:
    if healthy:
        return max(80, min(100, int(100 - severity_pct)))
    return max(0, min(100, int(100 - severity_pct * 1.2 - (1 - confidence) * 20)))


def confidence_bar_html(label: str, value: float, rank: int) -> str:
    pct = value * 100
    colours = ["#2E7D32", "#558B2F", "#7CB342"]
    colour = colours[min(rank, 2)]
    width = max(pct, 2)
    return (
        f'<div class="conf-bar-wrap">'
        f'  <div class="conf-bar-header">'
        f'    <span class="conf-bar-name">{format_label(label)}</span>'
        f'    <span class="conf-bar-pct">{pct:.1f}%</span>'
        f'  </div>'
        f'  <div class="conf-bar-bg">'
        f'    <div class="conf-bar-fill" style="width:{width}%;background:{colour}"></div>'
        f'  </div>'
        f'</div>'
    )


def build_disease_overlay_rgb(result) -> np.ndarray:
    base = bgr_to_rgb(result.shadow_removed if result.shadow_removed is not None else result.resized).copy()
    if result.disease_mask is not None and result.disease_mask.any():
        overlay = base.copy()
        if result.yellow_mask is not None:
            overlay[result.yellow_mask > 0] = [255, 230, 0]
        if result.brown_mask is not None:
            overlay[result.brown_mask > 0] = [230, 50, 50]
        alpha = 0.55
        mask_3ch = np.stack([result.disease_mask > 0] * 3, axis=-1)
        base = np.where(mask_3ch, cv2.addWeighted(overlay, alpha, base, 1 - alpha, 0), base)
    return base


# ═════════════════════════════════════════════════════════════════════════
# Core analysis
# ═════════════════════════════════════════════════════════════════════════

def analyse_leaf(image_bgr: np.ndarray):
    model, class_names, feature_columns, pipeline = load_artifacts()
    result = pipeline.run(image_bgr)
    features = extract_features_from_pipeline_result(result)
    missing = [c for c in feature_columns if c not in features]
    if missing:
        raise ValueError(f"Missing features: {missing[:10]}")

    X = np.array([features[c] for c in feature_columns], dtype=np.float32).reshape(1, -1)
    pred_idx = int(model.predict(X)[0])
    label = class_names[pred_idx]

    confidence = None
    top_preds = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        confidence = float(np.max(probs))
        top_idx = np.argsort(probs)[::-1][:3]
        top_preds = [
            {"label": class_names[int(i)], "confidence": float(probs[int(i)])}
            for i in top_idx
        ]

    return result, {"label": label, "confidence": confidence, "top_predictions": top_preds}


# ═════════════════════════════════════════════════════════════════════════
# UI render functions
# ═════════════════════════════════════════════════════════════════════════

def render_scanning_animation(placeholder, pil_image):
    import io
    buf = io.BytesIO()
    pil_image.resize((400, 400)).save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    steps = [
        ("🔍 Detecting leaf region…", 12),
        ("🧹 Removing background…", 28),
        ("🌙 Removing shadows…", 44),
        ("🎨 Analysing colour patterns…", 58),
        ("🔬 Extracting 109 features…", 72),
        ("🧬 Comparing with disease database…", 88),
        ("✅ Finalising diagnosis…", 100),
    ]
    scan_html = f"""
    <div class="scan-container scan-corners" style="max-width:400px;margin:0 auto">
        <div class="scan-line"></div>
        <img src="data:image/png;base64,{img_b64}" alt="Scanning..." />
    </div>
    """
    with placeholder.container():
        st.markdown(scan_html, unsafe_allow_html=True)
        bar = st.progress(0, text="Initialising analysis…")
        for msg, pct in steps:
            time.sleep(0.4)
            bar.progress(pct, text=msg)
        time.sleep(0.3)
    placeholder.empty()


def render_pipeline_steps(result):
    stages = []
    stages.append(("📷  Original Image", bgr_to_rgb(result.original),
                    "Raw uploaded image before any processing."))
    stages.append(("✂️  Background Removed", bgr_to_rgb(result.background_removed),
                    "Deep-learning (U2-Net) background removal — leaf isolated on transparent background."))
    stages.append(("📐  Resized (300×300)", bgr_to_rgb(result.resized),
                    "Standardised spatial resolution using Lanczos-4 interpolation."))
    if result.shadow_removed is not None:
        stages.append(("🌙  Shadow Removed", bgr_to_rgb(result.shadow_removed),
                        "HSV-based shadow detection and correction on the leaf surface."))
    if result.disease_overlay is not None:
        stages.append(("🔴  Disease Regions Detected", bgr_to_rgb(result.disease_overlay),
                        "Yellow = chlorosis, Red = necrosis. Based on HSV colour segmentation."))
    if result.leaf_mask is not None:
        stages.append(("🍃  Leaf Mask", result.leaf_mask,
                        "Binary mask of the leaf area extracted from alpha channel."))
    if result.disease_mask is not None:
        stages.append(("🦠  Disease Mask", result.disease_mask,
                        "Binary mask of diseased regions (yellow + brown combined)."))

    cols_per_row = 3
    for row_start in range(0, len(stages), cols_per_row):
        row = stages[row_start : row_start + cols_per_row]
        cols = st.columns(len(row))
        for col, (title, img, desc) in zip(cols, row):
            with col:
                st.markdown(f"**{title}**")
                if img is not None:
                    st.image(img, use_container_width=True)
                st.caption(desc)


def render_health_gauge(score: int, sev_pct: float, healthy: bool):
    if score >= 80:
        emoji, status, colour = "✅", "Healthy", "#43A047"
    elif score >= 60:
        emoji, status, colour = "⚠️", "Mild Disease", "#FDD835"
    elif score >= 35:
        emoji, status, colour = "🟠", "Moderate Disease", "#FF9800"
    else:
        emoji, status, colour = "🔴", "Severe Disease", "#E53935"

    indicator_left = max(0, min(96, score))
    st.markdown(f"""
    <div class="gauge-container">
        <div class="gauge-label">Plant Health Score</div>
        <div class="gauge-value" style="color:{colour} !important">{emoji} {score}/100</div>
        <div class="gauge-label">{status} &nbsp;|&nbsp; Severity: {sev_pct:.1f}%</div>
        <div class="gauge-bar"></div>
        <div class="gauge-indicator" style="margin-left:{indicator_left}%"></div>
    </div>
    """, unsafe_allow_html=True)


def render_confidence_bars(top_preds):
    html = ""
    for i, pred in enumerate(top_preds):
        html += confidence_bar_html(pred["label"], pred["confidence"], i)
    st.markdown(html, unsafe_allow_html=True)


def render_prediction_explanation(label: str, result):
    info = DISEASE_INFO.get(label, None)
    if info is None:
        st.info("No detailed description available for this class.")
        return
    st.markdown(f"**{info['description']}**")
    st.markdown(f"""
    <div class="step-card">
        <h4>🔬 What the model detected</h4>
        <p>{info['features_seen']}</p>
    </div>
    """, unsafe_allow_html=True)


def render_recommendations(label: str):
    info = DISEASE_INFO.get(label, None)
    if info is None:
        return
    recs_html = "".join(f"<li>{r}</li>" for r in info["recommendations"])
    st.markdown(f"""
    <div class="rec-card">
        <strong>📋 Recommended Actions</strong>
        <ul style="margin-top:8px;margin-bottom:4px">{recs_html}</ul>
        <p style="font-size:0.82rem;color:#888 !important;margin-top:8px">
            {info['severity_note']}
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.caption(
        "Recommendations are sourced from established plant pathology references "
        "(APS, university extension services) and encoded in the app's disease "
        "knowledge base. They are general-purpose — consult a plant pathologist "
        "or agricultural extension officer for site-specific advice."
    )


def render_root_causes(label: str):
    info = DISEASE_INFO.get(label, None)
    if info is None or "root_causes" not in info:
        return
    causes_html = "".join(f"<li>{c}</li>" for c in info["root_causes"])
    st.markdown(f"""
    <div class="root-card">
        <strong>🌱 Possible Root / Systemic Causes</strong>
        <ul style="margin-top:8px;margin-bottom:4px">{causes_html}</ul>
        <p style="font-size:0.82rem;color:#666 !important;margin-top:8px">
            <strong>Insight:</strong> {info['root_insight']}
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.caption(
        "Root-cause linkages are inferred from established plant pathology literature. "
        "Leaf symptoms do not confirm root-zone issues — consider professional soil "
        "and root diagnosis for a comprehensive assessment."
    )


def render_confidence_warning(confidence: float):
    if confidence is not None and confidence < 0.65:
        st.markdown(f"""
        <div class="warning-banner">
            <p>⚠️ Low confidence prediction ({confidence:.0%}) — the image may be unclear or the disease is unusual.</p>
            <p style="font-weight:400 !important;font-size:0.85rem">
                Try better lighting, a closer photo of the leaf, or a flatter angle.
            </p>
        </div>
        """, unsafe_allow_html=True)


def render_before_after(result):
    original_rgb = bgr_to_rgb(result.resized)
    overlay_rgb = build_disease_overlay_rgb(result)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Before (processed leaf)**")
        st.image(original_rgb, use_container_width=True)
    with col2:
        st.markdown("**After (disease highlighted)**")
        st.image(overlay_rgb, use_container_width=True)


def render_severity_metrics(result):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Leaf Pixels", f"{result.total_leaf_pixels:,}")
    c2.metric("Diseased Pixels", f"{result.diseased_pixels:,}")
    c3.metric("Yellow (Chlorosis)", f"{result.yellow_pixels:,}")
    c4.metric("Brown (Necrosis)", f"{result.brown_pixels:,}")


def render_fact_ticker():
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


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════

def main():
    # ── CSS ──────────────────────────────────────────────────────────────
    bg_b64 = _load_bg_base64()
    st.markdown(_build_css(bg_b64), unsafe_allow_html=True)

    # ── Brand header (clickable back to home) ─────────────────────────
    logo_nav_b64 = _load_logo_b64()
    if logo_nav_b64:
        st.markdown(
            f'<div class="ss-brand-header">'
            f'<img src="data:image/png;base64,{logo_nav_b64}"/>'
            f'</div>',
            unsafe_allow_html=True,
        )
    bl, bc, br = st.columns([2, 1, 2])
    with bc:
        st.markdown('<div class="brand-link">', unsafe_allow_html=True)
        st.page_link("main.py", label="🌿 SolanaSense", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Nav links ─────────────────────────────────────────────────────
    st.markdown('<div class="nav-links">', unsafe_allow_html=True)
    nl, nc1, nc2, nc3, nr = st.columns([1, 1, 1, 1, 1])
    with nc1:
        st.page_link("pages/1_diagnose.py", label="🔬 Diagnose", use_container_width=True)
    with nc2:
        st.page_link("pages/2_glossary.py", label="📖 Glossary", use_container_width=True)
    with nc3:
        st.page_link("pages/3_references.py", label="📚 References", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Page header ──────────────────────────────────────────────────────
    st.markdown(
        "<h1 style='text-align:center;font-size:2rem !important;margin:0 0 1rem 0 !important;'>"
        "🔬 Diagnose Your Plant</h1>",
        unsafe_allow_html=True,
    )

    # ── Display options ──────────────────────────────────────────────────
    with st.expander("⚙️ Display Options", expanded=False):
        opt_cols = st.columns(3)
        with opt_cols[0]:
            show_pipeline = st.toggle("Preprocessing Pipeline", value=False,
                                      help="Show the full preprocessing pipeline stages")
        with opt_cols[1]:
            show_scanning = st.toggle("Scan animation", value=True,
                                      help="Show a live-scanning effect while analysing")
        with opt_cols[2]:
            show_before_after = st.toggle("Before / After", value=True,
                                          help="Side-by-side original vs disease-highlighted view")

    # ── Artifact check ───────────────────────────────────────────────────
    try:
        model, class_names, feature_columns, pipeline = load_artifacts()
    except Exception as e:
        st.error(f"❌ Failed to load model artifacts: {e}")
        render_fact_ticker()
        return

    # ── Image input ──────────────────────────────────────────────────────
    st.subheader("📸 Input Image")
    tab_upload, tab_camera = st.tabs(["Upload", "Camera"])
    with tab_upload:
        uploaded_file = st.file_uploader(
            "Upload a leaf image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )
    with tab_camera:
        camera_file = st.camera_input("Take a photo", label_visibility="collapsed")

    selected_file = camera_file if camera_file is not None else uploaded_file

    if selected_file is None:
        st.info("👆 Upload or photograph a leaf to begin analysis.")
        render_fact_ticker()
        return

    try:
        pil_image = Image.open(selected_file)
    except Exception as e:
        st.error(f"Could not open image: {e}")
        render_fact_ticker()
        return

    # Small preview — constrained to 1/3 width
    col_img_l, col_img_c, col_img_r = st.columns([1, 1, 1])
    with col_img_c:
        st.image(pil_image, caption="Selected image", use_container_width=True)

    # ── Run analysis ─────────────────────────────────────────────────────
    if not st.button("🔬 Analyse Leaf", type="primary", use_container_width=True):
        render_fact_ticker()
        return

    image_bgr = pil_to_bgr(pil_image)

    if show_scanning:
        scan_ph = st.empty()
        render_scanning_animation(scan_ph, pil_image)

    with st.spinner("Running model inference…"):
        try:
            result, prediction = analyse_leaf(image_bgr)
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            return

    label = prediction["label"]
    confidence = prediction["confidence"]
    top_preds = prediction["top_predictions"]
    healthy = is_healthy(label)
    sev_pct = result.severity_percent
    h_score = health_score(sev_pct, confidence or 0.5, healthy)

    # ── Confidence warning ───────────────────────────────────────────────
    render_confidence_warning(confidence)

    # ── Result tabs ──────────────────────────────────────────────────────
    tab_names = ["🧪 Diagnosis", "🧠 Explanation", "📊 Metrics"]
    if show_before_after:
        tab_names.append("📸 Before / After")
    if show_pipeline:
        tab_names.append("🔍 Pipeline")
    tab_names += ["🌱 Root Causes", "📋 Recommendations"]

    result_tabs = st.tabs(tab_names)
    tab_idx = 0

    with result_tabs[tab_idx]:
        badge_colour = "#43A047" if healthy else "#E53935"
        st.markdown(
            f'<div class="result-card">'
            f'<h2 style="text-align:center;color:{badge_colour} !important;margin:0">'
            f'{"🌱" if healthy else "🦠"} {format_label(label)}</h2>'
            f'</div>',
            unsafe_allow_html=True,
        )
        col_gauge, col_conf = st.columns([1, 1])
        with col_gauge:
            render_health_gauge(h_score, sev_pct, healthy)
        with col_conf:
            st.markdown("**Top Predictions**")
            if top_preds:
                render_confidence_bars(top_preds)
    tab_idx += 1

    with result_tabs[tab_idx]:
        render_prediction_explanation(label, result)
    tab_idx += 1

    with result_tabs[tab_idx]:
        render_severity_metrics(result)
    tab_idx += 1

    if show_before_after:
        with result_tabs[tab_idx]:
            render_before_after(result)
        tab_idx += 1

    if show_pipeline:
        with result_tabs[tab_idx]:
            render_pipeline_steps(result)
        tab_idx += 1

    with result_tabs[tab_idx]:
        render_root_causes(label)
    tab_idx += 1

    with result_tabs[tab_idx]:
        render_recommendations(label)

    render_fact_ticker()


if __name__ == "__main__":
    main()
else:
    main()
