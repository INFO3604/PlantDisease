"""
SolanaSense — Plant Disease Detection Web Application
=====================================================
Enhanced UI with step-by-step analysis, scanning animation,
disease highlighting, confidence bars, root-cause linkage,
and actionable recommendations.
"""

from pathlib import Path
import base64
import json
import random
import sys
import time

import cv2
import joblib
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

# ── Project imports ──────────────────────────────────────────────────────
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "src"))

from plantdisease.data.preprocess.pipeline import PreprocessingPipeline
from plantdisease.features import extract_features_from_pipeline_result

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


# ── Background image loader ──────────────────────────────────────────────
BG_IMAGE_PATH = Path(__file__).resolve().parent / "static" / "background.png"


@st.cache_resource
def _load_bg_base64() -> str:
    """Return base64-encoded background image, or empty string if missing."""
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        p = BG_IMAGE_PATH.with_suffix(ext)
        if p.exists():
            return base64.b64encode(p.read_bytes()).decode()
    return ""


# ── Solanaceae fun facts (shown in ticker) ───────────────────────────────
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
# ── SVG logo (inline, no external file needed) ──────────────────────────
SOLANA_LOGO_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" width="64" height="64">
  <defs>
    <linearGradient id="leafG" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#43A047"/>
      <stop offset="100%" style="stop-color:#1B5E20"/>
    </linearGradient>
    <linearGradient id="scanG" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#81C784;stop-opacity:0"/>
      <stop offset="50%" style="stop-color:#A5D6A7;stop-opacity:0.7"/>
      <stop offset="100%" style="stop-color:#81C784;stop-opacity:0"/>
    </linearGradient>
  </defs>
  <!-- Outer circle -->
  <circle cx="60" cy="60" r="56" fill="#E8F5E9" stroke="#2E7D32" stroke-width="3"/>
  <!-- Leaf shape -->
  <path d="M60 22 C30 45, 28 80, 60 98 C92 80, 90 45, 60 22Z" fill="url(#leafG)" opacity="0.92"/>
  <!-- Leaf vein -->
  <line x1="60" y1="30" x2="60" y2="88" stroke="#E8F5E9" stroke-width="2" stroke-linecap="round"/>
  <line x1="60" y1="48" x2="42" y2="58" stroke="#E8F5E9" stroke-width="1.5" stroke-linecap="round"/>
  <line x1="60" y1="48" x2="78" y2="58" stroke="#E8F5E9" stroke-width="1.5" stroke-linecap="round"/>
  <line x1="60" y1="62" x2="44" y2="72" stroke="#E8F5E9" stroke-width="1.5" stroke-linecap="round"/>
  <line x1="60" y1="62" x2="76" y2="72" stroke="#E8F5E9" stroke-width="1.5" stroke-linecap="round"/>
  <!-- Scanning bar animation -->
  <rect x="32" y="0" width="56" height="6" rx="3" fill="url(#scanG)">
    <animateTransform attributeName="transform" type="translate" values="0,20;0,90;0,20" dur="2.5s" repeatCount="indefinite"/>
  </rect>
  <!-- Magnifying glass (small) -->
  <circle cx="82" cy="38" r="11" fill="none" stroke="#1B4332" stroke-width="2.5"/>
  <line x1="90" y1="46" x2="98" y2="54" stroke="#1B4332" stroke-width="2.5" stroke-linecap="round"/>
</svg>
"""


def _build_css(bg_b64: str, dark: bool = False) -> str:
    """Build the full CSS block, optionally injecting the background image."""
    if dark:
        if bg_b64:
            bg_rule = (
                f'background: linear-gradient(170deg,'
                f'rgba(18,18,18,0.92) 0%,rgba(30,30,30,0.90) 50%,rgba(24,24,24,0.92) 100%),'
                f'url("data:image/jpeg;base64,{bg_b64}") center/cover fixed no-repeat;'
            )
        else:
            bg_rule = 'background: linear-gradient(170deg, #121212 0%, #1E1E1E 50%, #181818 100%);'
    else:
        if bg_b64:
            bg_rule = (
                f'background: linear-gradient(170deg,'
                f'rgba(232,245,233,0.82) 0%,rgba(200,230,201,0.78) 50%,rgba(165,214,167,0.82) 100%),'
                f'url("data:image/jpeg;base64,{bg_b64}") center/cover fixed no-repeat;'
            )
        else:
            bg_rule = 'background: linear-gradient(170deg, #E8F5E9 0%, #C8E6C9 50%, #A5D6A7 100%);'

    # Colour tokens
    txt = '#E0E0E0' if dark else '#1B4332'
    txt_sub = '#BDBDBD' if dark else '#555'
    card_bg = 'rgba(40,40,40,0.92)' if dark else 'rgba(255,255,255,0.9)'
    card_bg_85 = 'rgba(40,40,40,0.88)' if dark else 'rgba(255,255,255,0.85)'
    gauge_bg = '#2A2A2A' if dark else '#fff'
    gauge_border = '#4CAF50' if dark else '#C8E6C9'
    bar_bg = '#333' if dark else '#C8E6C9'
    rec_bg = 'rgba(50,45,20,0.85)' if dark else '#FFFDE7'
    root_bg = 'rgba(30,35,55,0.85)' if dark else '#E8EAF6'
    warn_bg = 'rgba(50,35,15,0.85)' if dark else '#FFF3E0'
    sidebar_bg = '#1A1A1A' if dark else ''
    ticker_bg = 'rgba(10,10,10,0.95)' if dark else 'rgba(27,67,50,0.90)'
    hero_clr = '#A5D6A7' if dark else '#1B4332'
    tagline_clr = '#9E9E9E' if dark else '#555'
    brand_green = '#66BB6A' if dark else '#2E7D32'
    conf_name_clr = '#E0E0E0' if dark else '#1B4332'
    conf_pct_clr = '#81C784' if dark else '#2E7D32'

    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
*:not([class*="material"]):not([data-testid="stIconMaterial"]) {{ font-family: 'Poppins', sans-serif !important; }}
[data-testid="collapsedControl"] span {{ font-family: 'Material Symbols Rounded' !important; }}

/* ---- Page background ---- */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {{
    {bg_rule}
    color: {txt};
}}
.block-container {{ padding-top: 1.5rem; padding-bottom: 3.5rem; }}
h1, h2, h3, p, label, div, span {{ color: {txt} !important; }}

/* ---- Hero section ---- */
.hero-section {{
    text-align: center; padding: 2rem 0 1rem 0;
    animation: fadeSlideUp 0.6s ease-out both;
}}
.hero-section h1 {{
    font-size: 3rem !important; font-weight: 800 !important;
    letter-spacing: -0.5px; margin: 0.4rem 0 0 0 !important;
    color: {hero_clr} !important;
}}
.hero-section .hero-tagline {{
    font-size: 1.15rem; font-weight: 400;
    color: {tagline_clr} !important; margin: 0.4rem 0 0 0;
}}

/* ---- Compact sidebar ---- */
section[data-testid="stSidebar"] {{
    padding-top: 0.8rem;
    width: 260px !important;
    {f'background: {sidebar_bg} !important;' if dark else ''}
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
section[data-testid="stSidebar"] .stDivider {{
    margin: 0.4rem 0;
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
.anim-fade-up {{
    animation: fadeSlideUp 0.55s ease-out both;
}}
.anim-fade-in {{
    animation: fadeIn 0.6s ease-out both;
}}

/* ---- Confidence bar ---- */
.conf-bar-wrap {{
    margin-bottom: 6px;
}}
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
    background: {bar_bg}; border-radius: 6px; height: 14px;
    overflow: hidden;
}}
.conf-bar-fill {{
    height: 100%; border-radius: 6px;
    animation: barGrow 0.8s ease-out both;
}}
@keyframes barGrow {{
    from {{ width: 0% !important; }}
}}

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

/* ---- Step cards ---- */
.step-card {{
    background: {card_bg_85}; border-radius: 10px;
    padding: 12px 16px; margin-bottom: 8px;
    border-left: 4px solid {brand_green};
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    animation: fadeSlideUp 0.5s ease-out both;
}}
.step-card h4 {{ margin: 0 0 4px 0; font-size: 0.95rem; }}
.step-card p  {{ margin: 0; font-size: 0.85rem; color: {txt_sub} !important; }}

/* ---- Recommendation card ---- */
.rec-card {{
    background: {rec_bg}; border-radius: 10px; padding: 14px 18px;
    border-left: 4px solid #F9A825; margin-top: 6px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    animation: fadeSlideUp 0.55s ease-out both;
}}
.rec-card li {{ margin-bottom: 4px; font-size: 0.9rem; }}

/* ---- Root-cause card ---- */
.root-card {{
    background: {root_bg}; border-radius: 10px; padding: 14px 18px;
    border-left: 4px solid #5C6BC0; margin-top: 6px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    animation: fadeSlideUp 0.55s ease-out both;
}}
.root-card li {{ margin-bottom: 4px; font-size: 0.9rem; }}

/* ---- Warning banner ---- */
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
.scan-container img {{
    display: block; width: 100%;
}}
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

/* ---- Result card pop ---- */
.result-card {{
    background: {card_bg}; border-radius: 14px;
    padding: 20px; margin: 12px 0;
    box-shadow: 0 4px 16px rgba(0,0,0,0.08);
    animation: fadeSlideUp 0.5s ease-out both;
}}

/* ---- Upload widget dark mode ---- */
{'''
[data-testid="stFileUploader"],
[data-testid="stFileUploader"] section,
[data-testid="stFileUploader"] section > div,
[data-testid="stFileUploader"] small,
[data-testid="stFileUploadDropzone"],
[data-testid="stFileUploadDropzone"] div,
[data-testid="stFileUploadDropzone"] span,
[data-testid="stFileUploadDropzone"] small,
[data-testid="stCameraInput"] label,
[data-testid="stMarkdownContainer"] p,
.uploadedFileName,
.stTabs [data-baseweb="tab-list"] button,
.stTabs [data-baseweb="tab-list"] button div,
[data-testid="stFileUploader"] button {
    color: #E0E0E0 !important;
}
[data-testid="stFileUploadDropzone"] {
    background: rgba(40,40,40,0.7) !important;
    border-color: #4CAF50 !important;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] div {
    color: #E0E0E0 !important;
}
[data-testid="stTextInput"] input {
    color: #E0E0E0 !important;
    background: rgba(40,40,40,0.7) !important;
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary span,
[data-testid="stExpander"] summary div {
    color: #E0E0E0 !important;
}
''' if dark else ''}
</style>
"""


# ═════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_artifacts():
    """Load model, class names, feature columns, and preprocessing pipeline."""
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


def severity_colour(pct: float) -> str:
    if pct < 5:
        return "#43A047"
    if pct < 20:
        return "#FDD835"
    if pct < 40:
        return "#FF9800"
    return "#E53935"


def health_score(severity_pct: float, confidence: float, healthy: bool) -> int:
    """Return 0-100 health score."""
    if healthy:
        return max(80, min(100, int(100 - severity_pct)))
    return max(0, min(100, int(100 - severity_pct * 1.2 - (1 - confidence) * 20)))


def confidence_bar_html(label: str, value: float, rank: int) -> str:
    """Render a single confidence bar."""
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
    """Build a clean disease-region highlight image for display."""
    base = bgr_to_rgb(result.shadow_removed if result.shadow_removed is not None else result.resized).copy()
    if result.disease_mask is not None and result.disease_mask.any():
        overlay = base.copy()
        # Yellow regions -> yellow highlight
        if result.yellow_mask is not None:
            overlay[result.yellow_mask > 0] = [255, 230, 0]
        # Brown regions -> red highlight
        if result.brown_mask is not None:
            overlay[result.brown_mask > 0] = [230, 50, 50]
        # Blend 55% overlay so original texture shows through
        alpha = 0.55
        mask_3ch = np.stack([result.disease_mask > 0] * 3, axis=-1)
        base = np.where(mask_3ch, cv2.addWeighted(overlay, alpha, base, 1 - alpha, 0), base)
    return base


# ═════════════════════════════════════════════════════════════════════════
# Core analysis function
# ═════════════════════════════════════════════════════════════════════════

def analyse_leaf(image_bgr: np.ndarray):
    """Run full pipeline + classification.  Returns (pipeline_result, prediction_dict)."""
    model, class_names, feature_columns, pipeline = load_artifacts()

    # Preprocessing pipeline
    result = pipeline.run(image_bgr)

    # Feature extraction
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

    prediction = {
        "label": label,
        "confidence": confidence,
        "top_predictions": top_preds,
    }
    return result, prediction


# ═════════════════════════════════════════════════════════════════════════
# UI sections
# ═════════════════════════════════════════════════════════════════════════

def render_scanning_animation(placeholder, pil_image):
    """Show scanning animation overlaid on the actual leaf image."""
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

    # Show the image with a scanning overlay
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
    """Show the preprocessing pipeline stages as an expandable gallery."""
    stages = []

    # 1 - Original
    stages.append(("📷  Original Image", bgr_to_rgb(result.original),
                    "Raw uploaded image before any processing."))
    # 2 - Background removed
    stages.append(("✂️  Background Removed", bgr_to_rgb(result.background_removed),
                    "Deep-learning (U2-Net) background removal — leaf isolated on transparent background."))
    # 3 - Resized
    stages.append(("📐  Resized (300×300)", bgr_to_rgb(result.resized),
                    "Standardised spatial resolution using Lanczos-4 interpolation."))
    # 4 - Shadow removed
    if result.shadow_removed is not None:
        stages.append(("🌙  Shadow Removed", bgr_to_rgb(result.shadow_removed),
                        "HSV-based shadow detection and correction on the leaf surface."))
    # 5 - Disease overlay
    if result.disease_overlay is not None:
        stages.append(("🔴  Disease Regions Detected", bgr_to_rgb(result.disease_overlay),
                        "Yellow = chlorosis, Red = necrosis. Based on HSV colour segmentation."))
    # 6 - Leaf mask
    if result.leaf_mask is not None:
        stages.append(("🍃  Leaf Mask", result.leaf_mask,
                        "Binary mask of the leaf area extracted from alpha channel."))
    # 7 - Disease mask
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
    """Render a visual health gauge."""
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
    """Render top-3 prediction confidence bars."""
    html = ""
    for i, pred in enumerate(top_preds):
        html += confidence_bar_html(pred["label"], pred["confidence"], i)
    st.markdown(html, unsafe_allow_html=True)


def render_prediction_explanation(label: str, result):
    """Show a human-readable explanation of the prediction."""
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
    """Show actionable recommendations in a styled card."""
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
    """Show root / systemic cause linkage for the predicted disease."""
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
    """Show a warning if confidence is low."""
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
    """Side-by-side before (original) vs after (disease overlay) comparison."""
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
    """Display severity breakdown metrics."""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Leaf Pixels", f"{result.total_leaf_pixels:,}")
    c2.metric("Diseased Pixels", f"{result.diseased_pixels:,}")
    c3.metric("Yellow (Chlorosis)", f"{result.yellow_pixels:,}")
    c4.metric("Brown (Necrosis)", f"{result.brown_pixels:,}")


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════

def main():
    _logo_path = Path(__file__).resolve().parent / "static" / "logo.png"
    st.set_page_config(
        page_title="SolanaSense — Plant Disease Detection",
        page_icon=str(_logo_path) if _logo_path.exists() else "🌿",
        layout="centered",
    )

    # ── Compact sidebar (must come before CSS so dark_mode toggle is available) ──
    with st.sidebar:
        st.image("https://img.icons8.com/emoji/48/herb.png", width=32)
        st.markdown("**SolanaSense**")
        dark_mode = st.toggle("🌙 Dark mode", value=False,
                              help="Switch between light and dark themes",
                              key="dark_mode")
        show_steps = st.toggle("Analysis steps", value=True,
                               help="Toggle the full preprocessing pipeline view")
        show_scanning = st.toggle("Scan animation", value=True,
                                  help="Show a live-scanning effect while analysing")
        show_before_after = st.toggle("Before / After", value=True,
                                      help="Side-by-side original vs disease-highlighted view")
        st.divider()
        st.caption("v2.0 · SVM · 109 features")

    # Build CSS with background image if available
    bg_b64 = _load_bg_base64()
    st.markdown(_build_css(bg_b64, dark=dark_mode), unsafe_allow_html=True)

    # ── Hero header ──────────────────────────────────────────────────────
    logo_path = Path(__file__).resolve().parent / "static" / "logo.png"
    if logo_path.exists():
        logo_b64 = base64.b64encode(logo_path.read_bytes()).decode()
        logo_html = f'<img src="data:image/png;base64,{logo_b64}" style="max-width:120px;max-height:120px;object-fit:contain;" />'
    else:
        logo_html = SOLANA_LOGO_SVG.replace('width="64"', 'width="96"').replace('height="64"', 'height="96"')
    st.markdown(f"""
    <div class="hero-section">
        <div>{logo_html}</div>
        <h1>SolanaSense</h1>
        <p class="hero-tagline">AI-powered diagnostics for Solanaceae crops</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Glossary link ────────────────────────────────────────────────────
    col_l, col_c, col_r = st.columns([2, 1, 2])
    with col_c:
        st.page_link("pages/1_Glossary.py", label="📖 Glossary", use_container_width=True)

    # ── Artifact check (compact) ─────────────────────────────────────────
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

    # Small preview
    st.image(pil_image, caption="Selected image", use_container_width=True)

    # ── Run analysis ─────────────────────────────────────────────────────
    if not st.button("🔬 Analyse Leaf", type="primary", use_container_width=True):
        render_fact_ticker()
        return

    image_bgr = pil_to_bgr(pil_image)

    # Scanning animation
    if show_scanning:
        scan_ph = st.empty()
        render_scanning_animation(scan_ph, pil_image)

    # Actual inference
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

    # ── Result tabs for quick navigation ─────────────────────────────────
    tab_names = ["🧪 Diagnosis", "🧠 Explanation", "📊 Metrics"]
    if show_before_after:
        tab_names.append("📸 Before / After")
    if show_steps:
        tab_names.append("🔍 Pipeline")
    tab_names += ["🌱 Root Causes", "📋 Recommendations"]

    result_tabs = st.tabs(tab_names)
    tab_idx = 0

    # ── Tab: Diagnosis ───────────────────────────────────────────────────
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

    # ── Tab: Explanation ─────────────────────────────────────────────────
    with result_tabs[tab_idx]:
        render_prediction_explanation(label, result)
    tab_idx += 1

    # ── Tab: Metrics ─────────────────────────────────────────────────────
    with result_tabs[tab_idx]:
        render_severity_metrics(result)
    tab_idx += 1

    # ── Tab: Before / After (conditional) ────────────────────────────────
    if show_before_after:
        with result_tabs[tab_idx]:
            render_before_after(result)
        tab_idx += 1

    # ── Tab: Pipeline (conditional) ──────────────────────────────────────
    if show_steps:
        with result_tabs[tab_idx]:
            render_pipeline_steps(result)
        tab_idx += 1

    # ── Tab: Root Causes ─────────────────────────────────────────────────
    with result_tabs[tab_idx]:
        render_root_causes(label)
    tab_idx += 1

    # ── Tab: Recommendations ─────────────────────────────────────────────
    with result_tabs[tab_idx]:
        render_recommendations(label)

    # ── Fact ticker at bottom ────────────────────────────────────────────
    render_fact_ticker()


def render_fact_ticker():
    """Render a scrolling banner with ALL Solanaceae facts at the bottom."""
    separator = "&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;"
    facts_html = separator.join(f"🌿 {f}" for f in SOLANA_FACTS)
    # Duplicate the full string so the loop is seamless
    full_text = f"{facts_html}{separator}{facts_html}"
    st.markdown(f"""
    <div class="fact-ticker-wrap">
        <div class="fact-ticker-track">
            <span>{full_text}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()