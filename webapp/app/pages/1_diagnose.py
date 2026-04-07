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
            "Remove and destroy infected leaves immediately — do not compost them",
            "Apply copper-based bactericide as a preventive spray before symptoms spread",
            "Avoid overhead watering — switch to drip irrigation to keep foliage dry",
            "Rotate crops with non-Solanaceae plants for at least 2–3 years",
            "Sanitise pruning tools and stakes with a 10% bleach solution between plants",
            "Use certified disease-free seed and transplants from reputable suppliers",
            "Apply mulch around the base to reduce rain-splash of soil-borne bacteria onto leaves",
            "Increase plant spacing to improve air circulation and reduce humidity around foliage",
        ],
        "severity_note": "Bacterial spot spreads rapidly in warm, humid conditions.",
        "root_causes": [
            "Poor soil drainage and waterlogging promoting bacterial multiplication in the root zone",
            "Excess moisture on foliage from overhead or sprinkler irrigation",
            "Weakened plant immunity due to nutrient deficiency — especially calcium and potassium",
            "Contaminated seeds or transplants introducing Xanthomonas into the growing environment",
            "Warm temperatures (24–30 °C) combined with high humidity accelerating bacterial reproduction",
            "Wounds from wind damage, insect feeding, or mechanical handling providing entry points",
            "Infected crop debris left in the field from a previous season harbouring the pathogen",
        ],
        "root_insight": "Xanthomonas bacteria enter through stomata and wounds — persistent leaf wetness and root stress are the primary systemic enablers.",
    },
    "Pepper,_bell___healthy": {
        "description": "The leaf appears healthy with no visible signs of disease or stress.",
        "features_seen": "Uniform green colouration with consistent texture. No necrotic or chlorotic regions detected.",
        "recommendations": [
            "Continue current care routine — consistency is key to plant health",
            "Monitor regularly for early signs of disease, especially on lower leaves",
            "Maintain proper spacing (45–60 cm) for good air circulation",
            "Ensure balanced fertilisation — peppers benefit from calcium-rich feeds during fruiting",
            "Water at the base of the plant to keep foliage dry and reduce disease risk",
            "Scout weekly for pests such as aphids, which can transmit viral diseases",
        ],
        "severity_note": "No action required — plant is in good health.",
        "root_causes": ["No root stress detected — plant appears systemically healthy"],
        "root_insight": "Healthy foliage indicates adequate root function, nutrient uptake, and water balance.",
    },
    "Potato___Early_blight": {
        "description": "Early blight caused by *Alternaria solani*. Produces characteristic concentric ring ('target') lesions on older leaves first.",
        "features_seen": "Brown concentric ring patterns detected with surrounding chlorosis. Lesions show distinct target-like morphology.",
        "recommendations": [
            "Remove and destroy lower infected leaves promptly to slow upward spread",
            "Apply chlorothalonil or mancozeb fungicide on a 7–10 day schedule during humid weather",
            "Improve air circulation by widening plant spacing and pruning dense foliage",
            "Avoid wetting foliage during irrigation — use drip lines or soaker hoses",
            "Rotate crops — avoid planting Solanaceae in the same bed for at least 2 years",
            "Apply organic mulch around the base to reduce soil-splash of fungal spores onto leaves",
            "Ensure adequate potassium fertilisation to strengthen cell walls and natural defences",
            "Remove all plant debris at the end of the season to eliminate overwintering fungal inoculum",
        ],
        "severity_note": "Early blight progresses from lower to upper leaves. Early intervention is critical.",
        "root_causes": [
            "Water stress and irregular watering cycles weakening the plant's natural defences",
            "Nitrogen imbalance — excess nitrogen promotes lush, soft growth that is highly susceptible",
            "Root damage from compacted or waterlogged soil reducing nutrient uptake",
            "Fungal spores (conidia) overwintering in soil and infected debris from previous crops",
            "Warm, humid conditions (24–29 °C) with alternating wet/dry periods favouring spore germination",
            "Potassium or phosphorus deficiency reducing the plant's ability to resist infection",
            "Older, lower leaves naturally losing resistance and becoming the first point of infection",
        ],
        "root_insight": "Alternaria solani thrives when plants are stressed by inconsistent watering and poor root health. Soil-borne inoculum splashes onto lower leaves first.",
    },
    "Potato___Late_blight": {
        "description": "Late blight caused by *Phytophthora infestans*. Highly destructive — causes rapid, water-soaked lesions that turn dark and spread quickly.",
        "features_seen": "Large water-soaked dark lesions with pale green to brown margins. High texture variance indicating active tissue destruction.",
        "recommendations": [
            "⚠️ Act immediately — late blight can destroy an entire crop in under two weeks",
            "Remove and destroy all infected plant material — do NOT compost it",
            "Apply systemic fungicide (e.g., metalaxyl- or mandipropamid-based) as soon as symptoms appear",
            "Ensure good drainage to reduce soil saturation around roots and tubers",
            "Monitor neighbouring potato and tomato plants closely — Phytophthora spreads via airborne spores",
            "Harvest tubers promptly once foliage dies back to prevent tuber infection from soil",
            "In future seasons, plant certified disease-free seed tubers and consider blight-resistant varieties",
            "Avoid overhead irrigation — use drip systems to minimise leaf wetness duration",
        ],
        "severity_note": "Late blight is extremely aggressive and can devastate crops within days. Immediate treatment is essential.",
        "root_causes": [
            "Cool, wet conditions (10–20 °C with persistent moisture) creating an ideal pathogen environment",
            "Infected seed tubers or volunteer plants introducing Phytophthora spores into the soil",
            "Poor field drainage keeping soil saturated around roots and tubers",
            "Dense planting reducing air flow and trapping humidity in the canopy",
            "Airborne sporangia travelling long distances from neighbouring infected fields",
            "Lack of crop rotation allowing oomycete oospores to persist in the soil between seasons",
            "Late planting extending the growing period into cooler, wetter autumn conditions",
        ],
        "root_insight": "Phytophthora infestans is an oomycete (water mould) — root-zone waterlogging and infected tubers are the primary systemic entry points.",
    },
    "Potato___healthy": {
        "description": "The potato leaf appears healthy with no visible disease symptoms.",
        "features_seen": "Uniform green colouration with consistent texture across the leaf. No abnormal colour patterns detected.",
        "recommendations": [
            "Continue current care routine — consistency supports long-term plant health",
            "Monitor lower leaves weekly for early blight signs (dark concentric-ring spots)",
            "Maintain proper hilling to protect developing tubers from light and late-blight spores",
            "Ensure consistent watering schedule — irregular watering stresses plants and invites disease",
            "Scout for Colorado potato beetle and aphids, which weaken plants and vector viruses",
            "Apply balanced fertiliser with adequate potassium to strengthen natural defences",
        ],
        "severity_note": "No action required — plant is in good health.",
        "root_causes": ["No root stress detected — plant appears systemically healthy"],
        "root_insight": "Healthy foliage indicates adequate root function, nutrient uptake, and water balance.",
    },
    "Tomato___Bacterial_spot": {
        "description": "Bacterial spot caused by *Xanthomonas* species. Creates small, dark, greasy-looking spots on leaves.",
        "features_seen": "Multiple small dark spots detected with water-soaked appearance. Brown necrotic tissue with irregular distribution.",
        "recommendations": [
            "Remove and destroy infected leaves and any fallen debris around the plant",
            "Apply copper-based spray as a preventive measure — note that copper slows but does not cure active infections",
            "Avoid working with plants when foliage is wet to prevent hand- and tool-borne spread",
            "Increase plant spacing to at least 60 cm for better air flow and faster leaf drying",
            "Use certified disease-free seeds and transplants from reputable nurseries",
            "Sanitise pruning shears, stakes, and ties with a 10% bleach or 70% alcohol solution between plants",
            "Rotate out of Solanaceae crops in the affected bed for at least 2 years",
            "Consider resistant cultivars when planning next season's planting",
        ],
        "severity_note": "Bacterial diseases cannot be cured once established — focus on slowing spread.",
        "root_causes": [
            "Poor soil drainage and persistent waterlogging around the root zone",
            "Excess moisture from overhead or sprinkler irrigation keeping leaves wet for extended periods",
            "Reduced plant resistance due to root stress, compacted soil, or nutrient imbalance",
            "Contaminated tools, hands, or clothing spreading bacteria between plants",
            "Warm, humid weather (24–30 °C with high relative humidity) accelerating bacterial reproduction",
            "Wind-driven rain splashing bacteria from soil and debris onto lower foliage",
            "Infected seed carrying Xanthomonas internally, establishing the pathogen before symptoms appear",
        ],
        "root_insight": "Prolonged leaf wetness and root-zone stress are the primary systemic enablers of bacterial foliar diseases in tomatoes.",
    },
    "Tomato___Early_blight": {
        "description": "Early blight caused by *Alternaria solani*. Produces dark, concentric-ring lesions starting on lower, older leaves.",
        "features_seen": "Brown target-shaped lesions with concentric ring patterns. Chlorotic margins surrounding necrotic centres detected.",
        "recommendations": [
            "Prune lower leaves that touch the soil to remove the primary infection pathway",
            "Apply fungicide (chlorothalonil, mancozeb, or copper-based) on a 7–10 day schedule",
            "Mulch around plants with straw or wood chips to prevent rain-splash of soil-borne spores",
            "Water at the base of plants using drip irrigation — never overhead",
            "Rotate crops annually — avoid Solanaceae in the same bed for at least 2 years",
            "Ensure adequate potassium and phosphorus fertilisation to bolster cell-wall strength",
            "Remove and destroy all plant debris at the end of the season to reduce overwintering inoculum",
            "Stake or cage plants to keep foliage upright and improve air circulation",
        ],
        "severity_note": "Early blight is common and manageable with proper fungicide timing.",
        "root_causes": [
            "Water stress and irregular watering weakening the plant's systemic defences",
            "Nutrient imbalance — especially excess nitrogen (soft, lush growth) or potassium deficit",
            "Root damage from compacted, poorly drained, or waterlogged soil",
            "Infected plant debris left in soil from prior seasons harbouring overwintering conidia",
            "Warm temperatures (24–29 °C) with high humidity or frequent rainfall events",
            "Ageing lower leaves with naturally declining resistance becoming the first point of entry",
            "Dense planting that slows air movement and extends the duration of leaf wetness",
        ],
        "root_insight": "Alternaria solani spores survive in soil and splash onto lower foliage. Stressed roots reduce the plant's systemic resistance.",
    },
    "Tomato___Late_blight": {
        "description": "Late blight caused by *Phytophthora infestans*. Causes large, irregular, water-soaked lesions that quickly turn brown-black.",
        "features_seen": "Large irregular dark patches with rapid tissue degradation. High colour variance indicating severe infection progression.",
        "recommendations": [
            "⚠️ Urgent: Remove and destroy all infected material immediately — every hour counts",
            "Apply systemic fungicide (e.g., metalaxyl, mandipropamid, or cyazofamid) as soon as symptoms appear",
            "Do NOT compost infected plant material — bag and dispose of it to prevent spore release",
            "Check all nearby tomato and potato plants for spread — Phytophthora spores travel on wind and rain",
            "Improve greenhouse ventilation or open-field air flow to reduce canopy humidity",
            "Harvest any unaffected fruit early and monitor it for secondary symptoms during storage",
            "In future seasons, choose blight-resistant cultivars and plant certified disease-free transplants",
            "Avoid overhead irrigation and water early in the day so foliage dries before nightfall",
        ],
        "severity_note": "Late blight is a crop emergency — delay can cause total loss within 1–2 weeks.",
        "root_causes": [
            "Waterlogged soil providing an ideal habitat for the oomycete pathogen",
            "Infected transplants or volunteer plants acting as initial inoculum sources in the field",
            "Cool, rainy weather (10–20 °C with persistent moisture) overwhelming natural plant defences",
            "Dense canopy trapping humidity at the leaf surface and prolonging wetness duration",
            "Airborne sporangia dispersed from neighbouring infected fields or gardens",
            "Lack of crop rotation allowing oospores to persist in the soil between seasons",
            "Overhead irrigation or frequent rainfall keeping foliage wet for extended periods (>10 hours)",
        ],
        "root_insight": "Phytophthora infestans spreads via airborne sporangia but thrives systemically when roots sit in saturated soil.",
    },
    "Tomato___Septoria_leaf_spot": {
        "description": "Septoria leaf spot caused by *Septoria lycopersici*. Creates many small, circular spots with dark borders and light centres.",
        "features_seen": "Numerous small circular lesions with grey-white centres and dark brown margins. Scattered distribution across leaf surface.",
        "recommendations": [
            "Remove and destroy affected lower leaves as soon as spots appear",
            "Apply fungicide (mancozeb, chlorothalonil, or copper-based) on a regular 7–10 day schedule",
            "Avoid overhead irrigation — switch to drip or soaker hose to keep foliage dry",
            "Stake or cage plants to keep foliage off the ground and improve air circulation",
            "Clean up all plant debris thoroughly at the end of the season to remove overwintering pycnidia",
            "Mulch around the base of plants to create a barrier between soil-borne spores and lower leaves",
            "Rotate out of tomato and other Solanaceae crops for at least 2 years in the affected bed",
            "Water early in the day so that any splashed foliage has time to dry before nightfall",
        ],
        "severity_note": "Septoria rarely kills plants but severely reduces yield through progressive defoliation.",
        "root_causes": [
            "Persistent leaf wetness from rain, overhead watering, or heavy dew",
            "Infected crop debris harbouring pycnidia (spore-producing structures) in the soil from prior seasons",
            "Poor air circulation in densely planted or un-staked rows",
            "Weakened plant immunity from nutrient deficiency (especially potassium) or water stress",
            "Warm, humid conditions (20–25 °C with extended leaf wetness >10 hours) favouring spore release",
            "Rain-splash and overhead irrigation propelling pycnidiospores from soil onto lower foliage",
            "Lack of crop rotation allowing the pathogen population to build up in the soil over seasons",
        ],
        "root_insight": "Septoria lycopersici pycnidiospores splash from soil to lower leaves — the root zone and irrigation method are key systemic factors.",
    },
    "Tomato___Target_Spot": {
        "description": "Target spot caused by *Corynespora cassiicola*. Produces circular to irregular brown spots with concentric zonation.",
        "features_seen": "Brown spots with target-like concentric rings and diffuse margins. Moderate texture irregularity across affected areas.",
        "recommendations": [
            "Remove heavily infected leaves to reduce the source of new spores",
            "Apply broad-spectrum fungicide (e.g., chlorothalonil, azoxystrobin, or mancozeb) on a 7–14 day schedule",
            "Improve plant spacing and prune lower branches to boost air circulation",
            "Reduce foliage wetness duration by watering at the base early in the day",
            "Consider resistant or tolerant varieties for next season's planting",
            "Apply organic mulch to prevent rain-splash of fungal spores from soil to lower leaves",
            "Rotate crops — avoid planting Solanaceae in the same bed for at least one year",
            "Ensure balanced fertilisation, particularly adequate potassium, to support plant resilience",
        ],
        "severity_note": "Target spot is moderately aggressive — consistent fungicide application controls spread.",
        "root_causes": [
            "High humidity and prolonged leaf wetness providing ideal conditions for spore germination",
            "Nutrient-stressed plants with reduced systemic resistance — especially potassium- or calcium-deficient",
            "Dense canopy limiting air flow, light penetration, and leaf drying after rain or dew",
            "Fungal inoculum persisting in crop residue and volunteer plants from previous seasons",
            "Warm temperatures (25–32 °C) combined with frequent rainfall or overhead irrigation events",
            "Mechanical wounds from pruning, insect feeding, or wind providing entry points for the fungus",
            "Monoculture or short-rotation cropping allowing Corynespora populations to build in the soil",
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
    rec_bg = 'rgba(255,253,231,0.65)'
    root_bg = 'rgba(232,234,246,0.60)'
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
    background: rgba(255,255,255,0.45); border-radius: 12px; padding: 16px;
    border: 2px solid {gauge_border}; text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    backdrop-filter: blur(6px); -webkit-backdrop-filter: blur(6px);
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
    background: rgba(255,255,255,0.40); border-radius: 12px;
    padding: 18px 22px; margin-bottom: 10px;
    border-left: 5px solid {brand_green};
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    backdrop-filter: blur(6px); -webkit-backdrop-filter: blur(6px);
    animation: fadeSlideUp 0.5s ease-out both;
}}
.step-card h4 {{ margin: 0 0 8px 0; font-size: 1.15rem; }}
.step-card p  {{ margin: 0; font-size: 1.02rem; line-height: 1.65; color: {txt_sub} !important; }}

.rec-card {{
    background: rgba(255,253,231,0.45); border-radius: 12px; padding: 20px 24px;
    border-left: 5px solid #F9A825; margin-top: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    backdrop-filter: blur(6px); -webkit-backdrop-filter: blur(6px);
    animation: fadeSlideUp 0.55s ease-out both;
}}
.rec-card strong {{ font-size: 1.15rem; }}
.rec-card li {{ margin-bottom: 7px; font-size: 1.02rem; line-height: 1.6; }}

.root-card {{
    background: rgba(232,234,246,0.40); border-radius: 12px; padding: 20px 24px;
    border-left: 5px solid #5C6BC0; margin-top: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    backdrop-filter: blur(6px); -webkit-backdrop-filter: blur(6px);
    animation: fadeSlideUp 0.55s ease-out both;
}}
.root-card strong {{ font-size: 1.15rem; }}
.root-card li {{ margin-bottom: 7px; font-size: 1.02rem; line-height: 1.6; }}

/* ---- Keyword highlights ---- */
.kw {{ color: #C62828; font-weight: 600; }}
.kw-green {{ color: #2E7D32; font-weight: 600; }}
.kw-amber {{ color: #E65100; font-weight: 600; }}

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
    background: rgba(255,255,255,0.45); border-radius: 14px;
    padding: 20px; margin: 12px 0;
    box-shadow: 0 4px 16px rgba(0,0,0,0.08);
    backdrop-filter: blur(6px); -webkit-backdrop-filter: blur(6px);
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


# ── Keyword highlighter ──────────────────────────────────────────────────
import re as _re

_DISEASE_KEYWORDS = [
    "Xanthomonas", "Alternaria solani", "Phytophthora infestans",
    "Septoria lycopersici", "Corynespora cassiicola",
    "bacterial spot", "early blight", "late blight",
    "Septoria leaf spot", "target spot",
    "oomycete", "pycnidia", "sporangia", "conidia", "oospores",
    "necrotic", "necrosis", "chlorotic", "chlorosis", "lesions", "lesion",
    "water-soaked", "concentric ring", "inoculum",
]
_ACTION_KEYWORDS = [
    "immediately", "urgent", "Act immediately", "do NOT", "do not compost",
    "remove and destroy", "fungicide", "bactericide", "copper-based",
    "systemic fungicide", "crop rotation", "rotate crops",
    "drip irrigation", "resistant varieties", "certified disease-free",
    "sanitise", "mulch",
]
_HEALTHY_KEYWORDS = [
    "healthy", "no visible", "uniform green", "good health",
    "no action required", "no root stress",
]


def _highlight(text: str) -> str:
    """Wrap known keywords in coloured <span> tags."""
    for kw in _HEALTHY_KEYWORDS:
        text = _re.sub(
            _re.escape(kw),
            lambda m: f'<span class="kw-green">{m.group()}</span>',
            text,
            flags=_re.IGNORECASE,
        )
    for kw in _ACTION_KEYWORDS:
        text = _re.sub(
            _re.escape(kw),
            lambda m: f'<span class="kw-amber">{m.group()}</span>',
            text,
            flags=_re.IGNORECASE,
        )
    for kw in _DISEASE_KEYWORDS:
        text = _re.sub(
            _re.escape(kw),
            lambda m: f'<span class="kw">{m.group()}</span>',
            text,
            flags=_re.IGNORECASE,
        )
    return text


def render_prediction_explanation(label: str, result):
    info = DISEASE_INFO.get(label, None)
    if info is None:
        st.info("No detailed description available for this class.")
        return
    st.markdown(f'<p style="font-size:1.08rem;line-height:1.6;">{_highlight(info["description"])}</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="step-card">
        <h4>🔬 What the model detected</h4>
        <p>{_highlight(info['features_seen'])}</p>
    </div>
    """, unsafe_allow_html=True)


def render_recommendations(label: str):
    info = DISEASE_INFO.get(label, None)
    if info is None:
        return
    recs_html = "".join(f"<li>{_highlight(r)}</li>" for r in info["recommendations"])
    st.markdown(f"""
    <div class="rec-card">
        <strong>📋 Recommended Actions</strong>
        <ul style="margin-top:10px;margin-bottom:6px">{recs_html}</ul>
        <p style="font-size:0.92rem;color:#888 !important;margin-top:10px">
            {_highlight(info['severity_note'])}
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
    causes_html = "".join(f"<li>{_highlight(c)}</li>" for c in info["root_causes"])
    st.markdown(f"""
    <div class="root-card">
        <strong>🌱 Possible Root / Systemic Causes</strong>
        <ul style="margin-top:10px;margin-bottom:6px">{causes_html}</ul>
        <p style="font-size:0.92rem;color:#666 !important;margin-top:10px">
            <strong>Insight:</strong> {_highlight(info['root_insight'])}
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
