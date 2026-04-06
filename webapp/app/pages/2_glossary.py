"""
SolanaSense — Glossary of Terms
================================
A beginner-friendly reference for the technical terms
used throughout the SolanaSense analysis interface.
"""

import streamlit as st
from pathlib import Path
import base64

st.set_page_config(
    page_title="SolanaSense — Glossary",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Static assets ────────────────────────────────────────────────────────
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@st.cache_resource
def _load_logo_b64():
    p = STATIC_DIR / "logo.png"
    return base64.b64encode(p.read_bytes()).decode() if p.exists() else ""


@st.cache_resource
def _load_bg_b64():
    p = STATIC_DIR / "background.png"
    return base64.b64encode(p.read_bytes()).decode() if p.exists() else ""


bg_b64 = _load_bg_b64()

# ── Colour tokens ────────────────────────────────────────────────────────
app_bg = "#ffffff"
txt = "#1B4332"
txt_sub = "#333"
card_bg = "rgba(255,255,255,0.92)"
card_border = "#2E7D32"
card_title = "#1B5E20"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
*:not([class*="material"]):not([data-testid="stIconMaterial"]) {{ font-family: 'Poppins', sans-serif !important; }}
[data-testid="collapsedControl"] span {{ font-family: 'Material Symbols Rounded' !important; }}

/* ---- Hide Streamlit chrome ---- */
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {{ display: none !important; }}
header[data-testid="stHeader"] {{ background: transparent !important; pointer-events: none !important; }}
[data-testid="collapsedControl"] {{ z-index: 10001 !important; pointer-events: auto !important; }}

.stApp {{ background: linear-gradient(170deg, rgba(232,245,233,0.82) 0%, rgba(200,230,201,0.78) 50%, rgba(165,214,167,0.82) 100%), url("data:image/png;base64,{bg_b64}") center/cover fixed no-repeat !important; }}
[data-testid="stAppViewContainer"], [data-testid="stMain"] {{ background: transparent !important; }}
.block-container {{ padding-top: 1.5rem !important; max-width: 1100px !important; }}
h1, h2, h3, p, label, div, span {{ color: {txt} !important; }}

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

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {{ background: #f5faf6 !important; }}
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] {{ display: none; }}

.glossary-card {{
    background: {card_bg};
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 10px;
    border-left: 4px solid {card_border};
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}}
.glossary-card h4 {{
    margin: 0 0 4px 0;
    font-size: 1rem;
    color: {card_title} !important;
}}
.glossary-card p {{
    margin: 0;
    font-size: 0.92rem;
    color: {txt_sub} !important;
    line-height: 1.5;
}}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary span,
[data-testid="stExpander"] summary div {{
    color: {txt} !important;
}}
[data-testid="stTextInput"] input {{
    color: {txt} !important;
}}
</style>
""", unsafe_allow_html=True)

# ── Brand header (clickable back to home) ─────────────────────────────
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

# ── Nav links ────────────────────────────────────────────────────────────
st.markdown('<div class="nav-links">', unsafe_allow_html=True)
nl, nc1, nc2, nc3, nr = st.columns([1, 1, 1, 1, 1])
with nc1:
    st.page_link("pages/1_diagnose.py", label="🔬 Diagnose", use_container_width=True)
with nc2:
    st.page_link("pages/2_glossary.py", label="📖 Glossary", use_container_width=True)
with nc3:
    st.page_link("pages/3_references.py", label="📚 References", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center;padding:0.5rem 0 0.5rem 0;">
    <h1 style="font-size:2.5rem !important;font-weight:800 !important;margin:0 !important;">📖 Glossary of Terms</h1>
    <p style="font-size:1.05rem;color:{txt_sub} !important;margin:0.3rem 0 0 0;">
        A beginner-friendly guide to the technical words and phrases used in SolanaSense.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Search filter ────────────────────────────────────────────────────────
search = st.text_input("🔍 Search terms", placeholder="Type to filter…", label_visibility="collapsed")

# ── Glossary data ────────────────────────────────────────────────────────
GLOSSARY = {
    "🌱 Plant Science": [
        ("Solanaceae",
         "A large family of flowering plants (also called nightshades) that includes tomatoes, potatoes, peppers, and eggplants. SolanaSense is built specifically for this family."),
        ("Chlorosis",
         "Yellowing of leaf tissue caused by a loss of chlorophyll — the green pigment plants use for photosynthesis. Often a sign of nutrient deficiency or disease."),
        ("Necrosis",
         "Death and browning of plant tissue. Necrotic areas on a leaf appear as dark brown or black patches and usually indicate severe cell damage."),
        ("Lesion",
         "A visible area of damaged tissue on a leaf. Lesions can be spots, patches, or rings and are one of the main visual indicators of plant disease."),
        ("Pathogen",
         "An organism that causes disease. In plant pathology, common pathogens include bacteria, fungi, and oomycetes (water moulds)."),
        ("Systemic",
         "Affecting the whole plant rather than just one spot. A systemic disease spreads through the plant's vascular system, making it harder to control."),
    ],
    "🦠 Disease & Pathogens": [
        ("Xanthomonas",
         "A genus of bacteria responsible for bacterial spot disease in peppers and tomatoes. Enters leaves through natural pores (stomata) and wounds."),
        ("Alternaria solani",
         "A fungus that causes early blight in potatoes and tomatoes. Creates distinctive concentric ring ('target') patterns on leaves."),
        ("Phytophthora infestans",
         "An oomycete (water mould, not a true fungus) that causes late blight — one of the most devastating plant diseases. Historically responsible for the Irish Potato Famine."),
        ("Septoria lycopersici",
         "A fungus that causes Septoria leaf spot in tomatoes. Produces many small circular spots with grey centres and dark edges."),
        ("Corynespora cassiicola",
         "A fungus that causes target spot disease. Produces brown lesions with concentric ring patterns on tomato leaves."),
        ("Oomycete",
         "A group of organisms that look and behave like fungi but are actually more closely related to algae. Phytophthora (late blight) is the most famous example."),
        ("Pycnidia",
         "Tiny spore-producing structures formed by certain fungi (like Septoria). They appear as small dark dots inside leaf lesions and release spores that spread the disease."),
        ("Inoculum",
         "Any material (spores, bacteria, infected debris) that can start a new infection. 'Reducing inoculum' means removing sources of disease."),
    ],
    "🖼️ Image Processing": [
        ("Background Removal",
         "A deep-learning technique (using U2-Net) that separates the leaf from its background, so the analysis focuses only on the leaf itself."),
        ("U2-Net",
         "A neural network architecture designed for image segmentation. SolanaSense uses it to cut out the leaf from the background of your photo."),
        ("HSV Colour Space",
         "A way of representing colours using Hue (colour type), Saturation (colour intensity), and Value (brightness). It makes it easier to detect specific colours like yellows and browns on leaves."),
        ("Colour Segmentation",
         "Dividing an image into regions based on colour. SolanaSense uses this to separate healthy green tissue from diseased yellow or brown areas."),
        ("Binary Mask",
         "A black-and-white image where white pixels mark regions of interest (like the leaf or diseased areas) and black pixels mark everything else."),
        ("Lanczos Interpolation",
         "A high-quality method for resizing images. SolanaSense uses Lanczos-4 to resize your leaf photo to a standard 300×300 pixel size without losing important detail."),
        ("Shadow Removal",
         "A preprocessing step that detects and corrects shadows on the leaf surface, preventing dark shadows from being mistakenly identified as disease."),
        ("Disease Overlay",
         "A visual layer drawn on top of the leaf image highlighting detected disease regions — yellow for chlorosis and red for necrosis."),
    ],
    "🤖 Machine Learning": [
        ("SVM (Support Vector Machine)",
         "The machine learning model SolanaSense uses for classification. It finds the best boundary between different disease classes in a high-dimensional feature space."),
        ("Feature Extraction",
         "The process of computing numerical measurements (109 features) from the processed leaf image — things like colour averages, texture patterns, and disease area ratios."),
        ("Confidence Score",
         "A number between 0% and 100% showing how certain the model is about its prediction. Higher is better — scores below 65% trigger a low-confidence warning."),
        ("MobileNetV3",
         "A lightweight neural network architecture designed for mobile devices. SolanaSense's CNN model uses MobileNetV3-Small for efficient image classification."),
        ("CNN (Convolutional Neural Network)",
         "A type of deep learning model specially designed for analysing images. It learns to recognise patterns like spots, rings, and colour changes on leaves."),
        ("Classification",
         "The task of assigning an input (a leaf image) to one of several predefined categories (disease classes). SolanaSense classifies into 10 classes."),
        ("Prediction Probability",
         "The model's estimated likelihood for each possible disease class. The class with the highest probability becomes the predicted diagnosis."),
    ],
    "📊 Metrics & Scores": [
        ("Severity Percentage",
         "The ratio of diseased leaf area to total leaf area, shown as a percentage. A higher number means more of the leaf is affected."),
        ("Health Score",
         "A composite score from 0 to 100 that combines severity percentage, model confidence, and whether the plant is healthy. Higher is healthier."),
        ("Diseased Pixels",
         "The count of individual image pixels identified as part of a diseased region (yellow or brown tissue)."),
        ("Total Leaf Pixels",
         "The total number of pixels that make up the leaf area in the processed image. Used as the denominator when calculating severity."),
    ],
    "🧪 Preprocessing Pipeline": [
        ("Preprocessing Pipeline",
         "The series of image transformations applied to your photo before analysis: background removal → resize → shadow removal → colour segmentation → feature extraction."),
        ("Normalisation",
         "Scaling pixel values to a standard range (typically 0–1) so the model can process images consistently regardless of lighting or camera differences."),
        ("Capsaicin",
         "The chemical compound that makes chilli peppers hot. Produced by Solanaceae plants as a natural defence against fungal infections — mentioned in the fun facts ticker."),
        ("Lycopene",
         "The red pigment in tomatoes and a powerful antioxidant. Its concentration can be affected by disease, which changes the leaf's colour profile."),
        ("Tropane Alkaloids",
         "Naturally occurring chemical compounds found in some Solanaceae plants (like deadly nightshade). Used in medicines such as atropine."),
    ],
}

# ── Render ───────────────────────────────────────────────────────────────
for category, terms in GLOSSARY.items():
    if search:
        filtered = [(t, d) for t, d in terms if search.lower() in t.lower() or search.lower() in d.lower()]
        if not filtered:
            continue
    else:
        filtered = terms

    with st.expander(category, expanded=not search):
        for term, desc in filtered:
            st.markdown(f"""
            <div class="glossary-card">
                <h4>{term}</h4>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)
