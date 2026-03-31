"""
SolanaSense — Glossary of Terms
================================
A beginner-friendly reference for the technical terms
used throughout the SolanaSense analysis interface.
"""

import streamlit as st

st.set_page_config(
    page_title="SolanaSense — Glossary",
    page_icon="📖",
    layout="centered",
)

# ── Read dark mode from shared session state ─────────────────────────────
# The toggle on main.py stores its value under key "dark_mode" in session_state.
# We add a mirrored toggle here so users can also switch on this page.
with st.sidebar:
    dark = st.toggle("🌙 Dark mode", value=st.session_state.get("dark_mode", False),
                      key="dark_mode")

# ── Colour tokens ────────────────────────────────────────────────────────
if dark:
    app_bg = "linear-gradient(170deg, #121212 0%, #1E1E1E 50%, #181818 100%)"
    txt = "#E0E0E0"
    txt_sub = "#BDBDBD"
    card_bg = "rgba(40,40,40,0.90)"
    card_border = "#4CAF50"
    card_title = "#81C784"
    cat_border = "#4CAF50"
else:
    app_bg = "linear-gradient(170deg, #E8F5E9 0%, #C8E6C9 50%, #A5D6A7 100%)"
    txt = "#1B4332"
    txt_sub = "#333"
    card_bg = "rgba(255,255,255,0.88)"
    card_border = "#2E7D32"
    card_title = "#1B5E20"
    cat_border = "#A5D6A7"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
*:not([class*="material"]):not([data-testid="stIconMaterial"]) {{ font-family: 'Poppins', sans-serif !important; }}
[data-testid="collapsedControl"] span {{ font-family: 'Material Symbols Rounded' !important; }}

.stApp {{ background: {app_bg}; }}
h1, h2, h3, p, label, div, span {{ color: {txt} !important; }}

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

/* ---- Dark mode overrides for widgets ---- */
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary span,
[data-testid="stExpander"] summary div {{
    color: {txt} !important;
}}
[data-testid="stExpander"] [data-testid="stVerticalBlock"] {{
    {'background: rgba(30,30,30,0.6) !important;' if dark else ''}
}}
[data-testid="stTextInput"] input {{
    color: {txt} !important;
    {'background: rgba(40,40,40,0.7) !important;' if dark else ''}
}}
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] div {{
    {'color: #E0E0E0 !important; background-color: #1A1A1A;' if dark else ''}
}}
[data-testid="collapsedControl"] span {{ font-family: 'Material Symbols Rounded' !important; }}
</style>
""", unsafe_allow_html=True)

# ── Navigation back ─────────────────────────────────────────────────────
st.page_link("main.py", label="← Back to Diagnose", icon="🌿")

# ── Header ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center;padding:1rem 0 0.5rem 0;">
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


def render_glossary():
    """Render glossary terms in collapsible category expanders, optionally filtered."""
    query = search.strip().lower()
    found_any = False

    for category, terms in GLOSSARY.items():
        filtered = [
            (name, desc) for name, desc in terms
            if not query or query in name.lower() or query in desc.lower()
        ]
        if not filtered:
            continue

        found_any = True
        # When searching, expand all matching categories; otherwise collapsed
        with st.expander(category, expanded=bool(query)):
            for name, desc in filtered:
                display_name = name
                if query and query in name.lower():
                    idx = name.lower().index(query)
                    display_name = f"{name[:idx]}<strong>{name[idx:idx+len(query)]}</strong>{name[idx+len(query):]}"
                st.markdown(
                    f'<div class="glossary-card">'
                    f'<h4>{display_name}</h4>'
                    f'<p>{desc}</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    if not found_any:
        st.info(f'No terms matching "{search.strip()}" found. Try a different search.')


render_glossary()

# ── Footer ───────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "This glossary covers the technical terms used in SolanaSense. "
    "Definitions are simplified for accessibility — consult specialised "
    "references for full scientific detail."
)
