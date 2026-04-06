"""
SolanaSense — References
=========================
Cited sources for disease descriptions, recommendations,
and root-cause analyses used in SolanaSense.
"""

import streamlit as st
from pathlib import Path
import base64

st.set_page_config(
    page_title="SolanaSense — References",
    page_icon="📚",
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

.ref-card {{
    background: {card_bg};
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 10px;
    border-left: 4px solid {card_border};
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}}
.ref-card h4 {{
    margin: 0 0 4px 0;
    font-size: 1rem;
    color: {card_title} !important;
}}
.ref-card p {{
    margin: 0;
    font-size: 0.92rem;
    color: {txt_sub} !important;
    line-height: 1.6;
}}
.ref-card a {{
    color: #2E7D32 !important;
    text-decoration: underline;
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
    <h1 style="font-size:2.5rem !important;font-weight:800 !important;margin:0 !important;">📚 References</h1>
    <p style="font-size:1.05rem;color:{txt_sub} !important;margin:0.3rem 0 0 0;">
        Sources underpinning the disease descriptions, recommendations, and root-cause analyses in SolanaSense.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Reference data ───────────────────────────────────────────────────────
REFERENCES = {
    "🦠 Bacterial Spot (Pepper & Tomato)": [
        {
            "title": "Jones, J.B., Lacy, G.H., Bouzar, H., Stall, R.E. & Schaad, N.W. (2004)",
            "detail": "Reclassification of the Xanthomonads associated with bacterial spot disease of tomato and pepper. <em>Systematic and Applied Microbiology</em>, 27(6), 755–762.",
            "url": "https://doi.org/10.1078/0723202042369884",
        },
        {
            "title": "Potnis, N., Timilsina, S., Strayer, A. et al. (2015)",
            "detail": "Bacterial spot of tomato and pepper: diverse <em>Xanthomonas</em> species with a wide variety of virulence factors posing a worldwide challenge. <em>Molecular Plant Pathology</em>, 16(9), 907–920.",
            "url": "https://doi.org/10.1111/mpp.12244",
        },
        {
            "title": "UF/IFAS Extension — Bacterial Spot of Tomato",
            "detail": "University of Florida IFAS Extension fact sheet on identification, biology, and integrated management of bacterial spot in tomatoes.",
            "url": "https://edis.ifas.ufl.edu/publication/PP121",
        },
    ],
    "🥔 Early Blight (Potato & Tomato)": [
        {
            "title": "Adhikari, P., Oh, Y. & Panthee, D.R. (2017)",
            "detail": "Current status of early blight resistance in tomato: an update. <em>International Journal of Molecular Sciences</em>, 18(10), 2019.",
            "url": "https://doi.org/10.3390/ijms18102019",
        },
        {
            "title": "Cornell University Cooperative Extension",
            "detail": "Early Blight of Tomato and Potato — disease cycle, symptoms, and recommended fungicide programmes.",
            "url": "https://vegetables.cornell.edu/pest-management/disease-factsheets/early-blight-of-tomato/",
        },
        {
            "title": "Notteghem, J.L. & Andrivon, D. (1999)",
            "detail": "<em>Alternaria solani</em>: biology, epidemiology, and management strategies. <em>Agronomie</em>, 19(3–4), 305–317.",
            "url": "https://doi.org/10.1051/agro:19990308",
        },
    ],
    "🥔 Late Blight (Potato & Tomato)": [
        {
            "title": "Fry, W.E. (2008)",
            "detail": "<em>Phytophthora infestans</em>: the plant (and R gene) destroyer. <em>Molecular Plant Pathology</em>, 9(3), 385–402.",
            "url": "https://doi.org/10.1111/j.1364-3703.2007.00465.x",
        },
        {
            "title": "Haverkort, A.J., Boonekamp, P.M., Hutten, R. et al. (2008)",
            "detail": "Societal costs of late blight in potato and prospects of durable resistance through cisgenic modification. <em>Potato Research</em>, 51(1), 47–57.",
            "url": "https://doi.org/10.1007/s11540-008-9089-y",
        },
        {
            "title": "USAblight — Late Blight Decision Support System",
            "detail": "A US-wide monitoring project providing real-time tracking of <em>Phytophthora infestans</em> genotypes and management guidance.",
            "url": "https://usablight.org/",
        },
    ],
    "🍅 Septoria Leaf Spot": [
        {
            "title": "Elmer, W.H. & Ferrandino, F.J. (1995)",
            "detail": "Influence of spore density, leaf age, temperature, and dew periods on Septoria leaf spot of tomato. <em>Plant Disease</em>, 79(3), 287–290.",
            "url": "https://doi.org/10.1094/PD-79-0287",
        },
        {
            "title": "Clemson Cooperative Extension — Tomato Diseases",
            "detail": "Fact sheet covering Septoria leaf spot identification, cultural controls, and fungicide recommendations for home and commercial growers.",
            "url": "https://hgic.clemson.edu/factsheet/tomato-diseases/",
        },
    ],
    "🍅 Target Spot": [
        {
            "title": "MacKenzie, K.J., Sumabat, L.G., Xavier, K.V. & Vallad, G.E. (2018)",
            "detail": "A review of <em>Corynespora cassiicola</em> and its increasing relevance to tomato in Florida. <em>Plant Health Progress</em>, 19(4), 303–309.",
            "url": "https://doi.org/10.1094/PHP-05-18-0023-RV",
        },
        {
            "title": "UF/IFAS Extension — Target Spot of Tomato",
            "detail": "University of Florida fact sheet on target spot: symptoms, disease cycle, and integrated management strategies.",
            "url": "https://edis.ifas.ufl.edu/publication/PP104",
        },
    ],
    "📘 General Plant Pathology": [
        {
            "title": "Agrios, G.N. (2005)",
            "detail": "<em>Plant Pathology</em>, 5th Edition. Academic Press. — Comprehensive textbook covering the biology and management of plant diseases.",
            "url": "https://www.elsevier.com/books/plant-pathology/agrios/978-0-12-044565-3",
        },
        {
            "title": "APS Press — Compendium of Tomato Diseases and Pests (2nd Ed.)",
            "detail": "Disease-by-disease reference for tomato, including colour photographs, pathogen biology, and control strategies.",
            "url": "https://www.apsnet.org/edcenter/resources/commonnames/Pages/Tomato.aspx",
        },
        {
            "title": "APS Press — Compendium of Potato Diseases (2nd Ed.)",
            "detail": "Authoritative reference on potato diseases with management recommendations for commercial growers.",
            "url": "https://www.apsnet.org/edcenter/resources/commonnames/Pages/Potato.aspx",
        },
    ],
    "🧪 Copper-Based & Fungicide Recommendations": [
        {
            "title": "Abbasi, P.A. & Weselowski, B. (2015)",
            "detail": "Efficacy of copper-based bactericides for control of bacterial spot on tomato under greenhouse and field conditions. <em>Plant Disease</em>, 99(1), 62–70.",
            "url": "https://doi.org/10.1094/PDIS-05-14-0487-RE",
        },
        {
            "title": "McGrath, M.T. — Cornell University",
            "detail": "Managing diseases of tomatoes with fungicides — practical guidelines for timing, product selection, and resistance management.",
            "url": "https://vegetables.cornell.edu/pest-management/disease-factsheets/fungicides-for-managing-diseases-of-tomato/",
        },
    ],
    "🖼️ Image Processing & Machine Learning": [
        {
            "title": "Qin, X., Zhang, Z., Huang, C. et al. (2020)",
            "detail": "U²-Net: Going deeper with nested U-structure for salient object detection. <em>Pattern Recognition</em>, 106, 107404.",
            "url": "https://doi.org/10.1016/j.patcog.2020.107404",
        },
        {
            "title": "Howard, A., Sandler, M., Chen, B. et al. (2019)",
            "detail": "Searching for MobileNetV3. <em>Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)</em>, 1314–1324.",
            "url": "https://doi.org/10.1109/ICCV.2019.00140",
        },
        {
            "title": "Hughes, D.P. & Salathe, M. (2015)",
            "detail": "An open access repository of images on plant health to enable the development of mobile disease diagnostics. <em>arXiv preprint arXiv:1511.08060</em>.",
            "url": "https://arxiv.org/abs/1511.08060",
        },
    ],
}

# ── Search filter ────────────────────────────────────────────────────────
search = st.text_input("🔍 Search references", placeholder="Type to filter…", label_visibility="collapsed")

# ── Render ───────────────────────────────────────────────────────────────
for category, refs in REFERENCES.items():
    if search:
        filtered = [r for r in refs if search.lower() in r["title"].lower() or search.lower() in r["detail"].lower()]
        if not filtered:
            continue
    else:
        filtered = refs

    with st.expander(category, expanded=not search):
        for ref in filtered:
            link_html = f' <a href="{ref["url"]}" target="_blank">[Link]</a>' if ref.get("url") else ""
            st.markdown(f"""
            <div class="ref-card">
                <h4>{ref['title']}</h4>
                <p>{ref['detail']}{link_html}</p>
            </div>
            """, unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "These references are provided for educational purposes. "
    "Recommendations in SolanaSense are distilled from the sources above and "
    "from established university extension service guidelines. Always consult "
    "a qualified agronomist or plant pathologist for site-specific advice."
)
