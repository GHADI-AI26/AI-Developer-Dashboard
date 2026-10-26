"""
AI Changed the Game. But What's the Real Story?
Exploring AI Adoption, Productivity, and Trust Among Developers
Stack Overflow Developer Survey 2024  |  run with:  streamlit run app.py
Requires: streamlit, pandas, plotly   |  data: cleaned_data.csv (same folder)
Light + dark mode, sidebar filters, interactive country / developer-role explorer.
"""
from math import acos, pi, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

TITLE = "AI Changed the Game. But What's the Real Story?"
SUBTITLE = "Exploring AI Adoption, Productivity, and Trust Among Developers"

st.set_page_config(page_title=TITLE, page_icon="🌿", layout="wide", initial_sidebar_state="expanded")

# ----------------------------------------------------------------------------
# Constants (definitions follow the EDA notebook)
# ----------------------------------------------------------------------------
MIN_N = 30      # minimum valid responses for a role / group comparison
MIN_SAL_N = 20  # minimum group size used by the EDA for salary comparisons
US = "United States of America"
NO_PLAN = "No, and I don't plan to"
PLAN_SOON = "No, but I plan to soon"
EXP_ORDER = ["Beginner", "Intermediate", "Experienced", "Veteran"]
TRUST_ORDER = ["Highly trust", "Somewhat trust", "Neither trust nor distrust", "Somewhat distrust", "Highly distrust"]
PRODUCTIVITY_KEY = "Increase productivity"
ACCURACY_KEY = "Improve accuracy in coding"
DISTRUST_KEY = "trust the output or answers"  # "Don't trust the output or answers"
SECURITY_ROLE = "Security professional"
ND = "n"  # label for sample size

# documented in the EDA notebook (raw survey file is not shipped with this app)
RAW_ROWS, RAW_COLS = 65437, 114
RAW_DTYPES = {"object": 100, "float64": 13, "int64": 1}
RAW_DUPES = 0
RAW_MISSING = {  # notebook cell: missing values in the 10 selected columns
    "ConvertedCompYearly": 42002, "AIBen": 28543, "AIAcc": 28135, "AIChallenges": 27906, "AIThreat": 20748,
    "AISent": 19564, "YearsCodePro": 13827, "Country": 6507, "DevType": 5992, "AISelect": 4530,
}
SELECTED_COLS = list(RAW_MISSING.keys())

# ----------------------------------------------------------------------------
# Display labels
# ----------------------------------------------------------------------------
THREAT_EN = {"Yes": "Sees a threat", "I'm not sure": "Not sure", "No": "Not a threat"}
CH_EN = {
    "trust": "Don't trust the output or answers", "context": "Tools lack codebase and company context",
    "policy": "No right policies for security risks", "training": "Lack of training on new tools",
    "adoption": "Not everyone uses them", "morework": "They create more work (code/PRs to review)",
    "exec": "Lack of executive buy-in", "other": "Other",
}
EXP_HELP = {
    "Beginner": "under 3 years of professional coding", "Intermediate": "3 to under 10 years",
    "Experienced": "10 to under 20 years", "Veteran": "20+ years",
}


def ch_key(s):
    sl = str(s).lower()
    for needle, key in [("trust the output", "trust"), ("lack context", "context"), ("policies", "policy"),
                        ("proper training", "training"), ("not everyone", "adoption"),
                        ("more work", "morework"), ("executive", "exec")]:
        if needle in sl:
            return key
    return "other"


def ch_name(k):
    return CH_EN.get(k, k)


def ai_status_name(k):
    return {"Uses AI": "Uses AI", "Plans to soon": "Plans to soon",
            "Doesn't plan to": "Doesn't plan to"}[k]


# ----------------------------------------------------------------------------
# Palettes
# ----------------------------------------------------------------------------
LIGHT = dict(
    page=("radial-gradient(900px 600px at 88% 6%, rgba(198,227,74,.09), transparent 60%),"
          "radial-gradient(900px 700px at 4% 92%, rgba(11,69,56,.12), transparent 60%),"
          "linear-gradient(155deg,#F3F6F0 0%,#E6EFE8 40%,#D3E4DB 100%)"),
    side="linear-gradient(180deg,#EDF3EE 0%,#DAE8DF 100%)",
    card="rgba(251,253,249,.72)", card_solid="#FBFDF9", ink="#14301F", muted="#4F6359", line="rgba(8,48,42,.16)",
    forest="#1F4D36", sage="#8FB391", lime="#C6E34A", limeink="#14301F", warn="#B85C3C",
    heroink="#F6F1E4", grid="rgba(31,77,54,.10)", on_forest="#F6F1E4", on_sage="#14301F",
    hero="linear-gradient(135deg,#041A17 0%,#08302A 52%,#0B4538 100%)",
    panel="linear-gradient(160deg,#041A17 0%,#08302A 55%,#0B4538 100%)",
    hi_trust="#1F4D36", so_trust="#8FB391", neutral="#D6D0B8", so_dis="#E3A98D", hi_dis="#B85C3C",
)
DARK = dict(
    page=("radial-gradient(1000px 640px at 88% 2%, rgba(70,170,140,.15), transparent 62%),"
          "radial-gradient(1000px 700px at 4% 96%, rgba(40,120,100,.20), transparent 62%),"
          "radial-gradient(500px 360px at 96% 70%, rgba(198,227,74,.04), transparent 70%),"
          "linear-gradient(165deg,#031512 0%,#052520 45%,#08342B 100%)"),
    side="linear-gradient(180deg,#031512 0%,#062A24 100%)",
    card="rgba(9,48,40,.58)", card_solid="#0A2E27", ink="#EFEBDD", muted="#A9C0B3", line="rgba(143,179,145,.20)",
    forest="#6FA383", sage="#4F7A5F", lime="#C6E34A", limeink="#0C1B13", warn="#E08A66",
    heroink="#EFEBDD", grid="rgba(239,235,221,.09)", on_forest="#0C1B13", on_sage="#EFEBDD",
    hero="linear-gradient(135deg,#031512 0%,#062A24 52%,#093D32 100%)",
    panel="linear-gradient(160deg,#031512 0%,#062A24 55%,#093D32 100%)",
    hi_trust="#C6E34A", so_trust="#6FA383", neutral="#3D5B52", so_dis="#C9825F", hi_dis="#E8956F",
)

if "dark" not in st.session_state:
    st.session_state["dark"] = True  # opens in the deep cool-green atmosphere of the hero
DARK_MODE = bool(st.session_state["dark"])
C = DARK if DARK_MODE else LIGHT
FONT = "Inter, sans-serif"


def html(block: str) -> None:
    """Render HTML without markdown treating indented lines as code blocks."""
    cleaned = "\n".join(line.strip() for line in block.strip().splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"

# ----------------------------------------------------------------------------
# CSS (theme tokens are substituted; Arabic adds RTL + Arabic typography)
# ----------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,300;9..144,400;9..144,600;9..144,700&family=Inter:wght@400;500;600&display=swap');
:root {--f:'Inter',sans-serif; --fs:'Fraunces',Georgia,serif;}
html, body, .stApp, [data-testid="stAppViewContainer"] {color:%%ink%%;}
.stApp {background:%%page%% !important; background-attachment:fixed !important;}
[data-testid="stAppViewContainer"], [data-testid="stMain"] {background:transparent !important;}
[data-testid="stHeader"] {background:transparent !important;}
[data-testid="stMainMenu"], [data-testid="stDeployButton"], [data-testid="stAppDeployButton"], [data-testid="stStatusWidget"], #MainMenu, footer {visibility:hidden;}
[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"], [data-testid="stSidebarCollapseButton"], [data-testid="stSidebarHeader"] {visibility:visible !important; display:flex !important; z-index:1000002;}
[data-testid="stExpandSidebarButton"] *, [data-testid="stSidebarCollapsedControl"] *, [data-testid="collapsedControl"] *, [data-testid="stSidebarCollapseButton"] * {visibility:visible !important;}
.block-container {max-width:1180px; padding-top:1.2rem; padding-bottom:5rem;}
html {scroll-behavior:smooth;}
body, .stApp, .stMarkdown, .stMarkdown *, [data-testid="stWidgetLabel"] *, [data-testid="stSidebar"] .stMarkdown *, .stButton button *, input, [data-baseweb="select"] *, [data-testid="stExpander"] summary p, [data-testid="stCaptionContainer"] * {font-family:var(--f);}
h1,h2,h3,h4,.serif {font-family:var(--fs) !important; color:%%ink%% !important; letter-spacing:-0.01em;}
p, li, label, span, div[data-testid="stMarkdownContainer"] {color:%%ink%%;}
.stApp label p, .stApp [data-testid="stWidgetLabel"] p {color:%%ink%% !important; font-weight:500;}

/* sidebar */
[data-testid="stSidebar"] {background:%%side%% !important; border-right:1px solid %%line%%;}
[data-testid="stSidebar"] * {color:%%ink%%;}
[data-testid="stSidebar"][aria-expanded="true"] {width:340px !important; min-width:340px !important; max-width:340px !important;}
@media (max-width:900px){[data-testid="stSidebar"][aria-expanded="true"] {width:min(340px, 88vw) !important; min-width:0 !important;}}
[data-testid="stSidebar"] [data-testid="stSidebarHeader"] {padding:.6rem 1rem .2rem 1rem !important; height:auto !important; min-height:0;}
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {padding:.2rem 1.1rem 1.4rem 1.1rem !important;}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {gap:.5rem !important;}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p, [data-testid="stSidebar"] label p {font-size:12px !important; font-weight:500 !important; letter-spacing:.02em; color:%%muted%% !important; margin:0 0 2px 0 !important; line-height:1.3 !important;}
/* filter controls: precise rules only (no wildcard or nested-element overrides), light field + dark text in both themes so every internal part stays legible */
[data-testid="stSidebar"] [data-testid="stMultiSelect"] [data-baseweb="select"] > div {background-color:#EEF2E8; border:1px solid rgba(143,179,145,.55); border-radius:10px; color:#14301F;}
[data-testid="stSidebar"] [data-testid="stMultiSelect"] [data-baseweb="select"] > div:focus-within {border-color:#C6E34A; box-shadow:0 0 0 1px rgba(198,227,74,.45);}
[data-testid="stSidebar"] [data-testid="stMultiSelect"] [data-baseweb="select"] input {color:#14301F; -webkit-text-fill-color:#14301F;}
[data-testid="stSidebar"] [data-testid="stMultiSelect"] [data-baseweb="select"] input::placeholder {color:#5C6E63; -webkit-text-fill-color:#5C6E63; opacity:1;}
[data-testid="stSidebar"] [data-testid="stMultiSelect"] [data-baseweb="tag"] {background-color:#8FB391; color:#14301F;}
[data-testid="stSidebar"] [data-testid="stMultiSelect"] [data-baseweb="tag"] span {color:#14301F;}
[data-testid="stSidebar"] [data-testid="stMultiSelect"] [data-baseweb="tag"] svg {fill:#14301F;}
[data-testid="stSidebar"] [data-testid="stMultiSelect"] [data-baseweb="select"] svg {fill:#4F6359;}
[data-baseweb="popover"] [data-baseweb="menu"], [data-baseweb="popover"] ul {background:%%card_solid%% !important;}
[data-baseweb="popover"] li, [data-baseweb="popover"] li * {color:%%ink%% !important;}
[data-baseweb="popover"] li:hover, [data-baseweb="popover"] li[aria-selected="true"] {background:rgba(198,227,74,.14) !important;}
[data-testid="stSidebar"] [data-testid="stSlider"] {padding:0 .3rem; filter:hue-rotate(78deg) saturate(.7);}
[data-testid="stSidebar"] .stButton > button {background:transparent; border:1px solid %%line%%; border-radius:10px; padding:.3rem .8rem; min-height:0; box-shadow:none; width:100%;}
[data-testid="stSidebar"] .stButton > button p {font-size:12.5px; color:%%ink%%;}
[data-testid="stSidebar"] .stButton > button:hover {border-color:rgba(198,227,74,.7); background:rgba(198,227,74,.08);}
.sb-card {background:%%card%%; border:1px solid %%line%%; border-radius:12px; padding:11px 13px; margin:0 0 4px 0; line-height:1.45;}
.sb-card .t {font-weight:600; font-size:13px; color:%%ink%% !important; line-height:1.3;}
.sb-card .m {color:%%forest%% !important; font-size:12px; margin:3px 0 6px 0;}
.sb-card .d {color:%%muted%% !important; font-size:12px;}
.sb-h {font-family:var(--fs); font-size:15px; font-weight:600; margin:4px 0 8px 0; color:%%ink%% !important; white-space:nowrap; line-height:1.2;}
.sb-h::before {content:""; display:inline-block; vertical-align:middle; width:12px; height:2px; background:#C6E34A; border-radius:2px; margin-right:8px;}
.sb-hint {font-size:12px !important; line-height:1.4; color:%%muted%% !important; margin:-2px 0 4px 0;}
.sb-count {background:%%card%%; border:1px solid %%line%%; border-left:3px solid #C6E34A; border-radius:12px; padding:9px 13px; margin-top:6px; line-height:1.3;}
.sb-count .l {font-size:11.5px; color:%%muted%% !important; letter-spacing:.03em;}
.sb-count .n {font-family:var(--fs); font-size:22px; font-weight:600; color:%%ink%% !important; line-height:1.15;}
.sb-count .p {font-size:11.5px; color:%%muted%% !important;}

/* navigation */
.nav {display:flex; gap:4px; flex-wrap:wrap; align-items:center; padding:8px 0 16px 0; border-bottom:1px solid %%line%%;}
.nav a {text-decoration:none !important; color:%%muted%% !important; font-size:.78rem; padding:5px 10px; border-radius:99px; border:1px solid transparent; transition:all .2s;}
.nav a:hover {border-color:%%forest%%; color:%%ink%% !important; background:%%card%%;}
.brand {font-family:var(--fs); font-weight:700; font-size:1rem; color:%%ink%%; margin-right:12px;}

/* hero */
.hero {position:relative; background:%%hero%%; border-radius:32px; padding:64px 60px 56px 64px; margin:20px 0 6px 0; display:flex; gap:40px; align-items:center; justify-content:space-between; overflow:hidden; border:1px solid rgba(143,179,145,.22); box-shadow:0 30px 70px -30px rgba(2,18,14,.7), inset 0 1px 0 rgba(246,241,228,.06);}
.hero::before {content:""; position:absolute; inset:0; background:radial-gradient(620px 420px at 86% 18%, rgba(70,170,140,.20), transparent 65%), radial-gradient(460px 320px at 4% 112%, rgba(40,120,100,.28), transparent 70%), radial-gradient(260px 200px at 92% 96%, rgba(198,227,74,.07), transparent 70%); pointer-events:none;}
.hero::after {content:""; position:absolute; inset:0; background-image:linear-gradient(rgba(246,241,228,.028) 1px, transparent 1px), linear-gradient(90deg, rgba(246,241,228,.028) 1px, transparent 1px); background-size:44px 44px; -webkit-mask-image:linear-gradient(100deg, transparent 20%, #000 90%); mask-image:linear-gradient(100deg, transparent 20%, #000 90%); pointer-events:none;}
.hero > * {position:relative; z-index:1;}
.hero .copy {flex:1 1 auto; min-width:0;}
.hero .neural {flex:0 1 400px; min-width:220px; max-width:400px; width:100%;} .hero .neural svg {width:100%; height:auto; display:block;}
.hero .eyebrow {display:flex; align-items:center; gap:12px; letter-spacing:.22em; text-transform:uppercase; font-size:.7rem; color:#B9CDBE !important; font-weight:600;}
.hero .eyebrow::before {content:""; width:34px; height:2px; background:#C6E34A; border-radius:2px; flex:none;}
.hero h1 {font-family:var(--fs) !important; color:#F6F1E4 !important; font-size:3.5rem; line-height:1.04; margin:22px 0 20px 0; font-weight:500; letter-spacing:-.025em; font-variation-settings:"opsz" 144;}
.hero h1 em {display:block; color:#C6E34A !important; font-style:italic; font-weight:300; letter-spacing:-.02em;}
.hero .sub {font-family:var(--f); font-size:.98rem; font-weight:500; letter-spacing:.04em; color:#CFE0D3 !important; margin:0 0 18px 0; padding-left:14px; border-left:2px solid rgba(198,227,74,.7);}
.hero p {color:#B8CABD !important; font-size:.98rem; line-height:1.7; max-width:540px;}
.hero .chips {display:flex; gap:10px; flex-wrap:wrap; margin-top:24px;}
.hero .chip {border:1px solid rgba(143,179,145,.35); border-radius:99px; padding:7px 14px; font-size:.8rem; color:#DCE8DF !important; background:rgba(143,179,145,.08);}
.hero .chip b {color:#C6E34A !important;}

/* section headings */
.sec {display:flex; gap:22px; align-items:flex-end; margin-top:92px; margin-bottom:10px;}
.sec .num {font-family:var(--fs); font-size:6rem; line-height:.8; font-weight:300; color:transparent !important; -webkit-text-stroke:1.5px %%forest%%; opacity:.45;}
.sec .tag {font-size:.72rem; letter-spacing:.18em; text-transform:uppercase; color:%%forest%% !important; font-weight:600;}
.sec .question {font-family:var(--fs); font-size:2.15rem; line-height:1.15; margin-top:6px; color:%%ink%% !important; font-weight:600;}
.lede {color:%%muted%% !important; font-size:1.02rem; margin:4px 0 20px 0; max-width:800px; line-height:1.6;}

/* cards, stats, insights */
.card {background:%%card%%; border:1px solid %%line%%; border-radius:22px; padding:20px 22px; backdrop-filter:blur(10px); -webkit-backdrop-filter:blur(10px); margin-bottom:12px;}
.card.full {border-top:4px solid %%forest%%;}
.card.sel {border-top:4px solid %%lime%%;}
.stat {font-family:var(--fs); font-size:2.3rem; font-weight:600; line-height:1; color:%%ink%% !important;}
.stat.lime {background:%%lime%%; color:%%limeink%% !important; display:inline-block; padding:2px 12px; border-radius:12px;}
.stat-label {font-size:.86rem; color:%%muted%% !important; margin-top:9px; line-height:1.5;}
.kpis {display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px;}
.kpis .card {margin-bottom:0;}
.grp {font-size:.72rem; letter-spacing:.14em; text-transform:uppercase; font-weight:600; margin:18px 0 8px 0; color:%%forest%% !important;}
.giant {font-family:var(--fs); font-size:8rem; line-height:.9; font-weight:300; color:%%ink%% !important; letter-spacing:-.04em;}
.giant small {font-size:2.8rem; color:%%forest%% !important; margin-left:4px;}
.giant-cap {font-size:1.02rem; color:%%muted%% !important; max-width:340px; line-height:1.55; margin-top:10px;}
.insight {border-left:4px solid %%lime%%; background:%%card%%; padding:16px 22px; border-radius:0 18px 18px 0; margin:14px 0 6px 0; line-height:1.65; backdrop-filter:blur(10px);}
.insight b {color:%%ink%% !important;}
.why {color:%%muted%% !important; font-size:.94rem; margin:6px 0 0 4px; line-height:1.65;}
.why b {color:%%forest%% !important;}
.note {font-size:.8rem; color:%%muted%% !important; line-height:1.55; margin-top:8px;}
.note b {color:%%ink%% !important;}
.lowdata {border:1px dashed %%forest%%; border-radius:18px; padding:18px 22px; color:%%muted%% !important; background:%%card%%; margin:10px 0;}
.dot {display:inline-block; width:11px; height:11px; border-radius:50%; margin:0 6px 0 14px; vertical-align:middle;}
.badge {display:inline-block; font-size:.66rem; letter-spacing:.08em; text-transform:uppercase; font-weight:600; padding:3px 9px; border-radius:99px; margin-left:6px;}
.badge.ok {background:%%lime%%; color:%%limeink%% !important;}
.badge.doc {border:1px solid %%forest%%; color:%%forest%% !important;}

/* before / after */
.ba {display:grid; grid-template-columns:1fr 70px 1fr; gap:10px; align-items:stretch;}
.ba .arrow {display:flex; align-items:center; justify-content:center; font-size:2.2rem; color:%%forest%% !important;}
.ba .side {border-radius:24px; padding:26px 28px; border:1px solid %%line%%;}
.ba .before {background:%%card%%; border-style:dashed; border-color:%%forest%%;}
.ba .after {background:%%panel%%;}
.ba .after * {color:#EFEBDD !important;}
.ba .after .stat {color:#C6E34A !important;}
.ba .lbl {font-size:.72rem; letter-spacing:.16em; text-transform:uppercase; font-weight:600; color:%%forest%% !important;}
.ba .after .lbl {color:#C6E34A !important;}
.ba .big {font-family:var(--fs); font-size:3rem; font-weight:600; line-height:1.05; margin:8px 0;}
.ba .line {font-size:.88rem; line-height:1.7; color:%%muted%% !important;}
.steps {display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:12px; margin-top:12px;}
.step .h {font-family:var(--fs); font-size:1.1rem; font-weight:600; margin-bottom:6px;}
.step .b {font-size:.86rem; line-height:1.6; color:%%muted%% !important;}
.step .b b {color:%%ink%% !important;}

/* productivity paradox centrepiece */
.st-key-centerpiece {background:%%panel%%; border-radius:34px; padding:46px 48px 38px 48px; margin:34px -18px 0 -18px; border:1px solid rgba(198,227,74,.25); box-shadow:0 40px 80px -40px rgba(0,20,10,.65);}
.pc-tag {font-size:.72rem; letter-spacing:.22em; text-transform:uppercase; color:#C6E34A !important; font-weight:600;}
.pc-title {font-family:var(--fs); font-size:2.6rem; line-height:1.1; font-weight:600; color:#F6F1E4 !important; margin:8px 0 10px 0;}
.pc-title em {color:#C6E34A !important; font-style:italic; font-weight:400;}
.pc-lede {color:#C9D6CB !important; font-size:1.02rem; line-height:1.65; max-width:760px;}
.pc-big {font-family:var(--fs); font-size:7.2rem; line-height:.9; font-weight:300; color:#C6E34A !important; letter-spacing:-.04em;}
.pc-big small {font-size:2.6rem; color:#EFEBDD !important;}
.pc-cap {color:#EAE6D6 !important; font-size:1rem; line-height:1.6; margin-top:10px;}
.pc-cap b {color:#FFFFFF !important;}
.pc-note {font-size:.8rem; color:#A9BBAE !important; line-height:1.55; margin-top:10px;}
.pc-note b {color:#EFEBDD !important;}
.pc-card {border:1px solid rgba(239,235,221,.16); border-radius:20px; padding:18px 20px; background:rgba(255,255,255,.04); margin-top:14px;}
.pc-card .t {font-family:var(--fs); font-size:1.05rem; font-weight:600; color:#F6F1E4 !important;}
.pc-card .b {font-size:.84rem; color:#C9D6CB !important; line-height:1.55; margin-top:5px;}

/* story strip */
.story {display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); border-top:1px solid %%line%%; border-bottom:1px solid %%line%%; margin:22px 0;}
.story .it {padding:22px 20px; border-right:1px solid %%line%%; border-bottom:1px solid %%line%%;}
.story .n {font-family:var(--fs); font-size:2.3rem; font-weight:600; color:%%ink%% !important; line-height:1;}
.story .l {font-size:.82rem; color:%%muted%% !important; margin-top:9px; line-height:1.5;}
.conclusion {background:%%panel%%; border-radius:28px; padding:42px 48px; margin-top:18px; border:1px solid rgba(198,227,74,.22);}
.conclusion h3 {color:#C6E34A !important; margin-top:0; font-size:1.7rem;}
.conclusion p {color:#EAE6D6 !important; font-size:1.1rem; line-height:1.75;}

/* widgets */
.stButton > button, .stFormSubmitButton > button {background:%%forest%% !important; color:%%on_forest%% !important; border:none; border-radius:99px; padding:.55rem 1.6rem; font-weight:600; letter-spacing:.02em;}
.stButton > button:hover, .stFormSubmitButton > button:hover {background:%%lime%% !important; color:%%limeink%% !important;}
.stFormSubmitButton > button p, .stButton > button p {color:inherit !important;}
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div, .stTextInput input {background:%%card_solid%% !important; color:%%ink%% !important; border-color:%%line%% !important; border-radius:14px;}
div[data-baseweb="select"] span, div[data-baseweb="select"] div, .stTextInput input {color:%%ink%% !important;}
div[data-baseweb="popover"] ul, div[data-baseweb="popover"] li {background:%%card_solid%% !important; color:%%ink%% !important;}
span[data-baseweb="tag"] {background:%%lime%% !important;} span[data-baseweb="tag"] * {color:%%limeink%% !important;}
.stRadio div[role="radiogroup"] label p, .stToggle p {color:%%ink%% !important;}
[data-testid="stForm"] {background:%%card%%; border:1px solid %%line%% !important; border-radius:24px; padding:22px 26px; backdrop-filter:blur(10px);}
[data-testid="stExpander"] {background:%%card%%; border:1px solid %%line%% !important; border-radius:18px;}
[data-testid="stSlider"] [role="slider"] {background:%%lime%% !important; border:2px solid %%forest%% !important; box-shadow:none !important;}
[data-testid="stSlider"] [data-testid="stSliderThumbValue"], [data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"] {color:%%ink%% !important;}
[data-testid="stSlider"] div[data-baseweb="slider"] > div > div {background:linear-gradient(90deg,%%forest%%,%%forest%%) !important;}
hr {border-color:%%line%%;}
@media (max-width:900px){.hero{flex-direction:column; padding:36px 26px;} .hero h1{font-size:2.3rem;} .hero .neural{max-width:360px;} .sec .question{font-size:1.5rem;} .sec .num{font-size:3.2rem;} .giant,.pc-big{font-size:4.6rem;} .st-key-centerpiece{padding:28px 20px; margin:30px 0 0 0;} .ba{grid-template-columns:1fr;} .ba .arrow{transform:rotate(90deg);}}
</style>
"""
for _k, _v in C.items():
    CSS = CSS.replace(f"%%{_k}%%", _v)
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading survey data...")
def load_data() -> pd.DataFrame:
    path = Path(__file__).resolve().parent / "cleaned_data.csv"
    d = pd.read_csv(path)
    d["ExperienceLevel"] = pd.Categorical(d["ExperienceLevel"], categories=EXP_ORDER, ordered=True)
    return d


try:
    df = load_data()
except FileNotFoundError:
    st.error("cleaned_data.csv was not found. Place it in the same folder as app.py and rerun.")
    st.stop()

N_ALL, N_COLS = df.shape
ALL_ROLES = df["DevType"].dropna().value_counts().index.tolist()
ALL_COUNTRIES = df["Country"].dropna().value_counts().index.tolist()
TOP_ROLES = df["DevType"].value_counts().head(10).index.tolist()  # EDA definition: 10 most common DevType values
YMIN, YMAX = 0, 51

# ----------------------------------------------------------------------------
# Sidebar: dataset description + "Explore the Data" filters
# ----------------------------------------------------------------------------
if "reset_n" not in st.session_state:
    st.session_state["reset_n"] = 0


def reset_filters():
    st.session_state["reset_n"] += 1  # fresh widget keys = filters back to defaults


RN = st.session_state["reset_n"]

with st.sidebar:
    html(f"""
    <div class="sb-h">About the Data</div>
    <div class="sb-card">
    <div class="t">Stack Overflow Developer Survey 2024</div>
    <div class="m">{N_ALL:,} responses · {N_COLS} variables</div>
    <div class="d">AI adoption, productivity, and trust among developers.</div>
    </div>
    <div class="sb-h" style="margin-top:14px">Explore the Data</div>
    <div class="sb-hint">Use the filters below to explore the data.</div>
    """)
    f_roles = st.multiselect("Developer role", ALL_ROLES, key=f"f_roles_{RN}")
    f_countries = st.multiselect("Country", ALL_COUNTRIES, key=f"f_countries_{RN}")
    f_exp = st.multiselect("Experience level", EXP_ORDER, key=f"f_exp_{RN}",
                           format_func=lambda x: f"{x} ({EXP_HELP[x]})")
    f_years = st.slider("Years of professional coding", YMIN, YMAX, (YMIN, YMAX), key=f"f_years_{RN}")
    st.button("Reset Filters", on_click=reset_filters, key="reset_btn")


def _restricts(selected, options) -> bool:
    """A multiselect restricts the data only when some, but not all, options are chosen.
    Empty or all-options-selected means no restriction, so respondents with missing values stay in."""
    return bool(selected) and set(selected) != set(options)


F_ROLES_ON = _restricts(f_roles, ALL_ROLES)
F_COUNTRIES_ON = _restricts(f_countries, ALL_COUNTRIES)
F_EXP_ON = _restricts(f_exp, EXP_ORDER)
F_YEARS_ON = tuple(f_years) != (YMIN, YMAX)  # full range = unrestricted (missing years stay in)


def apply_filters(d: pd.DataFrame) -> pd.DataFrame:
    m = pd.Series(True, index=d.index)
    if F_ROLES_ON:
        m &= d["DevType"].isin(f_roles)
    if F_COUNTRIES_ON:
        m &= d["Country"].isin(f_countries)
    if F_EXP_ON:
        m &= d["ExperienceLevel"].isin(f_exp)
    if F_YEARS_ON:  # missing years are excluded only when the range is narrowed
        m &= d["YearsCodePro_clean"].between(f_years[0], f_years[1])
    return d[m]


fdf = apply_filters(df)
N_F = len(fdf)
FILTERS_ON = bool(F_ROLES_ON or F_COUNTRIES_ON or F_EXP_ON or F_YEARS_ON)
with st.sidebar:
    html(f'<div class="sb-count"><div class="l">Selected respondents</div><div class="n">{N_F:,}</div>'
         f'<div class="p">{N_F / N_ALL * 100:.1f}% of {N_ALL:,}</div></div>')

# ----------------------------------------------------------------------------
# Generic helpers
# ----------------------------------------------------------------------------


def pct(x: float, d: int = 1) -> str:
    return f"{x:.{d}f}%"


def nlab(n: int) -> str:
    return f"{ND} = {n:,}"


def wilson(k: int, n: int, z: float = 1.96):
    """95% Wilson interval for a proportion, returned in percent."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (centre - half) * 100, (centre + half) * 100


def adopt_counts(frame: pd.DataFrame):
    """AI adoption = share answering 'Yes' among respondents with a non-missing AISelect (EDA definition)."""
    s = frame["AISelect"].dropna()
    return len(s), {"Uses AI": int((s == "Yes").sum()), "Plans to soon": int((s == PLAN_SOON).sum()),
                    "Doesn't plan to": int((s == NO_PLAN).sum())}


def adoption(frame: pd.DataFrame):
    n, c = adopt_counts(frame)
    return (c["Uses AI"] / n * 100 if n else float("nan")), n


def multi_stats(frame: pd.DataFrame, col: str, keyf=lambda s: s):
    """Multi-select answers split on ';' and exploded; denominator = respondents with a non-null answer (EDA)."""
    s = frame[col].dropna()
    n = len(s)
    if n == 0:
        return pd.Series(dtype=int), pd.Series(dtype=float), 0
    cnt = s.str.split(";").explode().map(keyf).value_counts()
    return cnt, cnt / n * 100, n


def trust_dist(frame: pd.DataFrame):
    a = frame["AIAcc"].dropna()
    n = len(a)
    if n == 0:
        return pd.Series(0.0, index=TRUST_ORDER), 0
    return a.value_counts(normalize=True).reindex(TRUST_ORDER).fillna(0) * 100, n


def exp_table(frame: pd.DataFrame, min_n: int = 0) -> pd.DataFrame:
    rows = []
    for lvl in EXP_ORDER:
        n, c = adopt_counts(frame[frame["ExperienceLevel"] == lvl])
        if n and n >= min_n:
            rows.append((lvl, c["Uses AI"] / n * 100, c["Plans to soon"] / n * 100, c["Doesn't plan to"] / n * 100, n))
    return pd.DataFrame(rows, columns=["level", "adopt", "soon", "nope", "n"])


def role_table(frame: pd.DataFrame, roles, min_n: int = 0) -> pd.DataFrame:
    rows = []
    for r in roles:
        s = frame.loc[frame["DevType"] == r, "AISelect"].dropna()
        n = len(s)
        if n == 0 or n < min_n:
            continue
        k = int((s == "Yes").sum())
        lo, hi = wilson(k, n)
        rows.append((r, k / n * 100, n, lo, hi))
    return pd.DataFrame(rows, columns=["role", "adopt", "n", "lo", "hi"]).sort_values("adopt").reset_index(drop=True)


def distrust_table(frame: pd.DataFrame, min_n: int = MIN_N) -> pd.DataFrame:
    """Share answering Somewhat/Highly distrust among respondents with a non-null AIAcc, by role."""
    a = frame.dropna(subset=["AIAcc", "DevType"])
    rows = []
    for r, g in a.groupby("DevType"):
        n = len(g)
        if n < min_n:
            continue
        k = int(g["AIAcc"].isin(["Somewhat distrust", "Highly distrust"]).sum())
        lo, hi = wilson(k, n)
        rows.append((r, k / n * 100, n, lo, hi))
    return pd.DataFrame(rows, columns=["role", "dist", "n", "lo", "hi"]).sort_values("dist", ascending=False).reset_index(drop=True)


def paradox_stats(frame: pd.DataFrame) -> dict:
    """Population: respondents who answered BOTH AIBen and AIChallenges (only AI users see these questions)."""
    pop = frame[frame["AIBen"].notna() & frame["AIChallenges"].notna()]
    n = len(pop)
    prod = pop["AIBen"].str.contains(PRODUCTIVITY_KEY, regex=False)
    dist = pop["AIChallenges"].str.contains(DISTRUST_KEY, regex=False)
    both_, p_only, d_only, neither = int((prod & dist).sum()), int((prod & ~dist).sum()), int((~prod & dist).sum()), int((~prod & ~dist).sum())
    n_prod, n_dist = int(prod.sum()), int(dist.sum())
    return dict(n=n, both=both_, p_only=p_only, d_only=d_only, neither=neither, n_prod=n_prod, n_dist=n_dist,
                p_both=both_ / n * 100 if n else float("nan"), p_prod=n_prod / n * 100 if n else float("nan"),
                p_dist=n_dist / n * 100 if n else float("nan"),
                p_dist_given_prod=both_ / n_prod * 100 if n_prod else float("nan"))


# --- salary definitions (EDA): IQR rule on ALL respondents, applied to salary analyses only ---
_q1, _q3 = df["ConvertedCompYearly"].quantile(0.25), df["ConvertedCompYearly"].quantile(0.75)
SAL_Q1, SAL_Q3 = float(_q1), float(_q3)
SAL_LOWER, SAL_UPPER = SAL_Q1 - 1.5 * (SAL_Q3 - SAL_Q1), SAL_Q3 + 1.5 * (SAL_Q3 - SAL_Q1)


def sal_clean(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["ConvertedCompYearly"].between(0, SAL_UPPER)]


# --- full-survey benchmark values (used by hero, Find Yourself, Trust Test and the final summary) ---
N_SEL, _cnt = adopt_counts(df)
ADOPT = _cnt["Uses AI"] / N_SEL * 100
P_USE, P_SOON, P_NOPE = [_cnt[k] / N_SEL * 100 for k in ["Uses AI", "Plans to soon", "Doesn't plan to"]]
PX = paradox_stats(df)
ACC_PCT, N_ACC = trust_dist(df)
P_TRUST = float(ACC_PCT[["Highly trust", "Somewhat trust"]].sum())
P_DISTRUST = float(ACC_PCT[["Somewhat distrust", "Highly distrust"]].sum())
P_HIGH_TRUST = float(ACC_PCT["Highly trust"])
EXP_FULL = exp_table(df)
ROLE_FULL = role_table(df, TOP_ROLES)
DIS_FULL = distrust_table(df)
BEN_CNT_FULL, BEN_PCT_FULL, N_BEN_FULL = multi_stats(df, "AIBen")
CH_CNT_FULL, CH_PCT_FULL, N_CH_FULL = multi_stats(df, "AIChallenges", ch_key)

# ----------------------------------------------------------------------------
# Chart + layout helpers
# ----------------------------------------------------------------------------


def style(fig: go.Figure, height: int = 380, legend: bool = False, P: dict = None, margin=None) -> go.Figure:
    P = P or C
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=P["ink"], size=13), margin=margin or dict(l=10, r=20, t=20, b=10),
        showlegend=legend,
        hoverlabel=dict(bgcolor=P["card_solid"], font=dict(color=P["ink"], family=FONT), bordercolor=P["lime"]),
    )
    fig.update_traces(cliponaxis=False, selector=dict(type="scatter"))  # keep value labels from being cut at the plot edge
    fig.update_traces(cliponaxis=False, selector=dict(type="bar"))
    fig.update_xaxes(gridcolor=P["grid"], zeroline=False, linecolor=P["grid"], tickfont=dict(color=P["muted"]))
    fig.update_yaxes(gridcolor=P["grid"], zeroline=False, linecolor=P["grid"], tickfont=dict(color=P["ink"]))
    return fig


def _new_width_api() -> bool:
    try:
        major, minor = [int(x) for x in str(st.__version__).split(".")[:2]]
        return (major, minor) >= (1, 50)
    except Exception:
        return False


NEW_WIDTH = _new_width_api()


def show(fig: go.Figure) -> None:
    cfg = {"displayModeBar": False}
    try:
        if NEW_WIDTH:
            st.plotly_chart(fig, width="stretch", config=cfg)
        else:
            st.plotly_chart(fig, use_container_width=True, config=cfg)
    except TypeError:  # opposite API generation
        st.plotly_chart(fig, use_container_width=True, config=cfg)


def show_table(frame: pd.DataFrame) -> None:
    try:
        if NEW_WIDTH:
            st.dataframe(frame, width="stretch")
        else:
            st.dataframe(frame, use_container_width=True)
    except TypeError:
        st.dataframe(frame, use_container_width=True)


def section(num: str, tag: str, anchor: str, question: str, lede: str = "") -> None:
    html(f'<div id="{anchor}" class="sec"><div class="num">{num}</div><div><div class="tag">{tag}</div>'
         f'<div class="question">{question}</div></div></div>'
         + (f'<div class="lede">{lede}</div>' if lede else ""))


def insight(text: str, why: str) -> None:
    why_html = f'<div class="why"><b>Why it matters:</b> {why}</div>' if why else ""
    html(f'<div class="insight">{text}</div>{why_html}')


def stat_card(value: str, label: str, lime: bool = False, kind: str = "") -> str:
    cls = "stat lime" if lime else "stat"
    return f'<div class="card {kind}"><div class="{cls}">{value}</div><div class="stat-label">{label}</div></div>'


def low_data(n: int, need: int = MIN_N) -> None:
    html(f'<div class="lowdata">Not enough data to show this reliably: only {n:,} valid responses in the selected audience (minimum {need}). Widen the filters in the sidebar.</div>')


def audience_note(n_valid: int) -> str:
    """Short note stating which population a chart uses."""
    who = "selected audience" if FILTERS_ON else "full survey"
    return f'<div class="note">Base: {who}, {n_valid:,} valid responses.</div>'


def container(key: str):
    try:
        return st.container(key=key)
    except TypeError:  # very old Streamlit: no keyed containers (panel styling is skipped)
        return st.container()


def diverging_trust(rows, height=None):
    """Diverging 5-point trust bars. rows = [(label, distribution Series, n)]."""
    seg_def = [("Highly distrust", "hi_dis", -1), ("Somewhat distrust", "so_dis", -1), ("Neither trust nor distrust", "neutral", 0),
               ("Somewhat trust", "so_trust", 1), ("Highly trust", "hi_trust", 1)]
    ylab = [f"{nm}<br><span style='font-size:11px'>{ND}={n_:,}</span>" for nm, _, n_ in rows]
    fig = go.Figure()
    for label, ckey, side in seg_def:
        bases, widths, texts, hov = [], [], [], []
        for nm, d_, n_ in rows:
            w, neu = float(d_[label]), float(d_["Neither trust nor distrust"])
            if side == 0:
                base = -neu / 2
            elif side == 1:
                base = neu / 2 + (float(d_["Somewhat trust"]) if label == "Highly trust" else 0.0)
            else:
                base = -neu / 2 - w - (float(d_["Somewhat distrust"]) if label == "Highly distrust" else 0.0)
            bases.append(base)
            widths.append(w)
            texts.append(f"{w:.0f}%" if w >= 7 else "")
            hov.append(f"{nm}<br>{label}: {w:.1f}%<br>{nlab(n_)}")
        fig.add_trace(go.Bar(y=ylab, x=widths, base=bases, orientation="h", name=label, marker=dict(color=C[ckey]),
                             text=texts, textposition="inside", insidetextanchor="middle",
                             textfont=dict(color=("#F6F1E4" if (not DARK_MODE and ckey in ("hi_trust", "hi_dis")) else "#0C1B13"), size=12),
                             hovertext=hov, hoverinfo="text"))
    fig.update_layout(barmode="overlay", bargap=0.3, uniformtext=dict(minsize=10, mode="hide"),
                      legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.06, yanchor="top", font=dict(color=C["ink"], size=11), tracegroupgap=2))
    fig.add_vline(x=0, line_color=C["ink"], line_width=1, opacity=.5)
    fig.update_xaxes(range=[-60, 60], showticklabels=False, gridcolor="rgba(0,0,0,0)")
    fig.update_yaxes(autorange="reversed", automargin=True)
    return style(fig, height or 150 + 62 * len(rows), legend=True, margin=dict(l=10, r=10, t=10, b=70))


# ----------------------------------------------------------------------------
# Header: nav + theme
# ----------------------------------------------------------------------------
nav_col, tog_col = st.columns([6, 1.3])
with nav_col:
    html(f"""
    <div class="nav">
    <a href="#ov">Overview</a><a href="#explore">Explore</a><a href="#prep">Data prep</a>
    <a href="#d1">Adoption</a><a href="#d2">Experience</a><a href="#d3">Roles</a>
    <a href="#ben">Benefits</a><a href="#chal">Challenges</a>
    <a href="#d4">Productivity & trust</a><a href="#d5">Accuracy</a>
    <a href="#cyber">Security gap</a><a href="#threat">Job threat</a>
    <a href="#sal">Salary</a><a href="#you">You</a>
    <a href="#lab">Trust Test</a><a href="#story">Summary</a>
    </div>
    """)
with tog_col:
    st.toggle("Dark mode", key="dark")

# ----------------------------------------------------------------------------
# Hero
# ----------------------------------------------------------------------------
NEURAL = """<svg viewBox="0 0 400 340" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Abstract field of flowing data contours with one highlighted trend line">
<defs><filter id="bl" x="-10%" y="-40%" width="120%" height="180%"><feGaussianBlur stdDeviation="5"/></filter>
<linearGradient id="fr" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#F6F1E4" stop-opacity=".10"/><stop offset="1" stop-color="#F6F1E4" stop-opacity=".02"/></linearGradient>
<linearGradient id="fd" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#fff" stop-opacity="0"/><stop offset=".12" stop-color="#fff"/><stop offset=".88" stop-color="#fff"/><stop offset="1" stop-color="#fff" stop-opacity="0"/></linearGradient>
<mask id="mk"><rect width="400" height="340" fill="url(#fd)"/></mask></defs>
<rect x="1" y="1" width="398" height="338" rx="26" fill="url(#fr)" stroke="#9CC4A8" stroke-opacity=".28"/>
<g mask="url(#mk)"><path d="M22.0,60.0 Q27.9,61.8 30.9,62.8 Q33.9,63.8 36.8,64.8 Q39.8,65.8 42.8,66.9 Q45.7,68.0 48.7,69.1 Q51.7,70.2 54.6,71.4 Q57.6,72.6 60.6,73.7 Q63.5,74.9 66.5,76.1 Q69.5,77.3 72.4,78.6 Q75.4,79.8 78.4,81.0 Q81.3,82.2 84.3,83.4 Q87.3,84.6 90.2,85.8 Q93.2,86.9 96.2,88.1 Q99.1,89.2 102.1,90.3 Q105.1,91.4 108.0,92.4 Q111.0,93.4 114.0,94.4 Q116.9,95.3 119.9,96.2 Q122.9,97.1 125.8,97.8 Q128.8,98.6 131.8,99.2 Q134.7,99.8 137.7,100.4 Q140.7,100.9 143.6,101.3 Q146.6,101.6 149.6,101.9 Q152.5,102.1 155.5,102.2 Q158.5,102.3 161.4,102.3 Q164.4,102.2 167.4,102.0 Q170.3,101.8 173.3,101.5 Q176.3,101.1 179.2,100.6 Q182.2,100.1 185.2,99.5 Q188.1,98.8 191.1,98.1 Q194.1,97.3 197.0,96.4 Q200.0,95.5 203.0,94.4 Q205.9,93.4 208.9,92.3 Q211.9,91.1 214.8,89.9 Q217.8,88.7 220.8,87.4 Q223.7,86.1 226.7,84.7 Q229.7,83.4 232.6,82.0 Q235.6,80.6 238.6,79.1 Q241.5,77.7 244.5,76.3 Q247.5,74.8 250.4,73.4 Q253.4,72.0 256.4,70.6 Q259.3,69.2 262.3,67.9 Q265.3,66.5 268.2,65.2 Q271.2,63.9 274.2,62.7 Q277.1,61.5 280.1,60.4 Q283.1,59.2 286.0,58.2 Q289.0,57.2 292.0,56.3 Q294.9,55.4 297.9,54.6 Q300.9,53.8 303.8,53.1 Q306.8,52.4 309.8,51.9 Q312.7,51.3 315.7,50.9 Q318.7,50.5 321.6,50.1 Q324.6,49.8 327.6,49.7 Q330.5,49.5 333.5,49.4 Q336.5,49.4 339.4,49.4 Q342.4,49.5 345.4,49.6 Q348.3,49.8 351.3,50.0 Q354.3,50.3 357.2,50.6 Q360.2,50.9 363.2,51.3 Q366.1,51.7 369.1,52.2 Q372.1,52.6 375.0,53.1" fill="none" stroke="#9CC4A8" stroke-opacity="0.10" stroke-width="1"/><path d="M22.0,68.2 Q27.9,70.2 30.9,71.2 Q33.9,72.2 36.8,73.3 Q39.8,74.3 42.8,75.4 Q45.7,76.5 48.7,77.6 Q51.7,78.8 54.6,79.9 Q57.6,81.1 60.6,82.2 Q63.5,83.4 66.5,84.5 Q69.5,85.7 72.4,86.8 Q75.4,88.0 78.4,89.1 Q81.3,90.2 84.3,91.3 Q87.3,92.4 90.2,93.5 Q93.2,94.5 96.2,95.5 Q99.1,96.5 102.1,97.5 Q105.1,98.4 108.0,99.2 Q111.0,100.1 114.0,100.8 Q116.9,101.6 119.9,102.2 Q122.9,102.9 125.8,103.4 Q128.8,103.9 131.8,104.4 Q134.7,104.8 137.7,105.1 Q140.7,105.3 143.6,105.5 Q146.6,105.6 149.6,105.6 Q152.5,105.7 155.5,105.5 Q158.5,105.4 161.4,105.1 Q164.4,104.9 167.4,104.4 Q170.3,104.0 173.3,103.5 Q176.3,102.9 179.2,102.3 Q182.2,101.6 185.2,100.8 Q188.1,100.0 191.1,99.1 Q194.1,98.1 197.0,97.1 Q200.0,96.1 203.0,95.0 Q205.9,93.9 208.9,92.7 Q211.9,91.5 214.8,90.3 Q217.8,89.1 220.8,87.8 Q223.7,86.5 226.7,85.2 Q229.7,83.9 232.6,82.6 Q235.6,81.3 238.6,79.9 Q241.5,78.6 244.5,77.4 Q247.5,76.1 250.4,74.8 Q253.4,73.6 256.4,72.4 Q259.3,71.2 262.3,70.1 Q265.3,69.0 268.2,68.0 Q271.2,66.9 274.2,66.0 Q277.1,65.1 280.1,64.2 Q283.1,63.4 286.0,62.7 Q289.0,62.0 292.0,61.4 Q294.9,60.8 297.9,60.3 Q300.9,59.8 303.8,59.5 Q306.8,59.1 309.8,58.9 Q312.7,58.7 315.7,58.6 Q318.7,58.5 321.6,58.5 Q324.6,58.5 327.6,58.6 Q330.5,58.8 333.5,59.0 Q336.5,59.2 339.4,59.5 Q342.4,59.9 345.4,60.3 Q348.3,60.7 351.3,61.2 Q354.3,61.6 357.2,62.2 Q360.2,62.7 363.2,63.3 Q366.1,63.9 369.1,64.5 Q372.1,65.1 375.0,65.8" fill="none" stroke="#9CC4A8" stroke-opacity="0.11" stroke-width="1"/><path d="M22.0,76.4 Q27.9,78.4 30.9,79.4 Q33.9,80.4 36.8,81.5 Q39.8,82.5 42.8,83.6 Q45.7,84.7 48.7,85.7 Q51.7,86.8 54.6,87.9 Q57.6,89.0 60.6,90.1 Q63.5,91.1 66.5,92.2 Q69.5,93.3 72.4,94.3 Q75.4,95.3 78.4,96.3 Q81.3,97.3 84.3,98.3 Q87.3,99.2 90.2,100.1 Q93.2,101.0 96.2,101.8 Q99.1,102.6 102.1,103.4 Q105.1,104.1 108.0,104.7 Q111.0,105.4 114.0,105.9 Q116.9,106.5 119.9,106.9 Q122.9,107.3 125.8,107.6 Q128.8,108.0 131.8,108.1 Q134.7,108.3 137.7,108.4 Q140.7,108.5 143.6,108.4 Q146.6,108.3 149.6,108.1 Q152.5,107.9 155.5,107.6 Q158.5,107.3 161.4,106.8 Q164.4,106.4 167.4,105.8 Q170.3,105.2 173.3,104.5 Q176.3,103.8 179.2,103.0 Q182.2,102.2 185.2,101.3 Q188.1,100.4 191.1,99.5 Q194.1,98.5 197.0,97.4 Q200.0,96.4 203.0,95.2 Q205.9,94.1 208.9,93.0 Q211.9,91.8 214.8,90.6 Q217.8,89.5 220.8,88.3 Q223.7,87.1 226.7,85.9 Q229.7,84.7 232.6,83.5 Q235.6,82.3 238.6,81.2 Q241.5,80.1 244.5,79.0 Q247.5,77.9 250.4,76.9 Q253.4,75.9 256.4,75.0 Q259.3,74.0 262.3,73.2 Q265.3,72.4 268.2,71.6 Q271.2,70.9 274.2,70.3 Q277.1,69.6 280.1,69.1 Q283.1,68.6 286.0,68.2 Q289.0,67.8 292.0,67.5 Q294.9,67.3 297.9,67.1 Q300.9,67.0 303.8,66.9 Q306.8,66.9 309.8,67.0 Q312.7,67.1 315.7,67.3 Q318.7,67.5 321.6,67.8 Q324.6,68.1 327.6,68.5 Q330.5,68.9 333.5,69.4 Q336.5,69.9 339.4,70.4 Q342.4,71.0 345.4,71.6 Q348.3,72.2 351.3,72.9 Q354.3,73.6 357.2,74.3 Q360.2,75.0 363.2,75.7 Q366.1,76.5 369.1,77.2 Q372.1,78.0 375.0,78.7" fill="none" stroke="#9CC4A8" stroke-opacity="0.14" stroke-width="1"/><path d="M22.0,84.4 Q27.9,86.3 30.9,87.3 Q33.9,88.3 36.8,89.3 Q39.8,90.3 42.8,91.2 Q45.7,92.2 48.7,93.2 Q51.7,94.2 54.6,95.2 Q57.6,96.2 60.6,97.1 Q63.5,98.1 66.5,99.0 Q69.5,99.9 72.4,100.8 Q75.4,101.7 78.4,102.5 Q81.3,103.4 84.3,104.2 Q87.3,104.9 90.2,105.6 Q93.2,106.3 96.2,106.9 Q99.1,107.6 102.1,108.1 Q105.1,108.6 108.0,109.1 Q111.0,109.5 114.0,109.8 Q116.9,110.2 119.9,110.4 Q122.9,110.6 125.8,110.7 Q128.8,110.8 131.8,110.8 Q134.7,110.8 137.7,110.7 Q140.7,110.5 143.6,110.3 Q146.6,110.0 149.6,109.7 Q152.5,109.3 155.5,108.8 Q158.5,108.3 161.4,107.8 Q164.4,107.2 167.4,106.5 Q170.3,105.8 173.3,105.0 Q176.3,104.2 179.2,103.4 Q182.2,102.5 185.2,101.6 Q188.1,100.7 191.1,99.7 Q194.1,98.7 197.0,97.6 Q200.0,96.6 203.0,95.5 Q205.9,94.5 208.9,93.4 Q211.9,92.3 214.8,91.3 Q217.8,90.2 220.8,89.1 Q223.7,88.1 226.7,87.1 Q229.7,86.0 232.6,85.1 Q235.6,84.1 238.6,83.2 Q241.5,82.3 244.5,81.4 Q247.5,80.6 250.4,79.9 Q253.4,79.1 256.4,78.5 Q259.3,77.8 262.3,77.3 Q265.3,76.7 268.2,76.3 Q271.2,75.8 274.2,75.5 Q277.1,75.2 280.1,75.0 Q283.1,74.8 286.0,74.7 Q289.0,74.6 292.0,74.7 Q294.9,74.7 297.9,74.9 Q300.9,75.0 303.8,75.3 Q306.8,75.6 309.8,76.0 Q312.7,76.3 315.7,76.8 Q318.7,77.3 321.6,77.9 Q324.6,78.4 327.6,79.1 Q330.5,79.7 333.5,80.4 Q336.5,81.1 339.4,81.9 Q342.4,82.6 345.4,83.4 Q348.3,84.2 351.3,85.1 Q354.3,85.9 357.2,86.8 Q360.2,87.6 363.2,88.5 Q366.1,89.3 369.1,90.1 Q372.1,91.0 375.0,91.8" fill="none" stroke="#9CC4A8" stroke-opacity="0.17" stroke-width="1"/><path d="M22.0,92.1 Q27.9,93.8 30.9,94.7 Q33.9,95.6 36.8,96.5 Q39.8,97.4 42.8,98.3 Q45.7,99.1 48.7,100.0 Q51.7,100.9 54.6,101.7 Q57.6,102.5 60.6,103.4 Q63.5,104.2 66.5,104.9 Q69.5,105.7 72.4,106.4 Q75.4,107.1 78.4,107.8 Q81.3,108.4 84.3,109.0 Q87.3,109.6 90.2,110.1 Q93.2,110.6 96.2,111.1 Q99.1,111.5 102.1,111.8 Q105.1,112.2 108.0,112.4 Q111.0,112.6 114.0,112.8 Q116.9,112.9 119.9,113.0 Q122.9,113.0 125.8,112.9 Q128.8,112.9 131.8,112.7 Q134.7,112.5 137.7,112.2 Q140.7,111.9 143.6,111.5 Q146.6,111.1 149.6,110.7 Q152.5,110.2 155.5,109.6 Q158.5,109.0 161.4,108.3 Q164.4,107.7 167.4,106.9 Q170.3,106.2 173.3,105.4 Q176.3,104.5 179.2,103.7 Q182.2,102.8 185.2,101.9 Q188.1,101.0 191.1,100.0 Q194.1,99.1 197.0,98.1 Q200.0,97.2 203.0,96.2 Q205.9,95.3 208.9,94.3 Q211.9,93.4 214.8,92.5 Q217.8,91.5 220.8,90.7 Q223.7,89.8 226.7,89.0 Q229.7,88.2 232.6,87.4 Q235.6,86.7 238.6,86.0 Q241.5,85.3 244.5,84.8 Q247.5,84.2 250.4,83.7 Q253.4,83.2 256.4,82.9 Q259.3,82.5 262.3,82.3 Q265.3,82.0 268.2,81.9 Q271.2,81.8 274.2,81.7 Q277.1,81.7 280.1,81.8 Q283.1,81.9 286.0,82.2 Q289.0,82.4 292.0,82.7 Q294.9,83.1 297.9,83.5 Q300.9,84.0 303.8,84.5 Q306.8,85.0 309.8,85.7 Q312.7,86.3 315.7,87.0 Q318.7,87.8 321.6,88.6 Q324.6,89.3 327.6,90.2 Q330.5,91.1 333.5,92.0 Q336.5,92.8 339.4,93.8 Q342.4,94.7 345.4,95.7 Q348.3,96.6 351.3,97.6 Q354.3,98.5 357.2,99.5 Q360.2,100.4 363.2,101.4 Q366.1,102.3 369.1,103.2 Q372.1,104.1 375.0,105.0" fill="none" stroke="#9CC4A8" stroke-opacity="0.21" stroke-width="1"/><path d="M22.0,99.2 Q27.9,100.8 30.9,101.5 Q33.9,102.3 36.8,103.1 Q39.8,103.8 42.8,104.6 Q45.7,105.3 48.7,106.0 Q51.7,106.8 54.6,107.4 Q57.6,108.1 60.6,108.8 Q63.5,109.4 66.5,110.0 Q69.5,110.6 72.4,111.2 Q75.4,111.7 78.4,112.2 Q81.3,112.7 84.3,113.1 Q87.3,113.5 90.2,113.8 Q93.2,114.2 96.2,114.4 Q99.1,114.7 102.1,114.8 Q105.1,115.0 108.0,115.1 Q111.0,115.1 114.0,115.1 Q116.9,115.1 119.9,115.0 Q122.9,114.8 125.8,114.6 Q128.8,114.4 131.8,114.1 Q134.7,113.8 137.7,113.4 Q140.7,113.0 143.6,112.5 Q146.6,112.0 149.6,111.5 Q152.5,110.9 155.5,110.3 Q158.5,109.6 161.4,108.9 Q164.4,108.2 167.4,107.4 Q170.3,106.7 173.3,105.9 Q176.3,105.1 179.2,104.2 Q182.2,103.4 185.2,102.6 Q188.1,101.7 191.1,100.9 Q194.1,100.0 197.0,99.2 Q200.0,98.3 203.0,97.5 Q205.9,96.7 208.9,95.9 Q211.9,95.1 214.8,94.4 Q217.8,93.6 220.8,93.0 Q223.7,92.3 226.7,91.7 Q229.7,91.1 232.6,90.6 Q235.6,90.1 238.6,89.7 Q241.5,89.3 244.5,89.0 Q247.5,88.7 250.4,88.5 Q253.4,88.3 256.4,88.2 Q259.3,88.1 262.3,88.2 Q265.3,88.2 268.2,88.4 Q271.2,88.6 274.2,88.8 Q277.1,89.1 280.1,89.5 Q283.1,89.9 286.0,90.4 Q289.0,90.9 292.0,91.5 Q294.9,92.2 297.9,92.9 Q300.9,93.6 303.8,94.4 Q306.8,95.2 309.8,96.0 Q312.7,96.9 315.7,97.9 Q318.7,98.8 321.6,99.8 Q324.6,100.8 327.6,101.8 Q330.5,102.8 333.5,103.9 Q336.5,105.0 339.4,106.0 Q342.4,107.1 345.4,108.2 Q348.3,109.2 351.3,110.3 Q354.3,111.4 357.2,112.4 Q360.2,113.4 363.2,114.4 Q366.1,115.4 369.1,116.3 Q372.1,117.3 375.0,118.1" fill="none" stroke="#9CC4A8" stroke-opacity="0.25" stroke-width="1"/><path d="M22.0,105.9 Q27.9,107.2 30.9,107.8 Q33.9,108.4 36.8,109.0 Q39.8,109.7 42.8,110.3 Q45.7,110.9 48.7,111.4 Q51.7,112.0 54.6,112.5 Q57.6,113.1 60.6,113.6 Q63.5,114.0 66.5,114.5 Q69.5,114.9 72.4,115.3 Q75.4,115.7 78.4,116.0 Q81.3,116.3 84.3,116.6 Q87.3,116.8 90.2,117.0 Q93.2,117.2 96.2,117.3 Q99.1,117.4 102.1,117.4 Q105.1,117.4 108.0,117.3 Q111.0,117.3 114.0,117.1 Q116.9,117.0 119.9,116.7 Q122.9,116.5 125.8,116.2 Q128.8,115.8 131.8,115.4 Q134.7,115.0 137.7,114.6 Q140.7,114.1 143.6,113.5 Q146.6,113.0 149.6,112.4 Q152.5,111.8 155.5,111.1 Q158.5,110.4 161.4,109.7 Q164.4,109.0 167.4,108.3 Q170.3,107.6 173.3,106.8 Q176.3,106.0 179.2,105.3 Q182.2,104.5 185.2,103.7 Q188.1,103.0 191.1,102.2 Q194.1,101.5 197.0,100.8 Q200.0,100.1 203.0,99.4 Q205.9,98.8 208.9,98.2 Q211.9,97.6 214.8,97.0 Q217.8,96.5 220.8,96.1 Q223.7,95.6 226.7,95.3 Q229.7,94.9 232.6,94.6 Q235.6,94.4 238.6,94.2 Q241.5,94.1 244.5,94.1 Q247.5,94.0 250.4,94.1 Q253.4,94.2 256.4,94.4 Q259.3,94.6 262.3,95.0 Q265.3,95.3 268.2,95.7 Q271.2,96.2 274.2,96.8 Q277.1,97.3 280.1,98.0 Q283.1,98.7 286.0,99.4 Q289.0,100.2 292.0,101.1 Q294.9,102.0 297.9,102.9 Q300.9,103.9 303.8,104.9 Q306.8,105.9 309.8,107.0 Q312.7,108.1 315.7,109.2 Q318.7,110.3 321.6,111.5 Q324.6,112.7 327.6,113.8 Q330.5,115.0 333.5,116.2 Q336.5,117.4 339.4,118.6 Q342.4,119.8 345.4,120.9 Q348.3,122.1 351.3,123.2 Q354.3,124.3 357.2,125.4 Q360.2,126.4 363.2,127.5 Q366.1,128.5 369.1,129.4 Q372.1,130.4 375.0,131.2" fill="none" stroke="#9CC4A8" stroke-opacity="0.29" stroke-width="1"/><path d="M22.0,112.0 Q27.9,113.1 30.9,113.6 Q33.9,114.1 36.8,114.6 Q39.8,115.0 42.8,115.5 Q45.7,116.0 48.7,116.4 Q51.7,116.8 54.6,117.2 Q57.6,117.6 60.6,118.0 Q63.5,118.3 66.5,118.6 Q69.5,118.9 72.4,119.1 Q75.4,119.4 78.4,119.6 Q81.3,119.7 84.3,119.9 Q87.3,120.0 90.2,120.0 Q93.2,120.0 96.2,120.0 Q99.1,120.0 102.1,119.9 Q105.1,119.8 108.0,119.6 Q111.0,119.4 114.0,119.1 Q116.9,118.9 119.9,118.5 Q122.9,118.2 125.8,117.8 Q128.8,117.4 131.8,116.9 Q134.7,116.5 137.7,115.9 Q140.7,115.4 143.6,114.8 Q146.6,114.2 149.6,113.6 Q152.5,113.0 155.5,112.3 Q158.5,111.7 161.4,111.0 Q164.4,110.3 167.4,109.6 Q170.3,108.9 173.3,108.2 Q176.3,107.5 179.2,106.9 Q182.2,106.2 185.2,105.5 Q188.1,104.9 191.1,104.3 Q194.1,103.7 197.0,103.1 Q200.0,102.6 203.0,102.1 Q205.9,101.6 208.9,101.2 Q211.9,100.8 214.8,100.5 Q217.8,100.2 220.8,100.0 Q223.7,99.7 226.7,99.6 Q229.7,99.5 232.6,99.5 Q235.6,99.5 238.6,99.7 Q241.5,99.8 244.5,100.0 Q247.5,100.3 250.4,100.6 Q253.4,101.0 256.4,101.5 Q259.3,102.0 262.3,102.6 Q265.3,103.2 268.2,103.9 Q271.2,104.7 274.2,105.5 Q277.1,106.3 280.1,107.3 Q283.1,108.2 286.0,109.2 Q289.0,110.2 292.0,111.3 Q294.9,112.4 297.9,113.6 Q300.9,114.8 303.8,116.0 Q306.8,117.2 309.8,118.5 Q312.7,119.8 315.7,121.0 Q318.7,122.3 321.6,123.6 Q324.6,124.9 327.6,126.2 Q330.5,127.5 333.5,128.8 Q336.5,130.1 339.4,131.3 Q342.4,132.6 345.4,133.8 Q348.3,135.0 351.3,136.1 Q354.3,137.3 357.2,138.3 Q360.2,139.4 363.2,140.4 Q366.1,141.4 369.1,142.3 Q372.1,143.2 375.0,144.0" fill="none" stroke="#9CC4A8" stroke-opacity="0.33" stroke-width="1"/><path d="M22.0,117.9 Q27.9,118.7 30.9,119.1 Q33.9,119.5 36.8,119.9 Q39.8,120.2 42.8,120.6 Q45.7,120.9 48.7,121.2 Q51.7,121.5 54.6,121.8 Q57.6,122.0 60.6,122.3 Q63.5,122.5 66.5,122.7 Q69.5,122.8 72.4,122.9 Q75.4,123.1 78.4,123.1 Q81.3,123.2 84.3,123.2 Q87.3,123.2 90.2,123.1 Q93.2,123.0 96.2,122.9 Q99.1,122.7 102.1,122.5 Q105.1,122.3 108.0,122.0 Q111.0,121.7 114.0,121.4 Q116.9,121.0 119.9,120.6 Q122.9,120.2 125.8,119.7 Q128.8,119.3 131.8,118.7 Q134.7,118.2 137.7,117.6 Q140.7,117.1 143.6,116.5 Q146.6,115.9 149.6,115.3 Q152.5,114.6 155.5,114.0 Q158.5,113.4 161.4,112.7 Q164.4,112.1 167.4,111.5 Q170.3,110.8 173.3,110.2 Q176.3,109.6 179.2,109.0 Q182.2,108.5 185.2,108.0 Q188.1,107.5 191.1,107.0 Q194.1,106.5 197.0,106.2 Q200.0,105.8 203.0,105.5 Q205.9,105.2 208.9,105.0 Q211.9,104.8 214.8,104.7 Q217.8,104.7 220.8,104.7 Q223.7,104.7 226.7,104.9 Q229.7,105.0 232.6,105.3 Q235.6,105.6 238.6,105.9 Q241.5,106.3 244.5,106.9 Q247.5,107.4 250.4,108.0 Q253.4,108.7 256.4,109.4 Q259.3,110.2 262.3,111.1 Q265.3,112.0 268.2,113.0 Q271.2,114.0 274.2,115.0 Q277.1,116.1 280.1,117.3 Q283.1,118.5 286.0,119.7 Q289.0,121.0 292.0,122.3 Q294.9,123.6 297.9,125.0 Q300.9,126.3 303.8,127.7 Q306.8,129.1 309.8,130.5 Q312.7,131.9 315.7,133.3 Q318.7,134.7 321.6,136.1 Q324.6,137.5 327.6,138.9 Q330.5,140.2 333.5,141.6 Q336.5,142.9 339.4,144.2 Q342.4,145.4 345.4,146.6 Q348.3,147.8 351.3,148.9 Q354.3,150.0 357.2,151.1 Q360.2,152.1 363.2,153.0 Q366.1,153.9 369.1,154.8 Q372.1,155.6 375.0,156.3" fill="none" stroke="#9CC4A8" stroke-opacity="0.37" stroke-width="1"/><path d="M22.0,123.8 Q27.9,124.4 30.9,124.6 Q33.9,124.9 36.8,125.2 Q39.8,125.4 42.8,125.7 Q45.7,125.9 48.7,126.1 Q51.7,126.3 54.6,126.5 Q57.6,126.6 60.6,126.7 Q63.5,126.8 66.5,126.9 Q69.5,126.9 72.4,126.9 Q75.4,126.9 78.4,126.9 Q81.3,126.8 84.3,126.7 Q87.3,126.6 90.2,126.4 Q93.2,126.2 96.2,126.0 Q99.1,125.7 102.1,125.4 Q105.1,125.1 108.0,124.7 Q111.0,124.3 114.0,123.9 Q116.9,123.5 119.9,123.0 Q122.9,122.5 125.8,122.0 Q128.8,121.5 131.8,120.9 Q134.7,120.3 137.7,119.7 Q140.7,119.2 143.6,118.6 Q146.6,118.0 149.6,117.3 Q152.5,116.7 155.5,116.1 Q158.5,115.5 161.4,115.0 Q164.4,114.4 167.4,113.8 Q170.3,113.3 173.3,112.8 Q176.3,112.3 179.2,111.9 Q182.2,111.4 185.2,111.1 Q188.1,110.7 191.1,110.4 Q194.1,110.2 197.0,110.0 Q200.0,109.8 203.0,109.7 Q205.9,109.6 208.9,109.7 Q211.9,109.7 214.8,109.9 Q217.8,110.0 220.8,110.3 Q223.7,110.6 226.7,111.0 Q229.7,111.4 232.6,112.0 Q235.6,112.5 238.6,113.2 Q241.5,113.9 244.5,114.7 Q247.5,115.5 250.4,116.4 Q253.4,117.3 256.4,118.4 Q259.3,119.4 262.3,120.5 Q265.3,121.7 268.2,122.9 Q271.2,124.1 274.2,125.5 Q277.1,126.8 280.1,128.2 Q283.1,129.5 286.0,131.0 Q289.0,132.4 292.0,133.9 Q294.9,135.4 297.9,136.9 Q300.9,138.4 303.8,139.9 Q306.8,141.4 309.8,142.9 Q312.7,144.4 315.7,145.9 Q318.7,147.3 321.6,148.8 Q324.6,150.2 327.6,151.6 Q330.5,153.0 333.5,154.3 Q336.5,155.6 339.4,156.8 Q342.4,158.0 345.4,159.2 Q348.3,160.3 351.3,161.3 Q354.3,162.3 357.2,163.3 Q360.2,164.2 363.2,165.0 Q366.1,165.8 369.1,166.5 Q372.1,167.1 375.0,167.7" fill="none" stroke="#9CC4A8" stroke-opacity="0.41" stroke-width="1"/><path d="M22.0,129.8 Q27.9,130.3 30.9,130.4 Q33.9,130.6 36.8,130.8 Q39.8,131.0 42.8,131.1 Q45.7,131.2 48.7,131.3 Q51.7,131.4 54.6,131.4 Q57.6,131.5 60.6,131.5 Q63.5,131.5 66.5,131.4 Q69.5,131.4 72.4,131.2 Q75.4,131.1 78.4,130.9 Q81.3,130.8 84.3,130.5 Q87.3,130.3 90.2,130.0 Q93.2,129.7 96.2,129.3 Q99.1,129.0 102.1,128.6 Q105.1,128.2 108.0,127.7 Q111.0,127.3 114.0,126.7 Q116.9,126.2 119.9,125.7 Q122.9,125.2 125.8,124.6 Q128.8,124.0 131.8,123.4 Q134.7,122.8 137.7,122.2 Q140.7,121.7 143.6,121.1 Q146.6,120.5 149.6,119.9 Q152.5,119.3 155.5,118.8 Q158.5,118.3 161.4,117.8 Q164.4,117.3 167.4,116.8 Q170.3,116.4 173.3,116.0 Q176.3,115.7 179.2,115.4 Q182.2,115.1 185.2,114.9 Q188.1,114.8 191.1,114.7 Q194.1,114.6 197.0,114.6 Q200.0,114.7 203.0,114.8 Q205.9,115.0 208.9,115.3 Q211.9,115.6 214.8,116.0 Q217.8,116.4 220.8,117.0 Q223.7,117.5 226.7,118.2 Q229.7,118.9 232.6,119.7 Q235.6,120.6 238.6,121.5 Q241.5,122.5 244.5,123.5 Q247.5,124.6 250.4,125.8 Q253.4,127.0 256.4,128.3 Q259.3,129.5 262.3,130.9 Q265.3,132.3 268.2,133.7 Q271.2,135.2 274.2,136.7 Q277.1,138.2 280.1,139.8 Q283.1,141.3 286.0,142.9 Q289.0,144.5 292.0,146.1 Q294.9,147.7 297.9,149.2 Q300.9,150.8 303.8,152.4 Q306.8,154.0 309.8,155.5 Q312.7,157.0 315.7,158.5 Q318.7,160.0 321.6,161.4 Q324.6,162.8 327.6,164.1 Q330.5,165.4 333.5,166.6 Q336.5,167.8 339.4,169.0 Q342.4,170.1 345.4,171.1 Q348.3,172.1 351.3,173.0 Q354.3,173.8 357.2,174.6 Q360.2,175.3 363.2,176.0 Q366.1,176.6 369.1,177.1 Q372.1,177.6 375.0,178.0" fill="none" stroke="#9CC4A8" stroke-opacity="0.44" stroke-width="1"/><path d="M22.0,136.3 Q27.9,136.6 30.9,136.6 Q33.9,136.7 36.8,136.8 Q39.8,136.9 42.8,136.9 Q45.7,136.9 48.7,136.9 Q51.7,136.9 54.6,136.8 Q57.6,136.7 60.6,136.6 Q63.5,136.5 66.5,136.3 Q69.5,136.1 72.4,135.9 Q75.4,135.6 78.4,135.3 Q81.3,135.0 84.3,134.7 Q87.3,134.3 90.2,133.9 Q93.2,133.5 96.2,133.0 Q99.1,132.5 102.1,132.0 Q105.1,131.5 108.0,131.0 Q111.0,130.4 114.0,129.9 Q116.9,129.3 119.9,128.7 Q122.9,128.1 125.8,127.5 Q128.8,126.9 131.8,126.3 Q134.7,125.7 137.7,125.2 Q140.7,124.6 143.6,124.0 Q146.6,123.5 149.6,123.0 Q152.5,122.5 155.5,122.0 Q158.5,121.6 161.4,121.2 Q164.4,120.8 167.4,120.5 Q170.3,120.3 173.3,120.1 Q176.3,119.9 179.2,119.8 Q182.2,119.7 185.2,119.7 Q188.1,119.7 191.1,119.9 Q194.1,120.1 197.0,120.3 Q200.0,120.6 203.0,121.1 Q205.9,121.5 208.9,122.0 Q211.9,122.6 214.8,123.3 Q217.8,124.0 220.8,124.8 Q223.7,125.7 226.7,126.6 Q229.7,127.6 232.6,128.7 Q235.6,129.8 238.6,131.0 Q241.5,132.2 244.5,133.5 Q247.5,134.8 250.4,136.3 Q253.4,137.7 256.4,139.2 Q259.3,140.7 262.3,142.2 Q265.3,143.8 268.2,145.4 Q271.2,147.0 274.2,148.6 Q277.1,150.3 280.1,151.9 Q283.1,153.6 286.0,155.2 Q289.0,156.9 292.0,158.5 Q294.9,160.2 297.9,161.8 Q300.9,163.4 303.8,164.9 Q306.8,166.5 309.8,167.9 Q312.7,169.4 315.7,170.8 Q318.7,172.2 321.6,173.5 Q324.6,174.8 327.6,175.9 Q330.5,177.1 333.5,178.2 Q336.5,179.3 339.4,180.2 Q342.4,181.2 345.4,182.0 Q348.3,182.8 351.3,183.5 Q354.3,184.2 357.2,184.7 Q360.2,185.2 363.2,185.7 Q366.1,186.1 369.1,186.4 Q372.1,186.6 375.0,186.8" fill="none" stroke="#9CC4A8" stroke-opacity="0.47" stroke-width="1"/><path d="M22.0,143.3 Q27.9,143.3 30.9,143.3 Q33.9,143.3 36.8,143.2 Q39.8,143.2 42.8,143.1 Q45.7,143.0 48.7,142.8 Q51.7,142.7 54.6,142.5 Q57.6,142.2 60.6,142.0 Q63.5,141.7 66.5,141.4 Q69.5,141.1 72.4,140.7 Q75.4,140.3 78.4,139.9 Q81.3,139.5 84.3,139.0 Q87.3,138.5 90.2,138.0 Q93.2,137.4 96.2,136.9 Q99.1,136.3 102.1,135.7 Q105.1,135.1 108.0,134.5 Q111.0,133.9 114.0,133.3 Q116.9,132.7 119.9,132.0 Q122.9,131.4 125.8,130.8 Q128.8,130.2 131.8,129.7 Q134.7,129.1 137.7,128.6 Q140.7,128.0 143.6,127.6 Q146.6,127.1 149.6,126.7 Q152.5,126.3 155.5,126.0 Q158.5,125.7 161.4,125.5 Q164.4,125.3 167.4,125.2 Q170.3,125.0 173.3,125.1 Q176.3,125.1 179.2,125.2 Q182.2,125.4 185.2,125.6 Q188.1,125.9 191.1,126.3 Q194.1,126.7 197.0,127.3 Q200.0,127.8 203.0,128.5 Q205.9,129.2 208.9,130.1 Q211.9,130.9 214.8,131.9 Q217.8,132.8 220.8,133.9 Q223.7,135.1 226.7,136.3 Q229.7,137.5 232.6,138.8 Q235.6,140.2 238.6,141.6 Q241.5,143.0 244.5,144.6 Q247.5,146.1 250.4,147.7 Q253.4,149.3 256.4,150.9 Q259.3,152.6 262.3,154.2 Q265.3,155.9 268.2,157.6 Q271.2,159.3 274.2,161.0 Q277.1,162.7 280.1,164.4 Q283.1,166.1 286.0,167.7 Q289.0,169.4 292.0,171.0 Q294.9,172.5 297.9,174.1 Q300.9,175.6 303.8,177.0 Q306.8,178.5 309.8,179.8 Q312.7,181.1 315.7,182.4 Q318.7,183.6 321.6,184.7 Q324.6,185.8 327.6,186.8 Q330.5,187.8 333.5,188.7 Q336.5,189.5 339.4,190.2 Q342.4,190.9 345.4,191.5 Q348.3,192.1 351.3,192.5 Q354.3,193.0 357.2,193.3 Q360.2,193.6 363.2,193.8 Q366.1,194.0 369.1,194.0 Q372.1,194.1 375.0,194.0" fill="none" stroke="#9CC4A8" stroke-opacity="0.49" stroke-width="1"/><path d="M22.0,150.6 Q27.9,150.4 30.9,150.3 Q33.9,150.2 36.8,150.0 Q39.8,149.8 42.8,149.5 Q45.7,149.3 48.7,149.0 Q51.7,148.7 54.6,148.3 Q57.6,147.9 60.6,147.5 Q63.5,147.1 66.5,146.6 Q69.5,146.2 72.4,145.7 Q75.4,145.1 78.4,144.6 Q81.3,144.0 84.3,143.4 Q87.3,142.8 90.2,142.2 Q93.2,141.6 96.2,140.9 Q99.1,140.3 102.1,139.6 Q105.1,139.0 108.0,138.3 Q111.0,137.7 114.0,137.0 Q116.9,136.4 119.9,135.8 Q122.9,135.2 125.8,134.6 Q128.8,134.1 131.8,133.6 Q134.7,133.1 137.7,132.7 Q140.7,132.2 143.6,131.9 Q146.6,131.6 149.6,131.3 Q152.5,131.1 155.5,131.0 Q158.5,130.8 161.4,130.8 Q164.4,130.8 167.4,130.9 Q170.3,131.0 173.3,131.3 Q176.3,131.5 179.2,131.9 Q182.2,132.3 185.2,132.9 Q188.1,133.4 191.1,134.1 Q194.1,134.8 197.0,135.6 Q200.0,136.4 203.0,137.4 Q205.9,138.4 208.9,139.5 Q211.9,140.6 214.8,141.8 Q217.8,143.0 220.8,144.4 Q223.7,145.7 226.7,147.1 Q229.7,148.6 232.6,150.1 Q235.6,151.6 238.6,153.2 Q241.5,154.8 244.5,156.5 Q247.5,158.1 250.4,159.8 Q253.4,161.5 256.4,163.2 Q259.3,165.0 262.3,166.7 Q265.3,168.4 268.2,170.1 Q271.2,171.8 274.2,173.5 Q277.1,175.1 280.1,176.7 Q283.1,178.3 286.0,179.9 Q289.0,181.4 292.0,182.9 Q294.9,184.3 297.9,185.7 Q300.9,187.1 303.8,188.3 Q306.8,189.6 309.8,190.7 Q312.7,191.8 315.7,192.8 Q318.7,193.8 321.6,194.7 Q324.6,195.6 327.6,196.3 Q330.5,197.0 333.5,197.6 Q336.5,198.2 339.4,198.7 Q342.4,199.2 345.4,199.5 Q348.3,199.8 351.3,200.0 Q354.3,200.2 357.2,200.3 Q360.2,200.4 363.2,200.3 Q366.1,200.3 369.1,200.1 Q372.1,200.0 375.0,199.8" fill="none" stroke="#9CC4A8" stroke-opacity="0.51" stroke-width="1"/><path d="M22.0,158.3 Q27.9,157.8 30.9,157.5 Q33.9,157.2 36.8,156.8 Q39.8,156.5 42.8,156.0 Q45.7,155.6 48.7,155.2 Q51.7,154.7 54.6,154.2 Q57.6,153.6 60.6,153.1 Q63.5,152.5 66.5,151.9 Q69.5,151.3 72.4,150.6 Q75.4,150.0 78.4,149.3 Q81.3,148.7 84.3,148.0 Q87.3,147.3 90.2,146.6 Q93.2,145.9 96.2,145.2 Q99.1,144.5 102.1,143.8 Q105.1,143.2 108.0,142.5 Q111.0,141.9 114.0,141.3 Q116.9,140.7 119.9,140.2 Q122.9,139.6 125.8,139.2 Q128.8,138.7 131.8,138.4 Q134.7,138.0 137.7,137.7 Q140.7,137.5 143.6,137.3 Q146.6,137.1 149.6,137.1 Q152.5,137.1 155.5,137.2 Q158.5,137.3 161.4,137.5 Q164.4,137.7 167.4,138.1 Q170.3,138.5 173.3,139.0 Q176.3,139.5 179.2,140.2 Q182.2,140.8 185.2,141.6 Q188.1,142.4 191.1,143.4 Q194.1,144.3 197.0,145.4 Q200.0,146.5 203.0,147.7 Q205.9,148.9 208.9,150.2 Q211.9,151.5 214.8,153.0 Q217.8,154.4 220.8,155.9 Q223.7,157.4 226.7,159.0 Q229.7,160.6 232.6,162.2 Q235.6,163.9 238.6,165.6 Q241.5,167.2 244.5,168.9 Q247.5,170.6 250.4,172.3 Q253.4,174.0 256.4,175.7 Q259.3,177.4 262.3,179.1 Q265.3,180.7 268.2,182.3 Q271.2,183.9 274.2,185.5 Q277.1,187.0 280.1,188.4 Q283.1,189.9 286.0,191.2 Q289.0,192.6 292.0,193.8 Q294.9,195.1 297.9,196.2 Q300.9,197.3 303.8,198.3 Q306.8,199.3 309.8,200.2 Q312.7,201.1 315.7,201.8 Q318.7,202.5 321.6,203.1 Q324.6,203.7 327.6,204.2 Q330.5,204.6 333.5,205.0 Q336.5,205.3 339.4,205.5 Q342.4,205.7 345.4,205.8 Q348.3,205.9 351.3,205.8 Q354.3,205.8 357.2,205.7 Q360.2,205.6 363.2,205.4 Q366.1,205.1 369.1,204.8 Q372.1,204.6 375.0,204.2" fill="none" stroke="#9CC4A8" stroke-opacity="0.52" stroke-width="1"/><path d="M22.0,165.9 Q27.9,165.1 30.9,164.6 Q33.9,164.1 36.8,163.6 Q39.8,163.1 42.8,162.5 Q45.7,161.9 48.7,161.3 Q51.7,160.6 54.6,160.0 Q57.6,159.3 60.6,158.6 Q63.5,157.9 66.5,157.2 Q69.5,156.5 72.4,155.7 Q75.4,155.0 78.4,154.2 Q81.3,153.5 84.3,152.8 Q87.3,152.0 90.2,151.3 Q93.2,150.6 96.2,149.9 Q99.1,149.2 102.1,148.6 Q105.1,148.0 108.0,147.4 Q111.0,146.9 114.0,146.4 Q116.9,145.9 119.9,145.5 Q122.9,145.1 125.8,144.8 Q128.8,144.5 131.8,144.3 Q134.7,144.1 137.7,144.1 Q140.7,144.0 143.6,144.1 Q146.6,144.1 149.6,144.3 Q152.5,144.5 155.5,144.9 Q158.5,145.2 161.4,145.7 Q164.4,146.2 167.4,146.8 Q170.3,147.5 173.3,148.2 Q176.3,149.0 179.2,149.9 Q182.2,150.8 185.2,151.9 Q188.1,152.9 191.1,154.1 Q194.1,155.3 197.0,156.6 Q200.0,157.9 203.0,159.2 Q205.9,160.6 208.9,162.1 Q211.9,163.6 214.8,165.1 Q217.8,166.7 220.8,168.3 Q223.7,169.9 226.7,171.5 Q229.7,173.2 232.6,174.8 Q235.6,176.5 238.6,178.1 Q241.5,179.8 244.5,181.4 Q247.5,183.1 250.4,184.7 Q253.4,186.3 256.4,187.9 Q259.3,189.4 262.3,190.9 Q265.3,192.4 268.2,193.8 Q271.2,195.2 274.2,196.6 Q277.1,197.9 280.1,199.1 Q283.1,200.3 286.0,201.4 Q289.0,202.5 292.0,203.4 Q294.9,204.4 297.9,205.2 Q300.9,206.1 303.8,206.8 Q306.8,207.5 309.8,208.1 Q312.7,208.7 315.7,209.1 Q318.7,209.6 321.6,209.9 Q324.6,210.2 327.6,210.4 Q330.5,210.6 333.5,210.7 Q336.5,210.7 339.4,210.7 Q342.4,210.7 345.4,210.6 Q348.3,210.5 351.3,210.3 Q354.3,210.1 357.2,209.8 Q360.2,209.5 363.2,209.2 Q366.1,208.9 369.1,208.5 Q372.1,208.1 375.0,207.7" fill="none" stroke="#9CC4A8" stroke-opacity="0.52" stroke-width="1"/><path d="M22.0,173.4 Q27.9,172.2 30.9,171.5 Q33.9,170.9 36.8,170.2 Q39.8,169.5 42.8,168.8 Q45.7,168.0 48.7,167.3 Q51.7,166.5 54.6,165.8 Q57.6,165.0 60.6,164.2 Q63.5,163.4 66.5,162.6 Q69.5,161.8 72.4,161.0 Q75.4,160.3 78.4,159.5 Q81.3,158.8 84.3,158.0 Q87.3,157.3 90.2,156.7 Q93.2,156.0 96.2,155.4 Q99.1,154.8 102.1,154.3 Q105.1,153.8 108.0,153.4 Q111.0,152.9 114.0,152.6 Q116.9,152.3 119.9,152.1 Q122.9,151.9 125.8,151.8 Q128.8,151.7 131.8,151.7 Q134.7,151.7 137.7,151.9 Q140.7,152.1 143.6,152.4 Q146.6,152.7 149.6,153.2 Q152.5,153.6 155.5,154.2 Q158.5,154.8 161.4,155.6 Q164.4,156.3 167.4,157.2 Q170.3,158.0 173.3,159.0 Q176.3,160.0 179.2,161.2 Q182.2,162.3 185.2,163.5 Q188.1,164.7 191.1,166.1 Q194.1,167.4 197.0,168.8 Q200.0,170.2 203.0,171.7 Q205.9,173.2 208.9,174.7 Q211.9,176.3 214.8,177.8 Q217.8,179.4 220.8,181.0 Q223.7,182.6 226.7,184.2 Q229.7,185.8 232.6,187.3 Q235.6,188.9 238.6,190.4 Q241.5,192.0 244.5,193.5 Q247.5,195.0 250.4,196.4 Q253.4,197.8 256.4,199.2 Q259.3,200.5 262.3,201.7 Q265.3,203.0 268.2,204.1 Q271.2,205.3 274.2,206.3 Q277.1,207.4 280.1,208.3 Q283.1,209.2 286.0,210.0 Q289.0,210.8 292.0,211.5 Q294.9,212.1 297.9,212.7 Q300.9,213.2 303.8,213.6 Q306.8,214.0 309.8,214.3 Q312.7,214.6 315.7,214.8 Q318.7,215.0 321.6,215.1 Q324.6,215.1 327.6,215.1 Q330.5,215.1 333.5,215.0 Q336.5,214.9 339.4,214.7 Q342.4,214.5 345.4,214.2 Q348.3,214.0 351.3,213.6 Q354.3,213.3 357.2,213.0 Q360.2,212.6 363.2,212.3 Q366.1,211.9 369.1,211.5 Q372.1,211.1 375.0,210.7" fill="none" stroke="#9CC4A8" stroke-opacity="0.52" stroke-width="1"/><path d="M22.0,180.6 Q27.9,179.1 30.9,178.3 Q33.9,177.5 36.8,176.7 Q39.8,175.8 42.8,175.0 Q45.7,174.2 48.7,173.4 Q51.7,172.5 54.6,171.7 Q57.6,170.9 60.6,170.0 Q63.5,169.2 66.5,168.4 Q69.5,167.7 72.4,166.9 Q75.4,166.2 78.4,165.5 Q81.3,164.8 84.3,164.2 Q87.3,163.6 90.2,163.0 Q93.2,162.5 96.2,162.0 Q99.1,161.6 102.1,161.2 Q105.1,160.9 108.0,160.6 Q111.0,160.4 114.0,160.3 Q116.9,160.2 119.9,160.2 Q122.9,160.2 125.8,160.3 Q128.8,160.5 131.8,160.8 Q134.7,161.0 137.7,161.5 Q140.7,161.9 143.6,162.4 Q146.6,163.0 149.6,163.7 Q152.5,164.4 155.5,165.2 Q158.5,166.0 161.4,166.9 Q164.4,167.9 167.4,168.9 Q170.3,170.0 173.3,171.1 Q176.3,172.3 179.2,173.5 Q182.2,174.8 185.2,176.1 Q188.1,177.4 191.1,178.8 Q194.1,180.2 197.0,181.7 Q200.0,183.1 203.0,184.6 Q205.9,186.1 208.9,187.6 Q211.9,189.0 214.8,190.5 Q217.8,192.0 220.8,193.5 Q223.7,195.0 226.7,196.4 Q229.7,197.8 232.6,199.2 Q235.6,200.6 238.6,202.0 Q241.5,203.3 244.5,204.5 Q247.5,205.8 250.4,206.9 Q253.4,208.1 256.4,209.2 Q259.3,210.2 262.3,211.2 Q265.3,212.1 268.2,213.0 Q271.2,213.8 274.2,214.6 Q277.1,215.3 280.1,215.9 Q283.1,216.5 286.0,217.0 Q289.0,217.5 292.0,217.9 Q294.9,218.2 297.9,218.5 Q300.9,218.8 303.8,218.9 Q306.8,219.1 309.8,219.1 Q312.7,219.2 315.7,219.1 Q318.7,219.1 321.6,219.0 Q324.6,218.9 327.6,218.7 Q330.5,218.5 333.5,218.3 Q336.5,218.0 339.4,217.7 Q342.4,217.4 345.4,217.1 Q348.3,216.8 351.3,216.5 Q354.3,216.1 357.2,215.8 Q360.2,215.4 363.2,215.0 Q366.1,214.7 369.1,214.4 Q372.1,214.0 375.0,213.7" fill="none" stroke="#9CC4A8" stroke-opacity="0.51" stroke-width="1"/><path d="M22.0,187.5 Q27.9,185.8 30.9,184.9 Q33.9,184.1 36.8,183.2 Q39.8,182.3 42.8,181.5 Q45.7,180.6 48.7,179.8 Q51.7,178.9 54.6,178.1 Q57.6,177.3 60.6,176.5 Q63.5,175.8 66.5,175.1 Q69.5,174.4 72.4,173.7 Q75.4,173.1 78.4,172.5 Q81.3,172.0 84.3,171.5 Q87.3,171.1 90.2,170.7 Q93.2,170.3 96.2,170.1 Q99.1,169.8 102.1,169.7 Q105.1,169.6 108.0,169.5 Q111.0,169.5 114.0,169.6 Q116.9,169.8 119.9,170.0 Q122.9,170.2 125.8,170.6 Q128.8,171.0 131.8,171.5 Q134.7,172.0 137.7,172.6 Q140.7,173.3 143.6,174.0 Q146.6,174.8 149.6,175.6 Q152.5,176.5 155.5,177.5 Q158.5,178.4 161.4,179.5 Q164.4,180.6 167.4,181.7 Q170.3,182.9 173.3,184.1 Q176.3,185.3 179.2,186.6 Q182.2,187.9 185.2,189.2 Q188.1,190.5 191.1,191.9 Q194.1,193.2 197.0,194.6 Q200.0,196.0 203.0,197.3 Q205.9,198.7 208.9,200.0 Q211.9,201.4 214.8,202.7 Q217.8,204.0 220.8,205.3 Q223.7,206.5 226.7,207.7 Q229.7,208.9 232.6,210.1 Q235.6,211.2 238.6,212.3 Q241.5,213.3 244.5,214.3 Q247.5,215.2 250.4,216.1 Q253.4,216.9 256.4,217.7 Q259.3,218.4 262.3,219.1 Q265.3,219.7 268.2,220.3 Q271.2,220.8 274.2,221.2 Q277.1,221.7 280.1,222.0 Q283.1,222.3 286.0,222.5 Q289.0,222.8 292.0,222.9 Q294.9,223.0 297.9,223.0 Q300.9,223.1 303.8,223.0 Q306.8,223.0 309.8,222.9 Q312.7,222.7 315.7,222.6 Q318.7,222.4 321.6,222.1 Q324.6,221.9 327.6,221.6 Q330.5,221.4 333.5,221.1 Q336.5,220.8 339.4,220.5 Q342.4,220.1 345.4,219.8 Q348.3,219.5 351.3,219.2 Q354.3,218.9 357.2,218.6 Q360.2,218.3 363.2,218.0 Q366.1,217.7 369.1,217.4 Q372.1,217.2 375.0,217.0" fill="none" stroke="#9CC4A8" stroke-opacity="0.49" stroke-width="1"/><path d="M22.0,194.6 Q27.9,192.8 30.9,191.9 Q33.9,191.0 36.8,190.1 Q39.8,189.3 42.8,188.5 Q45.7,187.7 48.7,186.9 Q51.7,186.1 54.6,185.4 Q57.6,184.7 60.6,184.1 Q63.5,183.4 66.5,182.9 Q69.5,182.3 72.4,181.8 Q75.4,181.4 78.4,181.0 Q81.3,180.6 84.3,180.4 Q87.3,180.1 90.2,180.0 Q93.2,179.8 96.2,179.8 Q99.1,179.7 102.1,179.8 Q105.1,179.9 108.0,180.1 Q111.0,180.3 114.0,180.7 Q116.9,181.0 119.9,181.4 Q122.9,181.9 125.8,182.4 Q128.8,183.0 131.8,183.7 Q134.7,184.4 137.7,185.1 Q140.7,185.9 143.6,186.8 Q146.6,187.7 149.6,188.6 Q152.5,189.6 155.5,190.6 Q158.5,191.6 161.4,192.7 Q164.4,193.8 167.4,195.0 Q170.3,196.1 173.3,197.3 Q176.3,198.5 179.2,199.7 Q182.2,200.9 185.2,202.1 Q188.1,203.4 191.1,204.6 Q194.1,205.8 197.0,207.0 Q200.0,208.2 203.0,209.4 Q205.9,210.5 208.9,211.6 Q211.9,212.8 214.8,213.8 Q217.8,214.9 220.8,215.9 Q223.7,216.9 226.7,217.8 Q229.7,218.7 232.6,219.5 Q235.6,220.4 238.6,221.1 Q241.5,221.9 244.5,222.5 Q247.5,223.2 250.4,223.7 Q253.4,224.3 256.4,224.7 Q259.3,225.2 262.3,225.6 Q265.3,225.9 268.2,226.2 Q271.2,226.4 274.2,226.6 Q277.1,226.8 280.1,226.9 Q283.1,227.0 286.0,227.0 Q289.0,227.0 292.0,226.9 Q294.9,226.8 297.9,226.7 Q300.9,226.6 303.8,226.4 Q306.8,226.2 309.8,226.0 Q312.7,225.8 315.7,225.5 Q318.7,225.3 321.6,225.0 Q324.6,224.7 327.6,224.4 Q330.5,224.1 333.5,223.8 Q336.5,223.5 339.4,223.3 Q342.4,223.0 345.4,222.7 Q348.3,222.4 351.3,222.2 Q354.3,221.9 357.2,221.7 Q360.2,221.5 363.2,221.3 Q366.1,221.1 369.1,221.0 Q372.1,220.8 375.0,220.7" fill="none" stroke="#9CC4A8" stroke-opacity="0.47" stroke-width="1"/><path d="M22.0,202.0 Q27.9,200.3 30.9,199.5 Q33.9,198.7 36.8,197.9 Q39.8,197.2 42.8,196.5 Q45.7,195.8 48.7,195.1 Q51.7,194.5 54.6,194.0 Q57.6,193.4 60.6,193.0 Q63.5,192.5 66.5,192.1 Q69.5,191.8 72.4,191.5 Q75.4,191.2 78.4,191.1 Q81.3,190.9 84.3,190.9 Q87.3,190.8 90.2,190.9 Q93.2,190.9 96.2,191.1 Q99.1,191.3 102.1,191.6 Q105.1,191.8 108.0,192.2 Q111.0,192.6 114.0,193.1 Q116.9,193.6 119.9,194.2 Q122.9,194.8 125.8,195.5 Q128.8,196.1 131.8,196.9 Q134.7,197.7 137.7,198.5 Q140.7,199.3 143.6,200.2 Q146.6,201.1 149.6,202.1 Q152.5,203.1 155.5,204.1 Q158.5,205.1 161.4,206.1 Q164.4,207.1 167.4,208.2 Q170.3,209.2 173.3,210.3 Q176.3,211.3 179.2,212.4 Q182.2,213.4 185.2,214.4 Q188.1,215.5 191.1,216.5 Q194.1,217.5 197.0,218.4 Q200.0,219.4 203.0,220.3 Q205.9,221.2 208.9,222.0 Q211.9,222.9 214.8,223.7 Q217.8,224.4 220.8,225.1 Q223.7,225.8 226.7,226.5 Q229.7,227.1 232.6,227.6 Q235.6,228.2 238.6,228.6 Q241.5,229.1 244.5,229.4 Q247.5,229.8 250.4,230.1 Q253.4,230.4 256.4,230.6 Q259.3,230.8 262.3,230.9 Q265.3,231.0 268.2,231.1 Q271.2,231.1 274.2,231.1 Q277.1,231.1 280.1,231.0 Q283.1,230.9 286.0,230.8 Q289.0,230.6 292.0,230.5 Q294.9,230.3 297.9,230.1 Q300.9,229.8 303.8,229.6 Q306.8,229.3 309.8,229.1 Q312.7,228.8 315.7,228.6 Q318.7,228.3 321.6,228.0 Q324.6,227.7 327.6,227.5 Q330.5,227.2 333.5,227.0 Q336.5,226.7 339.4,226.5 Q342.4,226.3 345.4,226.0 Q348.3,225.8 351.3,225.7 Q354.3,225.5 357.2,225.4 Q360.2,225.2 363.2,225.1 Q366.1,225.0 369.1,225.0 Q372.1,224.9 375.0,224.9" fill="none" stroke="#9CC4A8" stroke-opacity="0.44" stroke-width="1"/><path d="M22.0,210.3 Q27.9,208.8 30.9,208.2 Q33.9,207.5 36.8,206.9 Q39.8,206.3 42.8,205.8 Q45.7,205.3 48.7,204.8 Q51.7,204.4 54.6,204.0 Q57.6,203.7 60.6,203.4 Q63.5,203.2 66.5,203.0 Q69.5,202.8 72.4,202.8 Q75.4,202.7 78.4,202.8 Q81.3,202.8 84.3,202.9 Q87.3,203.1 90.2,203.3 Q93.2,203.5 96.2,203.8 Q99.1,204.2 102.1,204.6 Q105.1,205.0 108.0,205.5 Q111.0,206.0 114.0,206.5 Q116.9,207.1 119.9,207.8 Q122.9,208.4 125.8,209.1 Q128.8,209.8 131.8,210.6 Q134.7,211.3 137.7,212.1 Q140.7,212.9 143.6,213.8 Q146.6,214.6 149.6,215.5 Q152.5,216.3 155.5,217.2 Q158.5,218.1 161.4,219.0 Q164.4,219.8 167.4,220.7 Q170.3,221.6 173.3,222.4 Q176.3,223.3 179.2,224.1 Q182.2,224.9 185.2,225.7 Q188.1,226.5 191.1,227.2 Q194.1,227.9 197.0,228.6 Q200.0,229.3 203.0,229.9 Q205.9,230.6 208.9,231.1 Q211.9,231.7 214.8,232.2 Q217.8,232.7 220.8,233.1 Q223.7,233.5 226.7,233.9 Q229.7,234.2 232.6,234.5 Q235.6,234.8 238.6,235.0 Q241.5,235.2 244.5,235.3 Q247.5,235.5 250.4,235.5 Q253.4,235.6 256.4,235.6 Q259.3,235.6 262.3,235.6 Q265.3,235.5 268.2,235.4 Q271.2,235.3 274.2,235.2 Q277.1,235.0 280.1,234.9 Q283.1,234.7 286.0,234.5 Q289.0,234.2 292.0,234.0 Q294.9,233.8 297.9,233.5 Q300.9,233.3 303.8,233.0 Q306.8,232.8 309.8,232.5 Q312.7,232.2 315.7,232.0 Q318.7,231.7 321.6,231.5 Q324.6,231.2 327.6,231.0 Q330.5,230.8 333.5,230.6 Q336.5,230.4 339.4,230.2 Q342.4,230.1 345.4,229.9 Q348.3,229.8 351.3,229.7 Q354.3,229.5 357.2,229.5 Q360.2,229.4 363.2,229.3 Q366.1,229.3 369.1,229.3 Q372.1,229.3 375.0,229.3" fill="none" stroke="#9CC4A8" stroke-opacity="0.41" stroke-width="1"/><path d="M22.0,219.8 Q27.9,218.7 30.9,218.2 Q33.9,217.7 36.8,217.3 Q39.8,216.9 42.8,216.6 Q45.7,216.3 48.7,216.0 Q51.7,215.8 54.6,215.6 Q57.6,215.5 60.6,215.4 Q63.5,215.3 66.5,215.3 Q69.5,215.3 72.4,215.4 Q75.4,215.5 78.4,215.7 Q81.3,215.9 84.3,216.2 Q87.3,216.4 90.2,216.7 Q93.2,217.1 96.2,217.5 Q99.1,217.9 102.1,218.3 Q105.1,218.8 108.0,219.3 Q111.0,219.8 114.0,220.4 Q116.9,220.9 119.9,221.6 Q122.9,222.2 125.8,222.8 Q128.8,223.4 131.8,224.1 Q134.7,224.8 137.7,225.4 Q140.7,226.1 143.6,226.8 Q146.6,227.5 149.6,228.2 Q152.5,228.9 155.5,229.6 Q158.5,230.2 161.4,230.9 Q164.4,231.6 167.4,232.2 Q170.3,232.8 173.3,233.4 Q176.3,234.0 179.2,234.6 Q182.2,235.2 185.2,235.7 Q188.1,236.2 191.1,236.7 Q194.1,237.2 197.0,237.6 Q200.0,238.0 203.0,238.4 Q205.9,238.7 208.9,239.1 Q211.9,239.4 214.8,239.6 Q217.8,239.9 220.8,240.1 Q223.7,240.3 226.7,240.4 Q229.7,240.5 232.6,240.6 Q235.6,240.7 238.6,240.7 Q241.5,240.7 244.5,240.7 Q247.5,240.7 250.4,240.6 Q253.4,240.5 256.4,240.4 Q259.3,240.3 262.3,240.1 Q265.3,240.0 268.2,239.8 Q271.2,239.6 274.2,239.4 Q277.1,239.2 280.1,238.9 Q283.1,238.7 286.0,238.5 Q289.0,238.2 292.0,238.0 Q294.9,237.7 297.9,237.4 Q300.9,237.2 303.8,236.9 Q306.8,236.7 309.8,236.4 Q312.7,236.2 315.7,236.0 Q318.7,235.7 321.6,235.5 Q324.6,235.3 327.6,235.1 Q330.5,234.9 333.5,234.7 Q336.5,234.6 339.4,234.4 Q342.4,234.3 345.4,234.1 Q348.3,234.0 351.3,233.9 Q354.3,233.8 357.2,233.8 Q360.2,233.7 363.2,233.7 Q366.1,233.6 369.1,233.6 Q372.1,233.6 375.0,233.7" fill="none" stroke="#9CC4A8" stroke-opacity="0.37" stroke-width="1"/><path d="M22.0,230.7 Q27.9,230.0 30.9,229.7 Q33.9,229.4 36.8,229.2 Q39.8,229.0 42.8,228.8 Q45.7,228.7 48.7,228.6 Q51.7,228.5 54.6,228.5 Q57.6,228.5 60.6,228.6 Q63.5,228.6 66.5,228.8 Q69.5,228.9 72.4,229.1 Q75.4,229.3 78.4,229.5 Q81.3,229.8 84.3,230.0 Q87.3,230.3 90.2,230.7 Q93.2,231.0 96.2,231.4 Q99.1,231.8 102.1,232.2 Q105.1,232.6 108.0,233.1 Q111.0,233.5 114.0,234.0 Q116.9,234.5 119.9,235.0 Q122.9,235.5 125.8,235.9 Q128.8,236.4 131.8,236.9 Q134.7,237.4 137.7,237.9 Q140.7,238.4 143.6,238.9 Q146.6,239.4 149.6,239.9 Q152.5,240.4 155.5,240.8 Q158.5,241.3 161.4,241.7 Q164.4,242.1 167.4,242.5 Q170.3,242.9 173.3,243.3 Q176.3,243.7 179.2,244.0 Q182.2,244.3 185.2,244.6 Q188.1,244.9 191.1,245.1 Q194.1,245.4 197.0,245.6 Q200.0,245.8 203.0,245.9 Q205.9,246.1 208.9,246.2 Q211.9,246.3 214.8,246.4 Q217.8,246.4 220.8,246.5 Q223.7,246.5 226.7,246.5 Q229.7,246.5 232.6,246.4 Q235.6,246.3 238.6,246.2 Q241.5,246.1 244.5,246.0 Q247.5,245.9 250.4,245.7 Q253.4,245.5 256.4,245.4 Q259.3,245.2 262.3,245.0 Q265.3,244.7 268.2,244.5 Q271.2,244.3 274.2,244.0 Q277.1,243.8 280.1,243.5 Q283.1,243.3 286.0,243.0 Q289.0,242.7 292.0,242.5 Q294.9,242.2 297.9,241.9 Q300.9,241.7 303.8,241.4 Q306.8,241.2 309.8,240.9 Q312.7,240.7 315.7,240.4 Q318.7,240.2 321.6,240.0 Q324.6,239.7 327.6,239.5 Q330.5,239.3 333.5,239.1 Q336.5,239.0 339.4,238.8 Q342.4,238.6 345.4,238.5 Q348.3,238.3 351.3,238.2 Q354.3,238.1 357.2,238.0 Q360.2,237.9 363.2,237.9 Q366.1,237.8 369.1,237.7 Q372.1,237.7 375.0,237.7" fill="none" stroke="#9CC4A8" stroke-opacity="0.33" stroke-width="1"/><path d="M22.0,242.9 Q27.9,242.6 30.9,242.4 Q33.9,242.3 36.8,242.2 Q39.8,242.2 42.8,242.1 Q45.7,242.1 48.7,242.1 Q51.7,242.2 54.6,242.2 Q57.6,242.3 60.6,242.4 Q63.5,242.6 66.5,242.7 Q69.5,242.9 72.4,243.1 Q75.4,243.3 78.4,243.5 Q81.3,243.7 84.3,244.0 Q87.3,244.2 90.2,244.5 Q93.2,244.8 96.2,245.1 Q99.1,245.3 102.1,245.7 Q105.1,246.0 108.0,246.3 Q111.0,246.6 114.0,246.9 Q116.9,247.2 119.9,247.5 Q122.9,247.8 125.8,248.2 Q128.8,248.5 131.8,248.8 Q134.7,249.1 137.7,249.4 Q140.7,249.7 143.6,249.9 Q146.6,250.2 149.6,250.5 Q152.5,250.7 155.5,251.0 Q158.5,251.2 161.4,251.4 Q164.4,251.7 167.4,251.8 Q170.3,252.0 173.3,252.2 Q176.3,252.4 179.2,252.5 Q182.2,252.6 185.2,252.7 Q188.1,252.8 191.1,252.9 Q194.1,253.0 197.0,253.0 Q200.0,253.1 203.0,253.1 Q205.9,253.1 208.9,253.1 Q211.9,253.0 214.8,253.0 Q217.8,252.9 220.8,252.8 Q223.7,252.7 226.7,252.6 Q229.7,252.5 232.6,252.4 Q235.6,252.2 238.6,252.1 Q241.5,251.9 244.5,251.7 Q247.5,251.5 250.4,251.3 Q253.4,251.1 256.4,250.8 Q259.3,250.6 262.3,250.3 Q265.3,250.1 268.2,249.8 Q271.2,249.5 274.2,249.3 Q277.1,249.0 280.1,248.7 Q283.1,248.4 286.0,248.1 Q289.0,247.8 292.0,247.5 Q294.9,247.2 297.9,246.9 Q300.9,246.6 303.8,246.3 Q306.8,246.0 309.8,245.7 Q312.7,245.4 315.7,245.1 Q318.7,244.8 321.6,244.6 Q324.6,244.3 327.6,244.0 Q330.5,243.8 333.5,243.5 Q336.5,243.3 339.4,243.1 Q342.4,242.9 345.4,242.7 Q348.3,242.5 351.3,242.3 Q354.3,242.1 357.2,241.9 Q360.2,241.8 363.2,241.7 Q366.1,241.5 369.1,241.5 Q372.1,241.4 375.0,241.3" fill="none" stroke="#9CC4A8" stroke-opacity="0.29" stroke-width="1"/><path d="M22.0,256.2 Q27.9,256.1 30.9,256.0 Q33.9,256.0 36.8,256.0 Q39.8,256.0 42.8,256.0 Q45.7,256.0 48.7,256.1 Q51.7,256.1 54.6,256.2 Q57.6,256.3 60.6,256.4 Q63.5,256.5 66.5,256.6 Q69.5,256.7 72.4,256.8 Q75.4,256.9 78.4,257.1 Q81.3,257.2 84.3,257.3 Q87.3,257.5 90.2,257.6 Q93.2,257.8 96.2,257.9 Q99.1,258.0 102.1,258.2 Q105.1,258.3 108.0,258.5 Q111.0,258.6 114.0,258.7 Q116.9,258.9 119.9,259.0 Q122.9,259.1 125.8,259.3 Q128.8,259.4 131.8,259.5 Q134.7,259.6 137.7,259.7 Q140.7,259.8 143.6,259.9 Q146.6,260.0 149.6,260.1 Q152.5,260.2 155.5,260.3 Q158.5,260.3 161.4,260.4 Q164.4,260.4 167.4,260.5 Q170.3,260.5 173.3,260.5 Q176.3,260.6 179.2,260.6 Q182.2,260.6 185.2,260.6 Q188.1,260.5 191.1,260.5 Q194.1,260.5 197.0,260.4 Q200.0,260.4 203.0,260.3 Q205.9,260.2 208.9,260.1 Q211.9,260.0 214.8,259.9 Q217.8,259.7 220.8,259.6 Q223.7,259.4 226.7,259.3 Q229.7,259.1 232.6,258.9 Q235.6,258.7 238.6,258.5 Q241.5,258.2 244.5,258.0 Q247.5,257.7 250.4,257.5 Q253.4,257.2 256.4,256.9 Q259.3,256.6 262.3,256.3 Q265.3,256.0 268.2,255.7 Q271.2,255.3 274.2,255.0 Q277.1,254.6 280.1,254.3 Q283.1,253.9 286.0,253.6 Q289.0,253.2 292.0,252.8 Q294.9,252.4 297.9,252.1 Q300.9,251.7 303.8,251.3 Q306.8,250.9 309.8,250.5 Q312.7,250.2 315.7,249.8 Q318.7,249.4 321.6,249.1 Q324.6,248.7 327.6,248.3 Q330.5,248.0 333.5,247.7 Q336.5,247.3 339.4,247.0 Q342.4,246.7 345.4,246.5 Q348.3,246.2 351.3,245.9 Q354.3,245.7 357.2,245.5 Q360.2,245.3 363.2,245.1 Q366.1,244.9 369.1,244.8 Q372.1,244.7 375.0,244.6" fill="none" stroke="#9CC4A8" stroke-opacity="0.25" stroke-width="1"/><path d="M22.0,270.0 Q27.9,269.9 30.9,269.9 Q33.9,269.9 36.8,269.8 Q39.8,269.8 42.8,269.8 Q45.7,269.8 48.7,269.8 Q51.7,269.8 54.6,269.8 Q57.6,269.8 60.6,269.8 Q63.5,269.8 66.5,269.8 Q69.5,269.8 72.4,269.8 Q75.4,269.8 78.4,269.7 Q81.3,269.7 84.3,269.7 Q87.3,269.7 90.2,269.7 Q93.2,269.7 96.2,269.7 Q99.1,269.6 102.1,269.6 Q105.1,269.6 108.0,269.6 Q111.0,269.5 114.0,269.5 Q116.9,269.5 119.9,269.5 Q122.9,269.4 125.8,269.4 Q128.8,269.4 131.8,269.4 Q134.7,269.3 137.7,269.3 Q140.7,269.3 143.6,269.2 Q146.6,269.2 149.6,269.2 Q152.5,269.1 155.5,269.1 Q158.5,269.1 161.4,269.0 Q164.4,269.0 167.4,268.9 Q170.3,268.9 173.3,268.8 Q176.3,268.8 179.2,268.7 Q182.2,268.6 185.2,268.6 Q188.1,268.5 191.1,268.4 Q194.1,268.3 197.0,268.2 Q200.0,268.1 203.0,267.9 Q205.9,267.8 208.9,267.6 Q211.9,267.5 214.8,267.3 Q217.8,267.1 220.8,266.9 Q223.7,266.7 226.7,266.5 Q229.7,266.3 232.6,266.0 Q235.6,265.7 238.6,265.4 Q241.5,265.2 244.5,264.8 Q247.5,264.5 250.4,264.2 Q253.4,263.8 256.4,263.4 Q259.3,263.1 262.3,262.7 Q265.3,262.2 268.2,261.8 Q271.2,261.4 274.2,260.9 Q277.1,260.5 280.1,260.0 Q283.1,259.5 286.0,259.1 Q289.0,258.6 292.0,258.1 Q294.9,257.6 297.9,257.1 Q300.9,256.6 303.8,256.1 Q306.8,255.6 309.8,255.1 Q312.7,254.6 315.7,254.1 Q318.7,253.6 321.6,253.2 Q324.6,252.7 327.6,252.3 Q330.5,251.8 333.5,251.4 Q336.5,251.0 339.4,250.6 Q342.4,250.3 345.4,249.9 Q348.3,249.6 351.3,249.3 Q354.3,249.0 357.2,248.8 Q360.2,248.6 363.2,248.4 Q366.1,248.2 369.1,248.1 Q372.1,248.0 375.0,248.0" fill="none" stroke="#9CC4A8" stroke-opacity="0.21" stroke-width="1"/><path d="M22.0,283.6 Q27.9,283.4 30.9,283.3 Q33.9,283.2 36.8,283.2 Q39.8,283.1 42.8,282.9 Q45.7,282.8 48.7,282.7 Q51.7,282.6 54.6,282.5 Q57.6,282.3 60.6,282.2 Q63.5,282.1 66.5,281.9 Q69.5,281.8 72.4,281.6 Q75.4,281.5 78.4,281.3 Q81.3,281.1 84.3,281.0 Q87.3,280.8 90.2,280.7 Q93.2,280.5 96.2,280.3 Q99.1,280.2 102.1,280.0 Q105.1,279.9 108.0,279.7 Q111.0,279.6 114.0,279.5 Q116.9,279.3 119.9,279.2 Q122.9,279.1 125.8,279.0 Q128.8,278.8 131.8,278.7 Q134.7,278.6 137.7,278.5 Q140.7,278.4 143.6,278.3 Q146.6,278.3 149.6,278.2 Q152.5,278.1 155.5,278.0 Q158.5,277.9 161.4,277.8 Q164.4,277.8 167.4,277.7 Q170.3,277.6 173.3,277.5 Q176.3,277.4 179.2,277.3 Q182.2,277.2 185.2,277.1 Q188.1,277.0 191.1,276.9 Q194.1,276.7 197.0,276.6 Q200.0,276.4 203.0,276.2 Q205.9,276.1 208.9,275.9 Q211.9,275.6 214.8,275.4 Q217.8,275.2 220.8,274.9 Q223.7,274.6 226.7,274.3 Q229.7,274.0 232.6,273.6 Q235.6,273.2 238.6,272.8 Q241.5,272.4 244.5,272.0 Q247.5,271.6 250.4,271.1 Q253.4,270.6 256.4,270.1 Q259.3,269.6 262.3,269.0 Q265.3,268.5 268.2,267.9 Q271.2,267.3 274.2,266.7 Q277.1,266.1 280.1,265.5 Q283.1,264.9 286.0,264.3 Q289.0,263.6 292.0,263.0 Q294.9,262.4 297.9,261.7 Q300.9,261.1 303.8,260.5 Q306.8,259.8 309.8,259.2 Q312.7,258.6 315.7,258.0 Q318.7,257.5 321.6,256.9 Q324.6,256.4 327.6,255.8 Q330.5,255.3 333.5,254.9 Q336.5,254.4 339.4,254.0 Q342.4,253.6 345.4,253.3 Q348.3,253.0 351.3,252.7 Q354.3,252.4 357.2,252.2 Q360.2,252.0 363.2,251.9 Q366.1,251.8 369.1,251.8 Q372.1,251.8 375.0,251.9" fill="none" stroke="#9CC4A8" stroke-opacity="0.17" stroke-width="1"/><path d="M22.0,296.4 Q27.9,296.1 30.9,295.9 Q33.9,295.6 36.8,295.4 Q39.8,295.2 42.8,295.0 Q45.7,294.7 48.7,294.5 Q51.7,294.2 54.6,294.0 Q57.6,293.7 60.6,293.4 Q63.5,293.1 66.5,292.9 Q69.5,292.6 72.4,292.3 Q75.4,292.0 78.4,291.8 Q81.3,291.5 84.3,291.2 Q87.3,291.0 90.2,290.7 Q93.2,290.5 96.2,290.3 Q99.1,290.0 102.1,289.8 Q105.1,289.6 108.0,289.4 Q111.0,289.2 114.0,289.0 Q116.9,288.9 119.9,288.7 Q122.9,288.5 125.8,288.4 Q128.8,288.3 131.8,288.2 Q134.7,288.0 137.7,287.9 Q140.7,287.8 143.6,287.7 Q146.6,287.6 149.6,287.6 Q152.5,287.5 155.5,287.4 Q158.5,287.3 161.4,287.2 Q164.4,287.1 167.4,287.1 Q170.3,287.0 173.3,286.9 Q176.3,286.7 179.2,286.6 Q182.2,286.5 185.2,286.3 Q188.1,286.2 191.1,286.0 Q194.1,285.8 197.0,285.6 Q200.0,285.4 203.0,285.1 Q205.9,284.9 208.9,284.6 Q211.9,284.3 214.8,283.9 Q217.8,283.6 220.8,283.2 Q223.7,282.8 226.7,282.3 Q229.7,281.8 232.6,281.3 Q235.6,280.8 238.6,280.3 Q241.5,279.7 244.5,279.1 Q247.5,278.5 250.4,277.9 Q253.4,277.2 256.4,276.5 Q259.3,275.8 262.3,275.1 Q265.3,274.4 268.2,273.6 Q271.2,272.9 274.2,272.1 Q277.1,271.4 280.1,270.6 Q283.1,269.8 286.0,269.0 Q289.0,268.2 292.0,267.5 Q294.9,266.7 297.9,265.9 Q300.9,265.2 303.8,264.4 Q306.8,263.7 309.8,263.0 Q312.7,262.3 315.7,261.7 Q318.7,261.1 321.6,260.5 Q324.6,259.9 327.6,259.4 Q330.5,258.9 333.5,258.4 Q336.5,258.0 339.4,257.6 Q342.4,257.3 345.4,257.0 Q348.3,256.8 351.3,256.6 Q354.3,256.4 357.2,256.4 Q360.2,256.3 363.2,256.3 Q366.1,256.3 369.1,256.5 Q372.1,256.6 375.0,256.9" fill="none" stroke="#9CC4A8" stroke-opacity="0.14" stroke-width="1"/><path d="M22.0,308.1 Q27.9,307.4 30.9,307.1 Q33.9,306.8 36.8,306.4 Q39.8,306.1 42.8,305.7 Q45.7,305.4 48.7,305.0 Q51.7,304.6 54.6,304.3 Q57.6,303.9 60.6,303.5 Q63.5,303.2 66.5,302.8 Q69.5,302.5 72.4,302.2 Q75.4,301.8 78.4,301.5 Q81.3,301.2 84.3,300.9 Q87.3,300.6 90.2,300.4 Q93.2,300.1 96.2,299.9 Q99.1,299.6 102.1,299.4 Q105.1,299.2 108.0,299.0 Q111.0,298.9 114.0,298.7 Q116.9,298.6 119.9,298.5 Q122.9,298.3 125.8,298.2 Q128.8,298.1 131.8,298.0 Q134.7,298.0 137.7,297.9 Q140.7,297.8 143.6,297.8 Q146.6,297.7 149.6,297.6 Q152.5,297.6 155.5,297.5 Q158.5,297.4 161.4,297.3 Q164.4,297.2 167.4,297.1 Q170.3,297.0 173.3,296.9 Q176.3,296.7 179.2,296.5 Q182.2,296.4 185.2,296.1 Q188.1,295.9 191.1,295.6 Q194.1,295.4 197.0,295.0 Q200.0,294.7 203.0,294.3 Q205.9,293.9 208.9,293.4 Q211.9,293.0 214.8,292.5 Q217.8,292.0 220.8,291.4 Q223.7,290.8 226.7,290.2 Q229.7,289.5 232.6,288.8 Q235.6,288.1 238.6,287.3 Q241.5,286.6 244.5,285.8 Q247.5,285.0 250.4,284.1 Q253.4,283.3 256.4,282.4 Q259.3,281.5 262.3,280.6 Q265.3,279.7 268.2,278.8 Q271.2,277.9 274.2,276.9 Q277.1,276.0 280.1,275.1 Q283.1,274.2 286.0,273.3 Q289.0,272.4 292.0,271.5 Q294.9,270.7 297.9,269.8 Q300.9,269.0 303.8,268.3 Q306.8,267.5 309.8,266.8 Q312.7,266.1 315.7,265.5 Q318.7,264.9 321.6,264.3 Q324.6,263.8 327.6,263.4 Q330.5,262.9 333.5,262.6 Q336.5,262.3 339.4,262.1 Q342.4,261.8 345.4,261.7 Q348.3,261.6 351.3,261.6 Q354.3,261.6 357.2,261.8 Q360.2,261.9 363.2,262.1 Q366.1,262.3 369.1,262.7 Q372.1,263.0 375.0,263.5" fill="none" stroke="#9CC4A8" stroke-opacity="0.11" stroke-width="1"/><path d="M22.0,194.6 Q27.9,192.8 30.9,191.9 Q33.9,191.0 36.8,190.1 Q39.8,189.3 42.8,188.5 Q45.7,187.7 48.7,186.9 Q51.7,186.1 54.6,185.4 Q57.6,184.7 60.6,184.1 Q63.5,183.4 66.5,182.9 Q69.5,182.3 72.4,181.8 Q75.4,181.4 78.4,181.0 Q81.3,180.6 84.3,180.4 Q87.3,180.1 90.2,180.0 Q93.2,179.8 96.2,179.8 Q99.1,179.7 102.1,179.8 Q105.1,179.9 108.0,180.1 Q111.0,180.3 114.0,180.7 Q116.9,181.0 119.9,181.4 Q122.9,181.9 125.8,182.4 Q128.8,183.0 131.8,183.7 Q134.7,184.4 137.7,185.1 Q140.7,185.9 143.6,186.8 Q146.6,187.7 149.6,188.6 Q152.5,189.6 155.5,190.6 Q158.5,191.6 161.4,192.7 Q164.4,193.8 167.4,195.0 Q170.3,196.1 173.3,197.3 Q176.3,198.5 179.2,199.7 Q182.2,200.9 185.2,202.1 Q188.1,203.4 191.1,204.6 Q194.1,205.8 197.0,207.0 Q200.0,208.2 203.0,209.4 Q205.9,210.5 208.9,211.6 Q211.9,212.8 214.8,213.8 Q217.8,214.9 220.8,215.9 Q223.7,216.9 226.7,217.8 Q229.7,218.7 232.6,219.5 Q235.6,220.4 238.6,221.1 Q241.5,221.9 244.5,222.5 Q247.5,223.2 250.4,223.7 Q253.4,224.3 256.4,224.7 Q259.3,225.2 262.3,225.6 Q265.3,225.9 268.2,226.2 Q271.2,226.4 274.2,226.6 Q277.1,226.8 280.1,226.9 Q283.1,227.0 286.0,227.0 Q289.0,227.0 292.0,226.9 Q294.9,226.8 297.9,226.7 Q300.9,226.6 303.8,226.4 Q306.8,226.2 309.8,226.0 Q312.7,225.8 315.7,225.5 Q318.7,225.3 321.6,225.0 Q324.6,224.7 327.6,224.4 Q330.5,224.1 333.5,223.8 Q336.5,223.5 339.4,223.3 Q342.4,223.0 345.4,222.7 Q348.3,222.4 351.3,222.2 Q354.3,221.9 357.2,221.7 Q360.2,221.5 363.2,221.3 Q366.1,221.1 369.1,221.0 Q372.1,220.8 375.0,220.7" fill="none" stroke="#C6E34A" stroke-width="7" stroke-opacity=".55" filter="url(#bl)"/><path d="M22.0,194.6 Q27.9,192.8 30.9,191.9 Q33.9,191.0 36.8,190.1 Q39.8,189.3 42.8,188.5 Q45.7,187.7 48.7,186.9 Q51.7,186.1 54.6,185.4 Q57.6,184.7 60.6,184.1 Q63.5,183.4 66.5,182.9 Q69.5,182.3 72.4,181.8 Q75.4,181.4 78.4,181.0 Q81.3,180.6 84.3,180.4 Q87.3,180.1 90.2,180.0 Q93.2,179.8 96.2,179.8 Q99.1,179.7 102.1,179.8 Q105.1,179.9 108.0,180.1 Q111.0,180.3 114.0,180.7 Q116.9,181.0 119.9,181.4 Q122.9,181.9 125.8,182.4 Q128.8,183.0 131.8,183.7 Q134.7,184.4 137.7,185.1 Q140.7,185.9 143.6,186.8 Q146.6,187.7 149.6,188.6 Q152.5,189.6 155.5,190.6 Q158.5,191.6 161.4,192.7 Q164.4,193.8 167.4,195.0 Q170.3,196.1 173.3,197.3 Q176.3,198.5 179.2,199.7 Q182.2,200.9 185.2,202.1 Q188.1,203.4 191.1,204.6 Q194.1,205.8 197.0,207.0 Q200.0,208.2 203.0,209.4 Q205.9,210.5 208.9,211.6 Q211.9,212.8 214.8,213.8 Q217.8,214.9 220.8,215.9 Q223.7,216.9 226.7,217.8 Q229.7,218.7 232.6,219.5 Q235.6,220.4 238.6,221.1 Q241.5,221.9 244.5,222.5 Q247.5,223.2 250.4,223.7 Q253.4,224.3 256.4,224.7 Q259.3,225.2 262.3,225.6 Q265.3,225.9 268.2,226.2 Q271.2,226.4 274.2,226.6 Q277.1,226.8 280.1,226.9 Q283.1,227.0 286.0,227.0 Q289.0,227.0 292.0,226.9 Q294.9,226.8 297.9,226.7 Q300.9,226.6 303.8,226.4 Q306.8,226.2 309.8,226.0 Q312.7,225.8 315.7,225.5 Q318.7,225.3 321.6,225.0 Q324.6,224.7 327.6,224.4 Q330.5,224.1 333.5,223.8 Q336.5,223.5 339.4,223.3 Q342.4,223.0 345.4,222.7 Q348.3,222.4 351.3,222.2 Q354.3,221.9 357.2,221.7 Q360.2,221.5 363.2,221.3 Q366.1,221.1 369.1,221.0 Q372.1,220.8 375.0,220.7" fill="none" stroke="#C6E34A" stroke-width="2.2" stroke-linecap="round"/></g>
<line x1="265.3" y1="30" x2="265.3" y2="310" stroke="#F6F1E4" stroke-opacity=".28" stroke-dasharray="2 5"/><circle cx="265.3" cy="225.9" r="9" fill="none" stroke="#F6F1E4" stroke-opacity=".8" stroke-width="1.2"/><circle cx="81.3" cy="180.6" r="4.2" fill="#06251F" stroke="#C6E34A" stroke-width="1.8"/><circle cx="176.3" cy="198.5" r="4.2" fill="#06251F" stroke="#C6E34A" stroke-width="1.8"/><circle cx="265.3" cy="225.9" r="4.2" fill="#06251F" stroke="#C6E34A" stroke-width="1.8"/><circle cx="336.5" cy="223.5" r="4.2" fill="#06251F" stroke="#C6E34A" stroke-width="1.8"/>
<g stroke="#9CC4A8" stroke-opacity=".5" stroke-width="1.2" stroke-linecap="round"><line x1="28" y1="316" x2="372" y2="316" stroke-opacity=".25"/><line x1="28" y1="316" x2="68" y2="316" stroke="#C6E34A" stroke-opacity=".9" stroke-width="2"/></g>
</svg>"""
TITLE_HTML = "AI Changed the Game. <em>But What's the Real Story?</em>"
html(f"""
<div class="hero">
<div class="copy">
<div class="eyebrow">Stack Overflow Developer Survey 2024 · An interactive data exhibition</div>
<h1>{TITLE_HTML}</h1>
<div class="sub">{SUBTITLE}</div>
<p>AI is transforming software development. But does using AI mean trusting it? Walk through the analysis, find yourself in the data, and test your intuition against {N_ALL:,} survey responses.</p>
<div class="chips"><span class="chip"><b>{ADOPT:.0f}%</b> use AI</span><span class="chip"><b>{PX["p_both"]:.0f}%</b> feel faster and doubt it</span><span class="chip"><b>{P_HIGH_TRUST:.1f}%</b> highly trust it</span></div>
</div>
<div class="neural">{NEURAL}</div>
</div>
<div class="note">Percentages are calculated within each group, on respondents who actually answered the question. Sample sizes are shown throughout.</div>
""")
if FILTERS_ON:
    parts = []
    if F_ROLES_ON:
        parts.append("roles" + f": {len(f_roles)}")
    if F_COUNTRIES_ON:
        parts.append("countries" + f": {len(f_countries)}")
    if F_EXP_ON:
        parts.append("experience" + ": " + ", ".join(x for x in f_exp))
    if tuple(f_years) != (YMIN, YMAX):
        parts.append("years" + f": {f_years[0]}–{f_years[1]}")
    html(f'<div class="insight"><b>Filters are active</b> ({" · ".join(parts)}). '
         f'The analyses below describe {N_F:,} of {N_ALL:,} respondents. Find Yourself, the Trust Test and the final summary stay on the full survey.</div>')

# ----------------------------------------------------------------------------
# 00 - Data overview: summary statistics, data preview, who is in the survey
# ----------------------------------------------------------------------------
section("00", "Data overview", "ov", "Who Is in the Survey?",
        "Start with the size and shape of the data. Top row: the full survey. Bottom row: the audience you selected in the sidebar.")


def kpi(value, label, kind):
    return f'<div class="card {kind}"><div class="stat">{value}</div><div class="stat-label">{label}</div></div>'


nF_ai, cF_ai = adopt_counts(fdf)
adF = cF_ai["Uses AI"] / nF_ai * 100 if nF_ai else None
accF, nF_acc = trust_dist(fdf)
yrsF = fdf["YearsCodePro_clean"].dropna()
na = "–"
html(f'<div class="grp">Full survey (always the same)</div><div class="kpis">'
     + kpi(f"{N_ALL:,}", "survey respondents", "full")
     + kpi(f"{N_COLS}", "variables in the cleaned dataset (10 selected + 2 created)", "full")
     + kpi(f"{df['Country'].nunique()}", "countries represented", "full")
     + kpi(f"{df['DevType'].nunique()}", "developer roles", "full")
     + kpi(pct(ADOPT), f"use AI ({nlab(N_SEL)})", "full")
     + kpi(f"{df['YearsCodePro_clean'].median():.0f}", f"median years of professional coding ({nlab(int(df['YearsCodePro_clean'].notna().sum()))})", "full")
     + '</div>')
html(f'<div class="grp">Selected audience (changes with the filters)</div><div class="kpis">'
     + kpi(f"{N_F:,}", f"respondents selected ({N_F / N_ALL * 100:.1f}% of the survey)", "sel")
     + kpi(f"{fdf['Country'].nunique()}", "countries", "sel")
     + kpi(f"{fdf['DevType'].nunique()}", "developer roles", "sel")
     + kpi(pct(adF) if adF is not None else na, f"use AI ({nlab(nF_ai)})", "sel")
     + kpi(pct(float(accF[["Highly trust", "Somewhat trust"]].sum())) if nF_acc else na,
           f"of AI users trust AI accuracy ({nlab(nF_acc)})", "sel")
     + kpi(f"{yrsF.median():.0f}" if len(yrsF) else na, f"median years of professional coding ({nlab(len(yrsF))})", "sel")
     + '</div>')
if not FILTERS_ON:
    html(f'<div class="note">No filters are active, so the selected audience equals the full survey.</div>')

with st.expander("Data preview (cleaned dataset, respects your filters)"):
    prev_n = st.slider("Rows to show", 5, 100, 10, key="preview_rows")
    st.caption(f"{N_F:,} rows × {N_COLS} columns in the selected audience (full file: {N_ALL:,} × {N_COLS}). Column names are the original dataset field names.")
    if N_F:
        show_table(fdf.head(prev_n))
    else:
        st.info("No rows match the current filters.")

# who is in the survey (univariate: experience levels, top-10 roles)
u1, u2 = st.columns([1, 1.25])
with u1:
    st.markdown(f"**Experience levels**")
    ex_n = [int((fdf["ExperienceLevel"] == l).sum()) for l in EXP_ORDER]
    if sum(ex_n) >= MIN_N:
        tot = sum(ex_n)
        fig = go.Figure(go.Bar(
            x=[l for l in EXP_ORDER], y=[v / tot * 100 for v in ex_n],
            marker=dict(color=[C["sage"], C["forest"], C["forest"], C["sage"]]),
            text=[f"{v / tot * 100:.1f}%" for v in ex_n], textposition="outside", textfont=dict(color=C["ink"]),
            customdata=ex_n, hovertemplate="%{x}<br>%{y:.1f}%<br>" + ND + " = %{customdata:,}<extra></extra>"))
        fig.update_yaxes(ticksuffix="%", range=[0, max(v / tot * 100 for v in ex_n) * 1.25])
        show(style(fig, 300))
        html(f'<div class="note">Share of respondents with a known experience level ({nlab(tot)}). '
             f'{N_F - tot:,} have no years-of-experience answer.</div>')
    else:
        low_data(sum(ex_n))
with u2:
    st.markdown(f"**The 10 most common roles in the survey**")
    rc = fdf["DevType"].value_counts().reindex(TOP_ROLES).fillna(0).astype(int)
    if int(rc.sum()) >= MIN_N:
        rc = rc.sort_values()
        fig = go.Figure(go.Bar(
            y=[r for r in rc.index], x=rc.values, orientation="h", marker=dict(color=C["forest"]),
            text=[f"{v:,}" for v in rc.values], textposition="outside", textfont=dict(color=C["ink"]),
            hovertemplate="%{y}<br>%{x:,}<extra></extra>"))
        fig.update_xaxes(range=[0, max(rc.values) * 1.2])
        show(style(fig, 300, margin=dict(l=10, r=40, t=10, b=10)))
        html(f'<div class="note">Role counts among the selected audience. “Top 10” is defined on the full survey, as in the EDA.</div>')
    else:
        low_data(int(rc.sum()))


# ----------------------------------------------------------------------------
# 01 - Explore Your Developer World (country + role + experience explorer; full-survey baseline stays fixed)
# ----------------------------------------------------------------------------
section("01", "Interactive exploration", "explore", "Explore Your Developer World",
        "Choose a country and developer role to discover how AI adoption, benefits, challenges, and trust differ across developers. "
        "Every result is compared with the full survey, which never changes. This explorer is separate from the sidebar filters and from Find Yourself in Data.")
ALL_C, ALL_R, ALL_E = "All Countries", "All Roles", "All levels"
if "ex_reset" not in st.session_state:
    st.session_state["ex_reset"] = 0


def reset_explore():
    st.session_state["ex_reset"] += 1  # fresh widget keys = back to "All"


XR = st.session_state["ex_reset"]
with container("explore_box"):
    x1, x2, x3, x4 = st.columns([1.4, 1.7, 1.3, 0.6])
    with x1:
        ex_country = st.selectbox("Country (type to search)", [ALL_C] + ALL_COUNTRIES, key=f"ex_country_{XR}")
    with x2:
        ex_role = st.selectbox("Developer role (type to search)", [ALL_R] + ALL_ROLES, key=f"ex_role_{XR}")
    with x3:
        ex_exp = st.selectbox("Experience level", [ALL_E] + EXP_ORDER, key=f"ex_exp_{XR}",
                              format_func=lambda x: x if x == ALL_E else f"{x} ({EXP_HELP[x]})")
    with x4:
        st.markdown("&nbsp;")
        st.button("Reset", on_click=reset_explore, key="ex_reset_btn")

xg = df
if ex_country != ALL_C:
    xg = xg[xg["Country"] == ex_country]
if ex_role != ALL_R:
    xg = xg[xg["DevType"] == ex_role]
if ex_exp != ALL_E:
    xg = xg[xg["ExperienceLevel"] == ex_exp]
xg_n = len(xg)
xg_label = " · ".join([ex_role if ex_role != ALL_R else "All roles", ex_country if ex_country != ALL_C else "All countries"]
                      + ([ex_exp] if ex_exp != ALL_E else []))
EX_METRICS = [
    ("adopt", "Use AI"), ("trust", "Trust AI accuracy"), ("distrust", "Distrust AI accuracy"),
    ("prod", "Report a productivity benefit"), ("acc", "Report better coding accuracy"),
    ("notrust", "Name “don't trust the output” as a challenge"), ("both", "Productivity benefit and distrust together"),
]
EX_BASIS = {
    "adopt": "respondents who answered the AI-use question", "trust": "AI users who rated AI accuracy",
    "distrust": "AI users who rated AI accuracy", "prod": "AI users who answered the benefits question",
    "acc": "AI users who answered the benefits question", "notrust": "AI users who answered the challenges question",
    "both": "AI users who answered both the benefits and challenges questions",
}


def explore_counts(frame: pd.DataFrame) -> dict:
    """(numerator, valid denominator) per metric, using the EDA definitions and question-specific denominators."""
    out = {}
    s = frame["AISelect"].dropna()
    out["adopt"] = (int((s == "Yes").sum()), len(s))
    a = frame["AIAcc"].dropna()
    out["trust"] = (int(a.isin(["Highly trust", "Somewhat trust"]).sum()), len(a))
    out["distrust"] = (int(a.isin(["Somewhat distrust", "Highly distrust"]).sum()), len(a))
    b = frame["AIBen"].dropna()
    out["prod"] = (int(b.str.contains(PRODUCTIVITY_KEY, regex=False).sum()), len(b))
    out["acc"] = (int(b.str.contains(ACCURACY_KEY, regex=False).sum()), len(b))
    c = frame["AIChallenges"].dropna()
    out["notrust"] = (int(c.str.contains(DISTRUST_KEY, regex=False).sum()), len(c))
    px_ = paradox_stats(frame)
    out["both"] = (px_["both"], px_["n"])
    return out


EX_BASE = explore_counts(df)  # fixed full-survey baseline: never recomputed from the selected group


def dumbbell(labels, group_vals, base_vals, ns, los, his, height, base_name="Full survey", group_name="Selected group", xmax=100):
    """Group (lime, with 95% interval) vs full-survey baseline (sage) for each row."""
    fig = go.Figure()
    for lab, g, b_ in zip(labels, group_vals, base_vals):
        fig.add_trace(go.Scatter(x=[b_, g], y=[lab, lab], mode="lines", line=dict(color=C["lime"] if g >= b_ else C["warn"], width=5),
                                 hoverinfo="skip", showlegend=False, opacity=.75))
    fig.add_trace(go.Scatter(x=base_vals, y=labels, mode="markers", name=base_name,
                             marker=dict(size=15, color=C["sage"], symbol="diamond", line=dict(color=C["ink"], width=2)),
                             hovertemplate="<b>%{y}</b><br>" + base_name + ": %{x:.1f}%<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=group_vals, y=labels, mode="markers+text", name=group_name,
        marker=dict(size=19, color=C["lime"], line=dict(color=C["ink"], width=2)),
        error_x=dict(type="data", symmetric=False, array=[h - g for h, g in zip(his, group_vals)], arrayminus=[g - l for g, l in zip(group_vals, los)],
                     color=C["muted"], thickness=1.3, width=4),
        text=[f"<b>{g:.1f}%</b>" for g in group_vals], textposition="top center", textfont=dict(color=C["ink"], size=12),
        customdata=list(zip(ns, los, his)),
        hovertemplate="<b>%{y}</b><br>" + group_name + ": %{x:.1f}%<br>95% CI %{customdata[1]:.1f}% – %{customdata[2]:.1f}%<br>" + ND + " = %{customdata[0]:,}<extra></extra>"))
    fig.update_xaxes(range=[0, xmax], ticksuffix="%")
    fig.update_yaxes(categoryorder="array", categoryarray=labels, autorange="reversed")
    fig.update_layout(legend=dict(orientation="h", y=1.1, font=dict(color=C["ink"])))
    return style(fig, height, legend=True, margin=dict(l=10, r=30, t=40, b=30))


if xg_n == 0:
    html('<div class="lowdata">No respondents match this combination of country, role and experience level. '
         'Try “All Countries”, “All Roles” or “All levels” to widen the selection.</div>')
elif xg_n < MIN_N:
    html(f'<div class="kpis">{kpi(f"{xg_n:,}", f"respondents match {xg_label}", "sel")}</div>')
    low_data(xg_n)
    html('<div class="note">Percentages are not shown for groups under 30 respondents because they would be unreliable. Widen the selection to see results.</div>')
else:
    XC = explore_counts(xg)
    rows_x, skipped = [], []
    for key, lab in EX_METRICS:
        k, n_ = XC[key]
        bk, bn_ = EX_BASE[key]
        if n_ >= MIN_N:
            lo_, hi_ = wilson(k, n_)
            rows_x.append(dict(key=key, label=lab, p=k / n_ * 100, n=n_, lo=lo_, hi=hi_, base=bk / bn_ * 100, base_n=bn_,
                               clear=not (lo_ <= bk / bn_ * 100 <= hi_)))
        else:
            skipped.append(lab)
    byk = {r["key"]: r for r in rows_x}
    # headline cards: audience size, adoption, trust, productivity (group value, valid n, difference vs full survey)
    def card(key, title):
        r = byk.get(key)
        if not r:
            return kpi("–", f"{title}: fewer than {MIN_N} valid answers", "sel")
        d_ = r["p"] - r["base"]
        return kpi(f"{r['p']:.1f}%", f"{title} ({nlab(r['n'])}). {abs(d_):.1f} pts {'above' if d_ >= 0 else 'below'} the full survey ({r['base']:.1f}%)", "sel")
    html('<div class="kpis">'
         + kpi(f"{xg_n:,}", f"respondents match {xg_label} ({xg_n / N_ALL * 100:.1f}% of the survey)", "sel")
         + card("adopt", "use AI") + card("trust", "of AI users trust AI accuracy") + card("distrust", "of AI users distrust AI accuracy") + card("prod", "report a productivity benefit")
         + '</div>')
    st.markdown("&nbsp;")
    # 1) overall comparison
    st.markdown("**How this group compares with the full survey**")
    if rows_x:
        labs_x = [f"{r['label']}  <span style='font-size:11px'>{ND}={r['n']:,}</span>" for r in rows_x]
        show(dumbbell(labs_x, [r["p"] for r in rows_x], [r["base"] for r in rows_x], [r["n"] for r in rows_x],
                      [r["lo"] for r in rows_x], [r["hi"] for r in rows_x], max(300, 56 * len(rows_x) + 100)))
        html('<div class="note">Lime dot = selected group with its 95% confidence interval; sage diamond = full survey (fixed). Each measure has its own base: '
             + "; ".join(f"{r['label'].lower()}: {EX_BASIS[r['key']]}" for r in rows_x[:3]) + '… (hover for details). '
             + (f"Hidden because fewer than {MIN_N} valid answers: {', '.join(s.lower() for s in skipped)}." if skipped else "") + '</div>')
    else:
        low_data(xg_n)
    # 2) benefits and challenges, group vs baseline
    ben_g, ben_p, ben_n = multi_stats(xg, "AIBen")
    ch_g, ch_p, ch_n = multi_stats(xg, "AIChallenges", ch_key)
    bcol1, bcol2 = st.columns(2)
    with bcol1:
        st.markdown("**AI benefits reported**")
        if ben_n >= MIN_N:
            keys_b = list(BEN_PCT_FULL.sort_values(ascending=False).index)
            vals_g = [float(ben_p.get(k, 0.0)) for k in keys_b]
            los_b, his_b = zip(*[wilson(int(ben_g.get(k, 0)), ben_n) for k in keys_b])
            show(dumbbell(keys_b, vals_g, [float(BEN_PCT_FULL[k]) for k in keys_b], [ben_n] * len(keys_b), list(los_b), list(his_b), 420))
            html(f'<div class="note">Multi-select answers from AI users; {nlab(ben_n)} in this group vs {nlab(N_BEN_FULL)} in the full survey. Reported or expected benefits, not measured improvements.</div>')
        else:
            low_data(ben_n)
    with bcol2:
        st.markdown("**AI challenges reported**")
        if ch_n >= MIN_N:
            keys_c = list(CH_PCT_FULL.sort_values(ascending=False).index)
            vals_c = [float(ch_p.get(k, 0.0)) for k in keys_c]
            los_c, his_c = zip(*[wilson(int(ch_g.get(k, 0)), ch_n) for k in keys_c])
            show(dumbbell([ch_name(k) for k in keys_c], vals_c, [float(CH_PCT_FULL[k]) for k in keys_c], [ch_n] * len(keys_c), list(los_c), list(his_c), 420))
            html(f'<div class="note">Multi-select answers from AI users; {nlab(ch_n)} in this group vs {nlab(N_CH_FULL)} in the full survey. Separate from the accuracy rating below.</div>')
        else:
            low_data(ch_n)
    # 3) accuracy / trust distribution and adoption by experience
    tcol1, tcol2 = st.columns(2)
    with tcol1:
        st.markdown("**Trust in AI accuracy**")
        xt, xt_n = trust_dist(xg)
        if xt_n >= MIN_N:
            show(diverging_trust([("Selected group", xt, xt_n), ("Full survey", ACC_PCT, N_ACC)], 290))
            html('<div class="note">Left = distrust, centre = neutral, right = trust. Asked only of AI users.</div>')
        else:
            low_data(xt_n)
    with tcol2:
        st.markdown("**AI adoption by experience level**")
        if ex_exp != ALL_E:
            html('<div class="note">An experience level is selected above, so this comparison across levels is hidden. Choose “All levels” to see it.</div>')
        else:
            xe = exp_table(xg, MIN_N)
            if len(xe) >= 2:
                fig = go.Figure()
                for r in xe.itertuples():
                    fig.add_trace(go.Scatter(x=[r.level, r.level], y=[0, r.adopt], mode="lines", line=dict(color=C["forest"], width=4), hoverinfo="skip", showlegend=False))
                fig.add_trace(go.Scatter(x=xe["level"], y=xe["adopt"], mode="markers+text", name="Selected group", text=[f"<b>{v:.0f}%</b>" for v in xe["adopt"]],
                                         textposition="top center", textfont=dict(color=C["ink"]), marker=dict(size=20, color=C["lime"], line=dict(color=C["ink"], width=2)),
                                         customdata=xe["n"], hovertemplate="%{x}<br>%{y:.1f}% use AI<br>" + ND + " = %{customdata:,}<extra></extra>"))
                fb = EXP_FULL.set_index("level")["adopt"]
                fig.add_trace(go.Scatter(x=list(xe["level"]), y=[float(fb[l]) for l in xe["level"]], mode="markers", name="Full survey",
                                         marker=dict(size=14, symbol="diamond", color=C["sage"], line=dict(color=C["ink"], width=2)),
                                         hovertemplate="%{x}<br>Full survey: %{y:.1f}%<extra></extra>"))
                fig.update_yaxes(range=[0, 105], ticksuffix="%")
                fig.update_xaxes(categoryorder="array", categoryarray=EXP_ORDER)
                fig.update_layout(legend=dict(orientation="h", y=1.15, font=dict(color=C["ink"])))
                show(style(fig, 280, legend=True, margin=dict(l=10, r=10, t=40, b=10)))
                html(f'<div class="note">Levels with fewer than {MIN_N} valid answers are hidden.</div>')
            else:
                html(f'<div class="note">Fewer than two experience levels have {MIN_N}+ valid answers in this group.</div>')
    # dynamic insight
    ad = byk.get("adopt")
    parts_i = [f"<b>{xg_label}: {xg_n:,} respondents ({xg_n / N_ALL * 100:.1f}% of the survey).</b>"]
    if ad:
        d_ = ad["p"] - ad["base"]
        parts_i.append(f"{ad['p']:.1f}% use AI ({nlab(ad['n'])}), {abs(d_):.1f} points {'above' if d_ >= 0 else 'below'} the full survey ({ad['base']:.1f}%)"
                       + (", a difference larger than the margin of error." if ad["clear"] else ", within the margin of error of the full-survey rate."))
    others = [r for r in rows_x if r["key"] != "adopt" and r["clear"]]
    if others:
        r = max(others, key=lambda r_: abs(r_["p"] - r_["base"]))
        parts_i.append(f"The clearest difference is “{r['label'].lower()}”: {r['p']:.1f}% vs {r['base']:.1f}% in the full survey ({nlab(r['n'])}).")
    elif rows_x:
        parts_i.append("No other measure differs clearly from the full survey.")
    if ch_n >= MIN_N:
        top_c = ch_p.idxmax()
        parts_i.append(f"The most reported challenge here is “{ch_name(top_c)}” ({ch_p.max():.1f}%)"
                       + (", the same as in the full survey." if top_c == CH_PCT_FULL.idxmax() else f", whereas the full survey's top challenge is “{ch_name(CH_PCT_FULL.idxmax())}”."))
    insight(" ".join(parts_i),
            "Groups can differ in AI use and trust, but these are survey associations, not causes. Gaps inside the 95% interval should not be over-read, and small groups have wide intervals.")


# ----------------------------------------------------------------------------
# 02 - From raw data to real insights (before & after)
# ----------------------------------------------------------------------------
section("02", "From raw data to real insights", "prep",
        "How Did We Prepare the Data for Analysis?",
        "Everything below is the full dataset (filters do not apply here). Items marked “verified” are recomputed live from cleaned_data.csv. Items marked “documented” come from the printed outputs of the EDA notebook, because the raw 152 MB survey file is not part of this app and cannot be re-checked here.")

B_OK = f'<span class="badge ok">verified live</span>'
B_DOC = f'<span class="badge doc">documented in notebook</span>'
miss_after = {c: int(df[c].isnull().sum()) for c in SELECTED_COLS}
MISS_MATCH = all(miss_after[c] == v for c, v in RAW_MISSING.items())
DUP_SEL = int(df[SELECTED_COLS].duplicated().sum())
n_lt1 = int((df["YearsCodePro"] == "Less than 1 year").sum())
n_gt50 = int((df["YearsCodePro"] == "More than 50 years").sum())
yrs = df["YearsCodePro_clean"].dropna()
exp_counts = df["ExperienceLevel"].value_counts(dropna=False)
sal_all = df["ConvertedCompYearly"].dropna()
n_sal_out = int(((sal_all < SAL_LOWER) | (sal_all > SAL_UPPER)).sum())
n_sal_clean = int(len(sal_clean(df)))
miss_lo = min(RAW_MISSING.values()) / RAW_ROWS * 100
miss_hi = max(RAW_MISSING.values()) / RAW_ROWS * 100

html(f"""
<div class="ba">
<div class="side before">
<div class="lbl">Before: raw survey file {B_DOC}</div>
<div class="big">{RAW_ROWS:,} × {RAW_COLS}</div>
<div class="line">
rows × columns<br>
Column types: {RAW_DTYPES['object']} text, {RAW_DTYPES['float64']} decimal, {RAW_DTYPES['int64']} integer<br>
{RAW_DUPES} duplicate rows (checked on all {RAW_COLS} columns)<br>
Missing values in the 10 columns we kept: {miss_lo:.1f}% to {miss_hi:.1f}%<br>
Years of coding stored as text (e.g. “Less than 1 year”)
</div>
</div>
<div class="arrow">→</div>
<div class="side after">
<div class="lbl">After: cleaned dataset {B_OK}</div>
<div class="stat big">{N_ALL:,} × {N_COLS}</div>
<div class="line">
rows × columns<br>
{len(SELECTED_COLS)} original columns kept + 2 new variables<br>
No rows removed ({"✓" if N_ALL == RAW_ROWS else "≠"} same as raw)<br>
Missing values kept, not filled in ({'counts identical to raw ✓' if MISS_MATCH else 'differs from notebook'})<br>
Years of coding converted to numbers + experience levels
</div>
</div>
</div>
""")

st.markdown("&nbsp;")
with st.container():
    st.markdown(f"**Columns: raw → kept → final**")
    steps_ = [("Raw survey", RAW_COLS, C["sage"]), ("Selected for this project", len(SELECTED_COLS), C["forest"]),
              ("Final cleaned (+2 new)", N_COLS, C["lime"])]
    fig = go.Figure(go.Bar(y=[s[0] for s in steps_], x=[s[1] for s in steps_], orientation="h",
                           marker=dict(color=[s[2] for s in steps_], line=dict(color=C["ink"], width=1.5)),
                           text=[f"<b>{s[1]}</b>" for s in steps_], textposition="outside", textfont=dict(color=C["ink"], size=15),
                           hovertemplate="%{y}: %{x}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(range=[0, 135], visible=False)
    show(style(fig, 240, margin=dict(l=10, r=30, t=10, b=10)))
    html(f'<div class="note">The notebook keeps only the columns needed for AI adoption, trust, benefits, challenges, role, experience, country and salary. It does not list a separate reason for each dropped column.</div>')

steps_html = [
    ("1 · Choose the columns",
     f"Kept <b>{len(SELECTED_COLS)} of {RAW_COLS}</b> columns: AISelect, AISent, DevType, YearsCodePro, Country, AIThreat, ConvertedCompYearly, AIAcc, AIBen, AIChallenges. Note: AISent was kept but is not analysed in the notebook.", B_DOC),
    ("2 · Missing values",
     "<b>Kept, not deleted or filled.</b> The notebook decided to handle missing answers separately in each analysis, so each chart uses only the people who answered that question. Many questions are simply not shown to everyone (for example, only AI users see the trust, benefit and challenge questions).", B_OK),
    ("3 · Duplicates",
     f"The raw survey has <b>{RAW_DUPES}</b> duplicate rows. On just the 10 selected columns, <b>{DUP_SEL:,}</b> rows look identical, but they are different respondents (they differ in columns we did not keep), so they were <b>kept</b>.", B_OK),
    ("4 · Fix a data type",
     f"<b>YearsCodePro</b> was text. “Less than 1 year” ({n_lt1:,} rows) became 0.5 and “More than 50 years” ({n_gt50:,} rows) became 51; everything else was converted to a number. Result: <b>YearsCodePro_clean</b> with {len(yrs):,} valid values (mean {yrs.mean():.2f}, median {yrs.median():.0f}).", B_OK),
    ("5 · New variable: experience level",
     f"<b>ExperienceLevel</b> groups years into Beginner (&lt;3), Intermediate (3 to &lt;10), Experienced (10 to &lt;20) and Veteran (20+). Counts: " + ", ".join(f"{l} {int(exp_counts.get(l, 0)):,}" for l in EXP_ORDER) + f"; missing {int(df['ExperienceLevel'].isna().sum()):,}.", B_OK),
    ("6 · Salary outliers",
     f"Using the IQR rule on yearly salary (Q1 {SAL_Q1:,.0f}, Q3 {SAL_Q3:,.0f}), the upper limit is <b>{SAL_UPPER:,.2f}</b>. <b>{n_sal_out:,}</b> of {len(sal_all):,} salaries are above it. They are excluded <b>only from salary analyses</b>, leaving <b>{n_sal_clean:,}</b> salary records; every other analysis keeps all rows.", B_OK),
]
html('<div class="steps">' + "".join(f'<div class="card step"><div class="h">{h}</div><div class="b">{b}</div><div style="margin-top:10px">{bd}</div></div>' for h, b, bd in steps_html) + '</div>')

# salary outlier chart
SAL_MED_ALL = float(sal_clean(df)['ConvertedCompYearly'].median())
SAL_CAP = 400000
vals = sal_all[(sal_all >= 0) & (sal_all <= SAL_CAP)].values
hist, edges = np.histogram(vals, bins=80, range=(0, SAL_CAP))
mids = (edges[:-1] + edges[1:]) / 2
fig = go.Figure(go.Bar(x=mids, y=hist, marker=dict(color=[C["forest"] if e <= SAL_UPPER else C["warn"] for e in edges[1:]]),
                       hovertemplate="Salary around" + " %{x:,.0f} USD<br>" + "respondents" + ": %{y:,}<extra></extra>"))
fig.add_vline(x=SAL_UPPER, line_dash="dash", line_color=C["lime"], line_width=2)
fig.add_annotation(x=SAL_UPPER, y=max(hist), xanchor="left", yanchor="top", showarrow=False, font=dict(size=12, color=C["ink"]),
                   text=f"<b>Upper limit {SAL_UPPER:,.0f}</b><br>{n_sal_out:,} salaries above it are<br>excluded from salary analyses")
fig.update_xaxes(title="Yearly salary (USD), shown up to 400,000", tickformat=",")
fig.update_yaxes(title="Respondents")
st.markdown(f"**Univariate view: yearly salary and the outlier rule**")
show(style(fig, 320))
html(f'<div class="note">{int((sal_all > SAL_CAP).sum()):,} salaries above 400,000 are not drawn but are counted among the {n_sal_out:,} outliers. Median salary of the cleaned salary records: {SAL_MED_ALL:,.0f} USD.</div>')


# ----------------------------------------------------------------------------
# 03 - AI adoption (giant number + ribbon)
# ----------------------------------------------------------------------------
section("03", "AI adoption", "d1", "How Widely Is AI Used Among Developers?",
        "The survey asked whether developers use AI tools in their development process. Missing answers are excluded from the percentages.")
nF1, cF1 = adopt_counts(fdf)
if nF1 < MIN_N:
    low_data(nF1)
else:
    pU, pS, pN = [cF1[k] / nF1 * 100 for k in ["Uses AI", "Plans to soon", "Doesn't plan to"]]
    g1, g2 = st.columns([1, 1.7])
    with g1:
        html(f'<div class="giant">{pU:.0f}<small>%</small></div>'
             f'<div class="giant-cap">of developers use AI tools. {nlab(nF1)} '
             f'({N_F - nF1:,} without an answer are excluded).</div>')
    with g2:
        segs = [("Uses AI", pU, C["forest"], C["on_forest"]), ("Plans to soon", pS, C["lime"], C["limeink"]),
                ("Doesn't plan to", pN, C["sage"], C["on_sage"])]
        fig = go.Figure()
        for key, v, col, tcol in segs:
            nm = ai_status_name(key)
            fig.add_trace(go.Bar(y=[""], x=[v], orientation="h", name=nm, marker=dict(color=col),
                                 text=[f"<b>{v:.1f}%</b><br>{nm}"], textposition="inside", insidetextanchor="middle",
                                 textfont=dict(color=tcol, size=14),
                                 hovertemplate=f"{nm}<br>{v:.1f}%<br>{nlab(cF1[key])}<extra></extra>"))
        fig.update_layout(barmode="stack", bargap=0.05)
        fig.update_xaxes(visible=False, range=[0, 100])
        fig.update_yaxes(visible=False)
        show(style(fig, 150, margin=dict(l=0, r=0, t=10, b=10)))
        html(f'<div class="note">Each segment is a share of respondents who answered. Hover for counts.</div>')
    vs = f" In the full survey it is {ADOPT:.1f}%." if FILTERS_ON else ""
    insight(f"<b>{pU:.0f}% of this audience use AI</b> ({nlab(nF1)}), a further {pS:.0f}% plan to start and {pN:.0f}% have no plans to.{vs}",
            "AI use is no longer a niche habit among survey respondents. The real question is what developers think of the tools they use.")

# ----------------------------------------------------------------------------
# 04 - Experience (lollipop + gap callout)
# ----------------------------------------------------------------------------
section("04", "Developer experience", "d2", "Does AI Adoption Differ by Experience Level?",
        "Experience level comes from years of professional coding: Beginner (<3), Intermediate (3 to <10), Experienced (10 to <20), Veteran (20+). Levels with fewer than 30 valid answers are hidden.")
ex = exp_table(fdf, MIN_N)
if len(ex) < 2:
    low_data(int(ex["n"].sum()) if len(ex) else 0)
    html(f'<div class="note">At least two experience levels with 30+ answers are needed to compare them.</div>')
else:
    hi_e, lo_e = ex.loc[ex["adopt"].idxmax()], ex.loc[ex["adopt"].idxmin()]
    gap = hi_e["adopt"] - lo_e["adopt"]
    e1, e2 = st.columns([1, 2.2])
    with e1:
        html(f'<div class="giant" style="font-size:6rem">{gap:.0f}<small>pts</small></div>'
             f'<div class="giant-cap">separate {hi_e.level} developers ({pct(hi_e.adopt)}) from {lo_e.level} developers ({pct(lo_e.adopt)}) in AI use.</div>')
    with e2:
        xs = [l for l in ex["level"]]
        ys = ex["adopt"].tolist()
        fig = go.Figure()
        for x, y in zip(xs, ys):  # stems from zero keep the scale honest
            fig.add_trace(go.Scatter(x=[x, x], y=[0, y], mode="lines", line=dict(color=C["forest"], width=4), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=C["muted"], width=1.5, dash="dot"), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers+text", showlegend=False,
            marker=dict(size=[16 + 14 * (n / ex["n"].max()) for n in ex["n"]],
                        color=[C["lime"] if v == max(ys) else C["forest"] for v in ys], line=dict(color=C["ink"], width=2)),
            text=[f"<b>{v:.1f}%</b>" for v in ys], textposition="top center", textfont=dict(color=C["ink"], size=15),
            customdata=list(zip(ex["n"], ex["soon"], ex["nope"])),
            hovertemplate="%{x}<br>" + "Use AI" + ": %{y:.1f}%<br>" + "Plan to soon" + ": %{customdata[1]:.1f}%<br>"
                          + "Don't plan to" + ": %{customdata[2]:.1f}%<br>" + ND + " = %{customdata[0]:,}<extra></extra>"))
        if len(xs) >= 2:
            fig.add_annotation(x=(len(xs) - 1) / 2, y=min(95, max(ys) + 14), showarrow=False, bgcolor=C["lime"], bordercolor=C["lime"], borderpad=6,
                               font=dict(size=13, color=C["limeink"]),
                               text=f"<b>{gap:.1f} pts</b> between {hi_e.level} and {lo_e.level}")
        fig.update_yaxes(range=[0, 100], ticksuffix="%", title="Share using AI")
        fig.update_xaxes(categoryorder="array", categoryarray=xs, tickvals=xs,
                         ticktext=[f"{r.level}<br><span style='font-size:11px'>{nlab(r.n)}</span>" for r in ex.itertuples()])
        show(style(fig, 400))
    trend = ex["adopt"].is_monotonic_decreasing
    insight(f"<b>{hi_e.level} developers use AI the most ({pct(hi_e.adopt)}); {lo_e.level} the least ({pct(lo_e.adopt)}).</b>" + (" Adoption falls at every step up in experience." if trend and len(ex) > 2 else ""),
            "This is an association in survey data, not proof that experience causes less AI use: job type, age and company policy could all play a part. Dot size shows group size.")
    html(audience_note(int(ex["n"].sum())))
    with st.expander("See all three answers by experience level (full crosstab from the EDA)"):
        fig = go.Figure()
        for col_, key, colr, tcol in [("adopt", "Uses AI", C["forest"], C["on_forest"]), ("soon", "Plans to soon", C["lime"], C["limeink"]), ("nope", "Doesn't plan to", C["sage"], C["on_sage"])]:
            fig.add_trace(go.Bar(y=[l for l in ex["level"]], x=ex[col_], orientation="h", name=ai_status_name(key), marker=dict(color=colr),
                                 text=[f"{v:.1f}%" for v in ex[col_]], textposition="inside", textfont=dict(color=tcol),
                                 hovertemplate="%{y}<br>" + ai_status_name(key) + ": %{x:.1f}%<extra></extra>"))
        fig.update_layout(barmode="stack", legend=dict(orientation="h", y=1.18, font=dict(color=C["ink"])))
        fig.update_xaxes(range=[0, 100], ticksuffix="%")
        fig.update_yaxes(autorange="reversed")
        show(style(fig, 280, legend=True, margin=dict(l=10, r=10, t=40, b=10)))

# ----------------------------------------------------------------------------
# 05 - Roles (dot plot with 95% intervals)
# ----------------------------------------------------------------------------
section("05", "Developer roles", "d3", "Which Developer Roles Use AI the Most?",
        "The ten most common roles in the survey (as defined in the EDA, including Student and Other), compared by adoption rate rather than head-count. Dot size shows sample size; whiskers are 95% confidence intervals; roles need 30+ valid answers.")
rt = role_table(fdf, TOP_ROLES, MIN_N)
if len(rt) < 2:
    low_data(int(rt["n"].sum()) if len(rt) else 0)
    html(f'<div class="note">At least two of the top-10 roles with 30+ answers are needed. Remove the role filter to compare roles.</div>')
else:
    base_rate, base_n = adoption(fdf)
    rt["label"] = [r for r in rt["role"]]
    nmax = rt["n"].max()
    fig = go.Figure()
    for r in rt.itertuples():
        fig.add_trace(go.Scatter(x=[base_rate, r.adopt], y=[r.label, r.label], mode="lines",
                                 line=dict(color=C["lime"] if r.adopt >= base_rate else C["warn"], width=3), hoverinfo="skip", showlegend=False, opacity=.6))
    fig.add_trace(go.Scatter(
        x=rt["adopt"], y=rt["label"], mode="markers+text", showlegend=False,
        marker=dict(size=[12 + 20 * sqrt(n / nmax) for n in rt["n"]], color=[C["lime"] if a >= base_rate else C["sage"] for a in rt["adopt"]], line=dict(color=C["ink"], width=2)),
        error_x=dict(type="data", symmetric=False, array=(rt["hi"] - rt["adopt"]).tolist(), arrayminus=(rt["adopt"] - rt["lo"]).tolist(), color=C["muted"], thickness=1.4, width=4),
        text=[f"<b>{a:.1f}%</b>" for a in rt["adopt"]], textposition="top center", textfont=dict(color=C["ink"], size=13),
        customdata=list(zip(rt["n"], rt["lo"], rt["hi"])),
        hovertemplate="<b>%{y}</b><br>%{x:.1f}% " + "use AI" + "<br>" + "95% CI" + " %{customdata[1]:.1f}% – %{customdata[2]:.1f}%<br>"
                      + ND + " = %{customdata[0]:,}<extra></extra>"))
    fig.add_vline(x=base_rate, line_dash="dash", line_color=C["muted"], line_width=1.5)
    fig.add_annotation(x=base_rate, y=1.02, yref="paper", showarrow=False, yanchor="bottom", font=dict(size=12, color=C["muted"]),
                       text=f"{'Selected audience' if FILTERS_ON else 'All respondents'} {base_rate:.1f}%")
    rtop, rbot = rt.iloc[-1], rt.iloc[0]
    fig.add_annotation(x=rtop.adopt, y=rtop.label, ax=40, ay=-46, showarrow=True, arrowcolor=C["forest"], xanchor="left", font=dict(size=12, color=C["ink"]),
                       text=f"<b>Highest</b><br>{rtop.label}")
    fig.add_annotation(x=rbot.adopt, y=rbot.label, ax=-40, ay=-46, showarrow=True, arrowcolor=C["warn"], xanchor="right", font=dict(size=12, color=C["ink"]),
                       text=f"<b>Lowest</b><br>{rbot.label}")
    x_lo, x_hi = max(0, float(rt["lo"].min()) - 6), min(100, float(rt["hi"].max()) + 8)
    fig.update_xaxes(range=[x_lo, x_hi], ticksuffix="%", title="Share using AI" + (" (axis does not start at zero; read each dot against the dashed line)" if x_lo > 0 else ""))
    fig.update_yaxes(categoryorder="array", categoryarray=rt["label"].tolist(), tickvals=rt["label"].tolist(),
                     ticktext=[f"{r.label}  <span style='font-size:11px'>{ND}={r.n:,}</span>" for r in rt.itertuples()])
    show(style(fig, max(380, 60 * len(rt) + 80), margin=dict(l=10, r=30, t=40, b=60)))
    r1, r2 = st.columns([1.3, 1])
    with r1:
        insight(f"<b>{rtop.label} lead ({pct(rtop.adopt)}); {rbot.label} trail ({pct(rbot.adopt)}).</b> That is a {rtop.adopt - rbot.adopt:.0f}-point spread across {len(rt)} roles.",
                "AI is not adopted evenly. This describes who uses AI, not why. Some roles may simply have more suitable tools or more permission to use them.")
    with r2:
        html(f'<div class="note"><b>Reading the chart:</b> lines run from the audience-wide rate (dashed) to each role, so line length shows the gap. Wide whiskers mean a smaller, less certain group.</div>')

# ----------------------------------------------------------------------------
# 06 - Benefits
# ----------------------------------------------------------------------------
section("06", "AI benefits", "ben", "What Benefits Do Developers Report From AI?",
        "A multi-select question shown to AI users: each person could tick several benefits. Percentages use everyone who answered the question as the denominator, so they add up to more than 100%. These are reported or expected benefits, not measured improvements.")
bc, bp, bn = multi_stats(fdf, "AIBen")
if bn < MIN_N:
    low_data(bn)
else:
    order_b = bp.sort_values()
    colors_b = [C["lime"] if k == PRODUCTIVITY_KEY else (C["forest"] if k == ACCURACY_KEY else C["sage"]) for k in order_b.index]
    pP, pA = float(bp.get(PRODUCTIVITY_KEY, 0)), float(bp.get(ACCURACY_KEY, 0))
    _, b1, _ = st.columns([0.35, 5, 0.35])
    with b1:
        lab_b = [k for k in order_b.index]
        fig = go.Figure()
        for k, l in zip(order_b.index, lab_b):
            fig.add_trace(go.Scatter(x=[0, order_b[k]], y=[l, l], mode="lines", line=dict(color=C["forest"], width=3), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=order_b.values, y=lab_b, mode="markers+text", showlegend=False,
                                 marker=dict(size=22, color=colors_b, line=dict(color=C["ink"], width=2)),
                                 text=[f"<b>{v:.1f}%</b>" for v in order_b.values], textposition="middle right", textfont=dict(color=C["ink"], size=14),
                                 customdata=[int(bc[k]) for k in order_b.index],
                                 hovertemplate="<b>%{y}</b><br>%{x:.1f}%<br>%{customdata:,} " + "respondents" + f" ({ND} = {bn:,})<extra></extra>"))
        fig.update_xaxes(range=[0, 100], ticksuffix="%")
        show(style(fig, 470, margin=dict(l=10, r=60, t=10, b=10)))
    top_b = order_b.index[-1]
    insight(f"<b>{top_b} is the most reported benefit ({pct(float(order_b.iloc[-1]))}, {nlab(bn)}).</b> ",
            "These are self-reported or expected benefits, so they show what developers believe AI does for them, not measured improvements in their work.")
    html(audience_note(bn))

# ----------------------------------------------------------------------------
# 07 - Challenges
# ----------------------------------------------------------------------------
section("07", "AI challenges", "chal", "What Challenges Do Developers Face When Using AI?",
        "Also a multi-select question for AI users, ranked by the share of respondents who ticked each challenge. This is a different question from the 5-point accuracy rating that follows later.")
cc, cp, cn = multi_stats(fdf, "AIChallenges", ch_key)
if cn < MIN_N:
    low_data(cn)
else:
    cps = cp.sort_values(ascending=False)
    labs = [f"{i + 1}. {ch_name(k)}" for i, k in enumerate(cps.index)]
    fig = go.Figure()
    for l, v in zip(labs, cps.values):
        fig.add_trace(go.Scatter(x=[0, v], y=[l, l], mode="lines", line=dict(color=C["forest"], width=3), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=cps.values, y=labs, mode="markers+text", showlegend=False,
                             marker=dict(size=[26 if i == 0 else 18 for i in range(len(cps))], color=[C["lime"] if i == 0 else C["sage"] for i in range(len(cps))], line=dict(color=C["ink"], width=2)),
                             text=[f"<b>{v:.1f}%</b>" for v in cps.values], textposition="middle right", textfont=dict(color=C["ink"], size=13),
                             customdata=[int(cc[k]) for k in cps.index],
                             hovertemplate="<b>%{y}</b><br>%{x:.1f}%<br>%{customdata:,} " + "respondents" + f" ({ND} = {cn:,})<extra></extra>"))
    fig.update_xaxes(range=[0, 100], ticksuffix="%")
    fig.update_yaxes(autorange="reversed")
    show(style(fig, 420, margin=dict(l=10, r=50, t=10, b=10)))
    k1_, k2_ = cps.index[0], (cps.index[1] if len(cps) > 1 else None)
    insight(f"<b>The most reported challenge is “{ch_name(k1_)}” ({pct(float(cps.iloc[0]))}, {nlab(cn)})</b>" + (f", followed by “{ch_name(k2_)}” ({pct(float(cps.iloc[1]))})." if k2_ else "."),
            "The top challenges are about trust and context, not about access or cost. Tick-box challenges are not the same as the accuracy rating in the next sections: here a person simply named “don't trust the output” as one of their problems.")
    html(audience_note(cn))


# ----------------------------------------------------------------------------
# 08 - Productivity & trust (visual centrepiece: area-proportional Euler diagram)
# ----------------------------------------------------------------------------
section("08", "Productivity & trust", "d4", "Can Developers Benefit From AI While Distrusting Its Output?")


def lens_area(r1_: float, r2_: float, d: float) -> float:
    if d >= r1_ + r2_:
        return 0.0
    if d <= abs(r1_ - r2_):
        return pi * min(r1_, r2_) ** 2
    a = r1_ ** 2 * acos((d * d + r1_ ** 2 - r2_ ** 2) / (2 * d * r1_))
    b = r2_ ** 2 * acos((d * d + r2_ ** 2 - r1_ ** 2) / (2 * d * r2_))
    c = 0.5 * sqrt((-d + r1_ + r2_) * (d + r1_ - r2_) * (d - r1_ + r2_) * (d + r1_ + r2_))
    return a + b - c


def euler_layout(p1: float, p2: float, both_: float):
    """Area-proportional two-circle diagram: circle areas = p1, p2 (percent); overlap area = both_."""
    r1_, r2_ = sqrt(p1 / pi), sqrt(p2 / pi)
    lo_, hi_ = abs(r1_ - r2_), r1_ + r2_
    for _ in range(80):
        mid = (lo_ + hi_) / 2
        if lens_area(r1_, r2_, mid) > both_:
            lo_ = mid
        else:
            hi_ = mid
    return r1_, r2_, (lo_ + hi_) / 2


PD = DARK  # the centrepiece panel is always dark, in both themes
PXF = paradox_stats(fdf)
PX_n = PXF["n"]
PX_both = PXF["both"]
PX_p_only = PXF["p_only"]
PX_d_only = PXF["d_only"]
PX_neither = PXF["neither"]
PX_n_prod = PXF["n_prod"]
PX_n_dist = PXF["n_dist"]
PX_p_both = PXF["p_both"]
PX_p_prod = PXF["p_prod"]
PX_p_dist = PXF["p_dist"]
PX_p_dist_given_prod = PXF["p_dist_given_prod"]
box = container("centerpiece")
with box:
    html(f'<div class="pc-tag">The heart of the project</div>'
         f'<div class="pc-title">Faster, <em>and</em> not fully believed.</div>')
    if min(PX_n, PX_n_prod, PX_n_dist) < MIN_N:
        low_data(PX_n)
    else:
        html(f'<div class="pc-lede">Among {PX_n:,} AI users who answered both follow-up questions, we look at two things they could tick: “Increase productivity” (a benefit) and “Don’t trust the output or answers” (a challenge). The two circles are drawn to scale, so the overlap is the paradox.</div>')
        pc1, pc2 = st.columns([1.55, 1])
        with pc1:
            R1, R2, D = euler_layout(PX_p_prod, PX_p_dist, PX_p_both)
            x2 = D
            fig = go.Figure()
            fig.add_shape(type="circle", xref="x", yref="y", x0=-R1, x1=R1, y0=-R1, y1=R1, fillcolor=rgba(PD["lime"], .58), line=dict(color=PD["lime"], width=2))
            fig.add_shape(type="circle", xref="x", yref="y", x0=x2 - R2, x1=x2 + R2, y0=-R2, y1=R2, fillcolor=rgba(PD["warn"], .58), line=dict(color=PD["warn"], width=2))
            c_left, c_mid, c_right = (-R1 + (x2 - R2)) / 2, ((x2 - R2) + R1) / 2, (R1 + (x2 + R2)) / 2
            regions = [(c_left, PX_p_only, "Productivity<br>only", "#0C1B13", 19, 10),
                       (c_mid, PX_both, "feel faster<br>and distrust", "#FFFFFF", 34, 11),
                       (c_right, PX_d_only, "Distrust<br>only", "#0C1B13", 19, 10)]
            for cx, cnt, name, tc, big, small in regions:
                share = cnt / PX_n * 100
                # one centred block per region (number above its label) so the pair always stays aligned and inside the region
                fig.add_annotation(x=cx, y=0, text=f"<span style='font-size:{big}px;font-family:Fraunces, Georgia, serif'><b>{share:.1f}%</b></span><br>"
                                                   f"<span style='font-size:{small}px'>{name}</span>",
                                   showarrow=False, align="center", xanchor="center", yanchor="middle", font=dict(size=small, color=tc))
                fig.add_trace(go.Scatter(x=[cx], y=[0], mode="markers", marker=dict(size=70, opacity=0), showlegend=False,
                                         hovertemplate=f"<b>{name.replace('<br>', ' ')}</b><br>{share:.1f}% ({nlab(cnt)})<extra></extra>"))
            fig.add_annotation(x=-R1 * 0.55, y=R1 * 1.12, showarrow=False, font=dict(size=12, color=PD["lime"]), xanchor="center", yanchor="bottom",
                               text=f"<b>Report a productivity benefit</b><br>{PX_p_prod:.1f}% ({nlab(PX_n_prod)})")
            fig.add_annotation(x=x2 + R2 * 0.5, y=R2 * 1.12, showarrow=False, font=dict(size=12, color=PD["warn"]), xanchor="center", yanchor="bottom",
                               text=f"<b>Don't trust the output</b><br>{PX_p_dist:.1f}% ({nlab(PX_n_dist)})")
            fig.add_annotation(x=(-R1 + x2 + R2) / 2, y=-R1 * 1.2, showarrow=False, font=dict(size=11, color=PD["muted"]), yanchor="top",
                               text=f"Neither: {PX_neither / PX_n * 100:.1f}% ({nlab(PX_neither)})")
            fig.update_xaxes(visible=False, range=[-R1 * 1.1, x2 + R2 * 1.1])
            fig.update_yaxes(visible=False, range=[-R1 * 1.35, R1 * 1.55], scaleanchor="x", scaleratio=1)
            show(style(fig, 480, P=PD, margin=dict(l=0, r=0, t=0, b=0)))
        with pc2:
            html(f'<div class="pc-big">{PX_p_both:.0f}<small>%</small></div>'
                 f'<div class="pc-cap"><b>{PX_both:,} of {PX_n:,}</b> AI users say AI boosts their productivity <b>and</b> that they have trouble trusting what it tells them.</div>')
            html(f'<div class="pc-card"><div class="t">Among the {PX_n_prod:,} who report a productivity benefit</div>'
                 f'<div class="b">{PX_p_dist_given_prod:.1f}% also name distrust of AI output as a challenge.</div></div>')
            fig = go.Figure()
            fig.add_trace(go.Bar(y=[""], x=[PX_p_dist_given_prod], orientation="h", marker=dict(color=PD["warn"]),
                                 text=[f"<b>{PX_p_dist_given_prod:.1f}%</b> also distrust"], textposition="inside", insidetextanchor="middle",
                                 textfont=dict(color="#0C1B13", size=13), hovertemplate=f"{PX_both:,} / {PX_n_prod:,}<extra></extra>"))
            fig.add_trace(go.Bar(y=[""], x=[100 - PX_p_dist_given_prod], orientation="h", marker=dict(color=PD["lime"]),
                                 text=[f"{100 - PX_p_dist_given_prod:.1f}% don't"], textposition="inside", insidetextanchor="middle",
                                 textfont=dict(color="#0C1B13", size=13), hovertemplate=f"{PX_p_only:,} / {PX_n_prod:,}<extra></extra>"))
            fig.update_layout(barmode="stack", bargap=0.1)
            fig.update_xaxes(visible=False, range=[0, 100])
            fig.update_yaxes(visible=False)
            show(style(fig, 90, P=PD, margin=dict(l=0, r=0, t=6, b=0)))
        B_P = multi_stats(fdf, "AIBen")[1].get(PRODUCTIVITY_KEY, float("nan"))
        B_D = multi_stats(fdf, "AIChallenges", ch_key)[1].get("trust", float("nan"))
        html(f'<div class="pc-note"><b>Population:</b> AI users who answered both the benefits and the challenges questions (both multi-select, shown only to AI users), {nlab(PX_n)}. Productivity = “{PRODUCTIVITY_KEY}” selected; distrust = “Don’t trust the output or answers” selected. <b>Method:</b> the EDA analysed the two questions separately (productivity {B_P:.1f}% of benefit respondents; distrust {B_D:.1f}% of challenge respondents). The overlap is a new calculation with the same definitions; it describes how answers overlap, not cause and effect.</div>')
if min(PX_n, PX_n_prod, PX_n_dist) >= MIN_N:
    insight(f"<b>{PX_p_both:.0f}% of this group feel faster and doubt the answers at the same time.</b> Among those who report productivity gains, {PX_p_dist_given_prod:.0f}% also have trouble trusting what AI says ({nlab(PX_n_prod)}).",
            "Developers are not blindly trusting AI. Many seem to treat it as a fast first draft that they still check. Adoption and productivity do not equal confidence.")

# ----------------------------------------------------------------------------
# 09 - AI accuracy & trust (diverging Likert bars)
# ----------------------------------------------------------------------------
accF, nAccF = trust_dist(fdf)
section("09", "AI accuracy & trust", "d5", "How Much Do Developers Trust AI Accuracy?",
        f"A 5-point rating asked only of AI users, so it describes AI users only (the EDA's AI-adoption × trust crosstab has a single row for this reason). Each bar splits the group into those who distrust (left), are neutral (centre) and trust (right). Base: {nAccF:,} answers.")
if nAccF < MIN_N:
    low_data(nAccF)
else:
    rows = [("All AI users", accF, nAccF)]
    for lvl in EXP_ORDER:
        d_, n_ = trust_dist(fdf[fdf["ExperienceLevel"] == lvl])
        if n_ >= MIN_N:
            rows.append((lvl, d_, n_))
    show(diverging_trust(rows))
    pT, pD_, pH = float(accF[["Highly trust", "Somewhat trust"]].sum()), float(accF[["Somewhat distrust", "Highly distrust"]].sum()), float(accF["Highly trust"])
    t_exp = [(nm, float(d_[["Highly trust", "Somewhat trust"]].sum())) for nm, d_, n_ in rows[1:]]
    exp_line_en = ""
    if len(t_exp) >= 2:
        t_hi, t_lo = max(t_exp, key=lambda t: t[1]), min(t_exp, key=lambda t: t[1])
        if t_hi[1] - t_lo[1] < 5:
            exp_line_en = f" Trust barely changes with experience ({t_lo[1]:.0f}% to {t_hi[1]:.0f}% trust across levels)."
        else:
            exp_line_en = f" Trust ranges from {t_lo[1]:.0f}% ({t_lo[0]}) to {t_hi[1]:.0f}% ({t_hi[0]}) across experience levels."
    insight(f"<b>{pH:.1f}% of AI users highly trust AI accuracy, while {pD_:.0f}% distrust it to some degree and {pT:.0f}% trust it somewhat or highly ({nlab(nAccF)}).</b>{exp_line_en}",
            "Even among people who use AI, confidence is cautious. Most land in the middle ground of “somewhat trust” or “neither”.")

# ----------------------------------------------------------------------------
# 10 - Cybersecurity trust gap
# ----------------------------------------------------------------------------
section("10", "Cybersecurity trust gap", "cyber", "Which Developer Roles Distrust AI Accuracy the Most?",
        "Distrust = “Somewhat distrust” plus “Highly distrust” among AI users who rated AI accuracy, by role. Only roles with 30+ valid answers are compared. Security professionals are highlighted in lime.")
dt = distrust_table(fdf, MIN_N)
sec_n_all = int(fdf.dropna(subset=["AIAcc"])["DevType"].eq(SECURITY_ROLE).sum())
if len(dt) < 2:
    low_data(int(dt["n"].sum()) if len(dt) else 0)
    html(f'<div class="note">At least two roles with 30+ valid answers are needed.</div>')
else:
    show_df = dt.head(10)
    if SECURITY_ROLE in dt["role"].values and SECURITY_ROLE not in show_df["role"].values:
        show_df = pd.concat([show_df, dt[dt["role"] == SECURITY_ROLE]])
    show_df = show_df.sort_values("dist").reset_index(drop=True)
    show_df["label"] = [r for r in show_df["role"]]
    base_d = float(accF[["Somewhat distrust", "Highly distrust"]].sum())
    is_sec = [r == SECURITY_ROLE for r in show_df["role"]]
    fig = go.Figure()
    for r in show_df.itertuples():
        fig.add_trace(go.Scatter(x=[0, r.dist], y=[r.label, r.label], mode="lines", line=dict(color=C["lime"] if r.role == SECURITY_ROLE else C["forest"], width=5 if r.role == SECURITY_ROLE else 3),
                                 hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(
        x=show_df["dist"], y=show_df["label"], mode="markers+text", showlegend=False,
        marker=dict(size=[28 if s else 18 for s in is_sec], color=[C["lime"] if s else C["sage"] for s in is_sec], line=dict(color=C["ink"], width=2)),
        error_x=dict(type="data", symmetric=False, array=(show_df["hi"] - show_df["dist"]).tolist(), arrayminus=(show_df["dist"] - show_df["lo"]).tolist(), color=C["muted"], thickness=1.3, width=4),
        text=[f"<b>{v:.1f}%</b>" for v in show_df["dist"]], textposition="top center", textfont=dict(color=C["ink"], size=13),
        customdata=list(zip(show_df["n"], show_df["lo"], show_df["hi"])),
        hovertemplate="<b>%{y}</b><br>" + "Distrust" + ": %{x:.1f}%<br>" + "95% CI" + " %{customdata[1]:.1f}% – %{customdata[2]:.1f}%<br>" + ND + " = %{customdata[0]:,}<extra></extra>"))
    fig.add_vline(x=base_d, line_dash="dash", line_color=C["muted"], line_width=1.5)
    fig.add_annotation(x=base_d, y=1.02, yref="paper", showarrow=False, yanchor="bottom", font=dict(size=12, color=C["muted"]),
                       text=f"All AI users {base_d:.1f}%")
    fig.update_xaxes(range=[0, min(100, float(show_df["hi"].max()) + 10)], ticksuffix="%", title="Distrust in AI accuracy (somewhat + highly)")
    fig.update_yaxes(categoryorder="array", categoryarray=show_df["label"].tolist(), tickvals=show_df["label"].tolist(),
                     ticktext=[f"{r.label}  <span style='font-size:11px'>{ND}={r.n:,}</span>" for r in show_df.itertuples()])
    show(style(fig, max(360, 52 * len(show_df) + 90), margin=dict(l=10, r=30, t=40, b=50)))
    if SECURITY_ROLE in dt["role"].values:
        rank = int(dt.index[dt["role"] == SECURITY_ROLE][0]) + 1
        srow = dt.iloc[rank - 1]
        if rank == 1:
            nxt = dt.iloc[1]
            clear = srow["lo"] > nxt["hi"]
            en = (f"<b>Security professionals show the highest distrust of AI accuracy: {srow['dist']:.1f}% ({nlab(int(srow['n']))})</b>, ahead of {nxt['role']} ({nxt['dist']:.1f}%) among {len(dt)} roles with 30+ answers."
                  + (" Their 95% interval sits above the runner-up's." if clear else " Their 95% interval overlaps the runner-up's, so the exact ordering is less certain."))
        else:
            top = dt.iloc[0]
            en = (f"<b>Security professionals rank #{rank} of {len(dt)} roles for distrust of AI accuracy ({srow['dist']:.1f}%, {nlab(int(srow['n']))}).</b> "
                  f"In this audience the highest is {top['role']} ({top['dist']:.1f}%), so the “security professionals distrust most” pattern does not hold here.")
    else:
        top = dt.iloc[0]
        en = (f"<b>Security professionals have only {sec_n_all} valid answers in this audience (minimum {MIN_N}), so they are not compared.</b> "
              f"Among the roles shown, the highest distrust is {top['role']} ({top['dist']:.1f}%, {nlab(int(top['n']))}).")
    insight(en, "Roles that deal with risk every day may be the most cautious about trusting AI-generated answers. This compares survey opinions across roles; it does not show why they feel this way.")
    html(f'<div class="note">Top 10 roles by distrust among {len(dt)} roles with 30+ AI-accuracy answers; the dashed line is the audience-wide rate. Small roles have wide whiskers.</div>')

# ----------------------------------------------------------------------------
# 11 - AI adoption vs job-threat perception
# ----------------------------------------------------------------------------
section("11", "Job security", "threat", "Do Developers Who Use AI See It as a Threat to Their Jobs?",
        "The job-threat question was only shown to people who use AI or plan to, so non-users are not part of this comparison. Groups need 30+ answers.")
thr = fdf.dropna(subset=["AIThreat", "AISelect"])
g_rows = []
for key, nm in [("Yes", "Uses AI"), (PLAN_SOON, "Plans to soon")]:
    sub = thr[thr["AISelect"] == key]["AIThreat"]
    if len(sub) >= MIN_N:
        g_rows.append((nm, len(sub), *[(sub == a).mean() * 100 for a in ["Yes", "I'm not sure", "No"]]))
if not g_rows:
    low_data(len(thr))
else:
    gdf = pd.DataFrame(g_rows, columns=["group", "n", "Yes", "Not sure", "No"])
    fig = go.Figure()
    for col, key, colr in [("Yes", "Yes", C["warn"]), ("Not sure", "I'm not sure", C["sage"]), ("No", "No", C["forest"])]:
        nm_ = THREAT_EN[key]
        fig.add_bar(y=[f"{ai_status_name(r.group)}  ({ND}={r.n:,})" for r in gdf.itertuples()], x=gdf[col], name=nm_, orientation="h", marker=dict(color=colr),
                    text=[f"{v:.0f}%" for v in gdf[col]], textposition="inside", textfont=dict(color="#FFFFFF" if colr != C["sage"] else C["on_sage"]),
                    hovertemplate="%{y}<br>" + nm_ + ": %{x:.1f}%<extra></extra>")
    fig.update_layout(barmode="stack", legend=dict(orientation="h", y=1.25, font=dict(color=C["ink"])))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(range=[0, 100], ticksuffix="%")
    t1_, t2_ = st.columns([1.6, 1])
    with t1_:
        show(style(fig, 120 + 60 * len(gdf), legend=True, margin=dict(l=10, r=10, t=50, b=10)))
    with t2_:
        u = gdf[gdf["group"] == "Uses AI"]
        if len(u):
            u = u.iloc[0]
            ex_en = ""
            if len(gdf) == 2:
                o = gdf[gdf["group"] == "Plans to soon"].iloc[0]
                same = abs(u["Yes"] - o["Yes"]) < 3
                ex_en = f" Those planning to start answer {'almost the same' if same else 'differently'} ({o['Yes']:.1f}% yes)."
            insight(f"<b>{u['No']:.0f}% of AI users do not see AI as a threat to their job, while {u['Yes']:.0f}% do ({nlab(int(u['n']))}).</b>{ex_en}",
                    "Most developers who use AI do not feel threatened by it. This reflects opinions, not actual job outcomes.")
        else:
            insight("Not enough AI users with an answer in this audience for a reliable reading.", "")


# ----------------------------------------------------------------------------
# 12 - Relationships between variables: AI adoption and salary (bivariate + multivariate)
# ----------------------------------------------------------------------------
section("12", "Relationships between variables", "sal", "Is AI Adoption Associated With Different Salaries?",
        f"The EDA compares yearly salary (USD) across AI adoption groups, first everywhere, then inside the United States, then by experience level, and finally by role and experience together. Salaries above the IQR limit ({SAL_UPPER:,.0f}) are excluded from these salary analyses only. This shows association, never cause.")
sc = sal_clean(fdf)
GROUPS = [("Yes", "Uses AI"), (PLAN_SOON, "Plans to soon"), (NO_PLAN, "Doesn't plan to")]
GCOL = {"Yes": C["lime"], PLAN_SOON: C["forest"], NO_PLAN: C["sage"]}


def group_sal(frame):
    g = frame.groupby("AISelect")["ConvertedCompYearly"].agg(["count", "mean", "median"])
    return g


# (a) bivariate: all countries, (b) bivariate: United States only
sa, sb = st.columns(2)
for col_, frame_, ttl_ in [(sa, sc, "Bivariate · all countries"),
                          (sb, sc[sc["Country"] == US], "Bivariate · United States only")]:
    with col_:
        st.markdown(f"**{ttl_}**")
        g_ = group_sal(frame_)
        ok_ = [(k, nm) for k, nm in GROUPS if k in g_.index and g_.loc[k, "count"] >= MIN_SAL_N]
        if len(ok_) < 2:
            low_data(int(g_["count"].sum()) if len(g_) else 0, MIN_SAL_N)
            continue
        fig = go.Figure()
        for k, nm in ok_:
            fig.add_trace(go.Scatter(x=[0, g_.loc[k, "median"]], y=[ai_status_name(nm)] * 2, mode="lines", line=dict(color=GCOL[k], width=5), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(
            x=[g_.loc[k, "median"] for k, _ in ok_], y=[ai_status_name(nm) for _, nm in ok_], mode="markers+text", showlegend=False,
            marker=dict(size=22, color=[GCOL[k] for k, _ in ok_], line=dict(color=C["ink"], width=2)),
            text=[f"<b>{g_.loc[k, 'median']:,.0f}</b>" for k, _ in ok_], textposition="top center", textfont=dict(color=C["ink"], size=13),
            customdata=[[g_.loc[k, "count"], g_.loc[k, "mean"]] for k, _ in ok_],
            hovertemplate="%{y}<br>" + "Median" + ": %{x:,.0f} USD<br>" + "Mean" + ": %{customdata[1]:,.0f} USD<br>" + ND + " = %{customdata[0]:,.0f}<extra></extra>"))
        fig.update_xaxes(range=[0, float(max(g_.loc[k, "median"] for k, _ in ok_)) * 1.2], tickformat=",", title="Median yearly salary (USD)")
        fig.update_yaxes(autorange="reversed")
        fig.update_traces(cliponaxis=False, selector=dict(mode="markers+text"))
        show(style(fig, 280, margin=dict(l=10, r=40, t=50, b=40)))
        html(f'<div class="note">Groups: ' + " · ".join(f"{ai_status_name(nm)} {nlab(int(g_.loc[k, 'count']))}" for k, nm in ok_) + '</div>')

gall, gus = group_sal(sc), group_sal(sc[sc["Country"] == US])
if all(k in gall.index and k in gus.index for k in ["Yes", NO_PLAN]) and gus.loc["Yes", "count"] >= MIN_SAL_N and gus.loc[NO_PLAN, "count"] >= MIN_SAL_N and gall.loc["Yes", "count"] >= MIN_SAL_N and gall.loc[NO_PLAN, "count"] >= MIN_SAL_N:
    d_all = gall.loc["Yes", "median"] - gall.loc[NO_PLAN, "median"]
    d_us = gus.loc["Yes", "median"] - gus.loc[NO_PLAN, "median"]
    insight(f"<b>Across all countries, AI users’ median salary is {abs(d_all):,.0f} USD {'lower' if d_all < 0 else 'higher'} than non-users’ ({gall.loc['Yes', 'median']:,.0f} vs {gall.loc[NO_PLAN, 'median']:,.0f}). Inside the United States the gap is {abs(d_us):,.0f} USD ({gus.loc['Yes', 'median']:,.0f} vs {gus.loc[NO_PLAN, 'median']:,.0f}).</b>",
            "Pay levels differ hugely between countries, which is why the EDA repeats the comparison inside one country. " + ("Within the United States the AI-user median is not below the non-user median in this audience. " if d_us >= 0 else "Within the United States the AI-user median is lower than the non-user median in this audience. ") + "Either way, this does not show that AI changes salaries.")
else:
    html(f'<div class="note">The all-countries vs United States comparison needs 20+ salary records in both AI-user and non-user groups; the current filters leave too few.</div>')

# (c) three variables: US salary x experience x AI adoption
us_sal = sc[sc["Country"] == US]
st.markdown(f"**Three variables · US median salary by experience level and AI adoption**")
ge = us_sal.groupby(["ExperienceLevel", "AISelect"], observed=True)["ConvertedCompYearly"].agg(["count", "median"])
if len(ge) == 0:
    low_data(0, MIN_SAL_N)
    html(f'<div class="note">No United States salary records in the selected audience.</div>')
else:
    fig = go.Figure()
    levels_ok = [l for l in EXP_ORDER if l in ge.index.get_level_values(0)]
    for off, (k, nm) in zip([-0.22, 0, 0.22], GROUPS):
        xs_, ys_, cd_ = [], [], []
        for i, l in enumerate(levels_ok):
            if (l, k) in ge.index and ge.loc[(l, k), "count"] >= MIN_SAL_N:
                xs_.append(i + off)
                ys_.append(ge.loc[(l, k), "median"])
                cd_.append([l, ge.loc[(l, k), "count"]])
        if xs_:
            fig.add_trace(go.Scatter(x=xs_, y=ys_, mode="markers", name=ai_status_name(nm),
                                     marker=dict(size=18, color=GCOL[k], line=dict(color=C["ink"], width=2)), customdata=cd_,
                                     hovertemplate="%{customdata[0]}<br>" + ai_status_name(nm) + "<br>" + "Median" + ": %{y:,.0f} USD<br>" + ND + " = %{customdata[1]:,.0f}<extra></extra>"))
            for x, y in zip(xs_, ys_):
                fig.add_shape(type="line", x0=x, x1=x, y0=0, y1=y, line=dict(color=GCOL[k], width=3), layer="below")
    fig.update_xaxes(tickvals=list(range(len(levels_ok))), ticktext=[l for l in levels_ok], range=[-0.6, len(levels_ok) - 0.4])
    fig.update_yaxes(tickformat=",", title="Median yearly salary (USD)", rangemode="tozero")
    fig.update_layout(legend=dict(orientation="h", y=1.15, font=dict(color=C["ink"])))
    show(style(fig, 380, legend=True, margin=dict(l=10, r=20, t=40, b=10)))
    comp, cnt_hi = 0, 0
    for l in levels_ok:
        if (l, "Yes") in ge.index and (l, NO_PLAN) in ge.index and ge.loc[(l, "Yes"), "count"] >= MIN_SAL_N and ge.loc[(l, NO_PLAN), "count"] >= MIN_SAL_N:
            comp += 1
            cnt_hi += int(ge.loc[(l, "Yes"), "median"] > ge.loc[(l, NO_PLAN), "median"])
    if comp:
        insight(f"<b>In the United States, AI users have a higher median salary than non-users in {cnt_hi} of {comp} experience groups.</b>",
                f"This compares three variables at once (salary, experience, AI adoption). It shows association only and does not establish that AI adoption raises or lowers salaries. Only groups with {MIN_SAL_N}+ salary records are drawn.")
    else:
        html(f'<div class="note">Not enough US salary records per experience group to compare AI users with non-users.</div>')

# (d) multivariate: role x experience x AI adoption (US) - groups with 20+ AI users and 20+ non-users
st.markdown(f"**Multivariate · US median salary by role, experience and AI adoption**")
valid_rows = []
if len(us_sal):
    gm = us_sal.groupby(["DevType", "ExperienceLevel", "AISelect"], observed=True)["ConvertedCompYearly"].agg(["count", "median"]).unstack("AISelect")
    if ("count", "Yes") in gm.columns and ("count", NO_PLAN) in gm.columns:
        ok_m = (gm[("count", "Yes")] >= MIN_SAL_N) & (gm[("count", NO_PLAN)] >= MIN_SAL_N)
        for (role_, lvl_), r_ in gm[ok_m].iterrows():
            valid_rows.append((role_, lvl_, r_[("median", NO_PLAN)], r_[("median", "Yes")], int(r_[("count", NO_PLAN)]), int(r_[("count", "Yes")])))
if not valid_rows:
    low_data(len(us_sal), MIN_SAL_N)
    html(f'<div class="note">This comparison keeps only role × experience groups with {MIN_SAL_N}+ AI users and {MIN_SAL_N}+ non-users (non-users = “don’t plan to”) among US salary records. None qualify in the selected audience.</div>')
else:
    vm = pd.DataFrame(valid_rows, columns=["role", "lvl", "non", "ai", "n_non", "n_ai"])
    vm["diff"] = vm["ai"] - vm["non"]
    vm = vm.sort_values("diff").reset_index(drop=True)
    vm["label"] = [f"{r} · {l}" for r, l in zip(vm["role"], vm["lvl"])]
    fig = go.Figure()
    for r in vm.itertuples():
        fig.add_trace(go.Scatter(x=[r.non, r.ai], y=[r.label, r.label], mode="lines", line=dict(color=C["lime"] if r.diff > 0 else C["warn"], width=6), hoverinfo="skip", showlegend=False, opacity=.8))
    fig.add_trace(go.Scatter(x=vm["non"], y=vm["label"], mode="markers", name="Non-users (don't plan to)",
                             marker=dict(size=16, color=C["sage"], line=dict(color=C["ink"], width=2)), customdata=list(zip(vm["n_non"], vm["diff"])),
                             hovertemplate="<b>%{y}</b><br>" + "Non-users" + ": %{x:,.0f} USD<br>" + ND + " = %{customdata[0]:,}<extra></extra>"))
    fig.add_trace(go.Scatter(x=vm["ai"], y=vm["label"], mode="markers", name="AI users",
                             marker=dict(size=16, color=C["forest"], line=dict(color=C["lime"], width=3)), customdata=list(zip(vm["n_ai"], vm["diff"])),
                             hovertemplate="<b>%{y}</b><br>" + "AI users" + ": %{x:,.0f} USD<br>" + ND + " = %{customdata[0]:,}<br>" + "Difference" + ": %{customdata[1]:+,.0f} USD<extra></extra>"))
    fig.update_xaxes(tickformat=",", title="Median yearly salary (USD)")
    fig.update_yaxes(categoryorder="array", categoryarray=vm["label"].tolist())
    fig.update_layout(legend=dict(orientation="h", y=1.1, font=dict(color=C["ink"])))
    show(style(fig, max(340, 48 * len(vm) + 110), legend=True, margin=dict(l=10, r=20, t=40, b=50)))
    higher = int((vm["diff"] > 0).sum())
    insight(f"<b>Among {len(vm)} comparable US role × experience groups, AI users have a higher median salary in {higher} and a lower one in {len(vm) - higher}.</b>",
            f"AI adoption is not consistently tied to lower salaries; differences vary by role and experience. Each group needs {MIN_SAL_N}+ AI users and {MIN_SAL_N}+ non-users. Association only: it does not show that AI changes pay.")
    html(f'<div class="note">Lime lines mean AI users have the higher median; red-orange lines mean non-users do.</div>')


# ----------------------------------------------------------------------------
# 13 - Find yourself in the data (always the FULL survey, a stable benchmark)
# ----------------------------------------------------------------------------
section("13", "Find yourself in the data", "you", "How Does Your Profile Compare?",
        "Pick your role and experience level. We compare you with survey respondents who share your profile. This section always uses the full survey, so the benchmark stays stable when you change the sidebar filters.")
f1, f2, f3, f4 = st.columns([1, 1.5, 1.2, 1])
with f1:
    name = st.text_input("Your name (optional)", placeholder="e.g. Sam", key="me_name")
with f2:
    my_role = st.selectbox("Your role (type to search)", ALL_ROLES, index=0, key="me_role")
with f3:
    my_exp = st.selectbox("Experience level", EXP_ORDER, index=1, key="me_exp", format_func=lambda x: f"{x} ({EXP_HELP[x]})")
with f4:
    my_ai = st.radio("Do you use AI?", ["Yes", "No"], horizontal=True, key="me_ai", format_func=lambda x: "Yes" if x == "Yes" else "No")

role_frame = df[df["DevType"] == my_role]
a_match, n_match = adoption(role_frame[role_frame["ExperienceLevel"] == my_exp])
a_role, n_role = adoption(role_frame)
who = f"{name.strip()}, " if name.strip() else ""
r_lbl, e_lbl = my_role, my_exp

if n_match >= MIN_N:
    level, a_cmp, n_cmp, cmp_label = "match", a_match, n_match, f"{e_lbl} · {r_lbl}"
elif n_role >= MIN_N:
    level, a_cmp, n_cmp, cmp_label = "role", a_role, n_role, f"All {r_lbl}"
else:
    level, a_cmp, n_cmp, cmp_label = "none", float("nan"), n_role, ""

if level == "none":
    html(f'<div class="insight"><b>{who}we don’t have enough data to compare you reliably.</b> '
         f'Only {n_role} respondents with the role “{r_lbl}” answered the AI question (we need at least {MIN_N}). Try a more common role to see a comparison.</div>')
else:
    if level == "role":
        html(f'<div class="note"><b>Heads-up:</b> only {n_match} respondents match both “{r_lbl}” and “{e_lbl}” (minimum {MIN_N}), so this comparison uses everyone in the role ({nlab(n_role)}).</div>')
    y1, y2 = st.columns([1, 1.25])
    with y1:  # 100-dot unit chart: "100 people like you"
        users = max(0, min(100, int(round(a_cmp))))
        xs_ = [i % 10 for i in range(100)]
        ys_ = [9 - i // 10 for i in range(100)]
        you_idx = 0 if my_ai == "Yes" else min(users, 99)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=xs_, y=ys_, mode="markers", showlegend=False,
            marker=dict(size=24, color=[C["lime"] if i < users else C["sage"] for i in range(100)], opacity=[1 if i < users else .45 for i in range(100)], line=dict(color=C["ink"], width=1)),
            hovertemplate=[("Uses AI" if i < users else "Doesn't use AI") + "<extra></extra>" for i in range(100)]))
        fig.add_trace(go.Scatter(x=[xs_[you_idx]], y=[ys_[you_idx]], mode="markers", showlegend=False,
                                 marker=dict(size=34, color="rgba(0,0,0,0)", line=dict(color=C["ink"], width=3.5)), hovertemplate="<b>You</b>" + "<extra></extra>"))
        fig.add_annotation(x=xs_[you_idx], y=ys_[you_idx], text="<b>You</b>", ax=-34, ay=-34, showarrow=True, arrowcolor=C["ink"], font=dict(size=13, color=C["ink"]))
        fig.update_xaxes(visible=False, range=[-0.8, 9.8])
        fig.update_yaxes(visible=False, range=[-0.8, 9.8], scaleanchor="x", scaleratio=1)
        show(style(fig, 400, margin=dict(l=0, r=0, t=6, b=0)))
        html(f'<div class="note">100 respondents like you ({cmp_label}, {nlab(n_cmp)}). '
             f'<span class="dot" style="background:{C["lime"]}"></span>use AI '
             f'<span class="dot" style="background:{C["sage"]};opacity:.5"></span>don’t.</div>')
    with y2:
        bars = [(cmp_label, a_cmp, n_cmp)]
        if level == "match":
            bars.append((f"All {r_lbl}", a_role, n_role))
        bars.append(("All respondents", ADOPT, N_SEL))
        fig = go.Figure()
        for lab_, v_, n_ in bars[:-1]:
            fig.add_trace(go.Scatter(x=[ADOPT, v_], y=[lab_, lab_], mode="lines", showlegend=False, hoverinfo="skip", line=dict(color=C["lime"] if v_ >= ADOPT else C["warn"], width=5), opacity=.7))
        fig.add_trace(go.Scatter(
            x=[b[1] for b in bars], y=[b[0] for b in bars], mode="markers+text", showlegend=False,
            marker=dict(size=[26, 20, 20][:len(bars)], color=[C["lime"]] + [C["forest"]] * (len(bars) - 1), line=dict(color=C["ink"], width=2)),
            text=[f"<b>{b[1]:.1f}%</b>" for b in bars], textposition="top center", textfont=dict(color=C["ink"], size=14),
            customdata=[b[2] for b in bars], hovertemplate="%{y}<br>%{x:.1f}% " + "use AI" + "<br>" + ND + " = %{customdata:,}<extra></extra>"))
        fig.update_xaxes(range=[0, 100], ticksuffix="%", title="Share using AI")
        fig.update_yaxes(autorange="reversed")
        show(style(fig, 230, margin=dict(l=10, r=20, t=20, b=40)))
        same = a_cmp if my_ai == "Yes" else 100 - a_cmp
        tone = "you’re in the majority" if same >= 55 else ("the group is nearly split, so you’re in very mixed company" if same >= 45 else "you’re in the minority")
        diff = a_cmp - ADOPT
        verb_en = "use" if my_ai == "Yes" else "don’t currently use"
        insight(f"<b>{who}you {verb_en} AI, and so do {same:.0f}% of comparable developers: {tone}.</b> Their AI use is {abs(diff):.1f} points {'above' if diff >= 0 else 'below'} the all-respondent rate.",
                "Roles and experience levels differ a lot in AI use, but plenty of people in every group do the opposite of the average.")

# ----------------------------------------------------------------------------
# 14 - The Trust Test (always the FULL survey)
# ----------------------------------------------------------------------------
section("14", "The Trust Test", "lab", "How Well Can You Predict the Results?",
        "Before seeing the answers, guess three numbers drawn from real survey results (full survey). Then see how your instincts compare.")
with st.form("trust_test"):
    t1, t2, t3 = st.columns(3)
    with t1:
        g1_ = st.slider("Of AI users, what percent say they trust AI accuracy (somewhat or highly)?", 0, 100, 50, key="tt_g1")
    with t2:
        g2_ = st.slider("Of AI users who report a productivity benefit, what percent also don't trust the output?", 0, 100, 50, key="tt_g2")
    with t3:
        g3_ = st.slider("Of all respondents, what percent use AI tools?", 0, 100, 50, key="tt_g3")
    go_btn = st.form_submit_button("Reveal the data")
if go_btn:
    st.session_state["tt"] = (g1_, g2_, g3_)  # kept so results survive a theme switch

if st.session_state.get("tt"):
    gs = st.session_state["tt"]
    items = [
        ("AI users who trust its accuracy", gs[0], P_TRUST, N_ACC),
        ("Productive users who also distrust the output", gs[1], PX["p_dist_given_prod"], PX["n_prod"]),
        ("All respondents who use AI", gs[2], ADOPT, N_SEL),
    ]
    gaps = [abs(g - a) for _, g, a, _ in items]
    avg_gap = sum(gaps) / 3
    fig = go.Figure()
    for i, (long_, g, a, n_) in enumerate(items):
        y = 2 - i
        close = abs(g - a) <= 10
        fig.add_trace(go.Scatter(x=[0, 100], y=[y, y], mode="lines", line=dict(color=rgba(C["forest"], .22), width=12), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=[g, a], y=[y, y], mode="lines", line=dict(color=C["lime"] if close else C["warn"], width=12), opacity=.85, hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=[g], y=[y], mode="markers", showlegend=False, marker=dict(size=24, symbol="diamond", color=C["card_solid"], line=dict(color=C["ink"], width=3)),
                                 hovertemplate="<b>Your guess</b>" + f": {g}%<extra></extra>"))
        fig.add_trace(go.Scatter(x=[a], y=[y], mode="markers", showlegend=False, marker=dict(size=26, color=C["forest"], line=dict(color=C["lime"], width=3)),
                                 hovertemplate="<b>Survey</b>" + f": {a:.1f}%<br>{nlab(n_)}<extra></extra>"))
        fig.add_annotation(x=0, y=y + 0.42, xanchor="left", text=f"<b>{long_}</b>", showarrow=False, font=dict(size=13, color=C["ink"]))
        fig.add_annotation(x=(g + a) / 2, y=y - 0.36, showarrow=False, font=dict(size=12, color=C["muted"]),
                           text=f"you {g}% · data {a:.0f}% · off by {abs(g - a):.0f}")
    fig.update_xaxes(range=[-2, 102], ticksuffix="%", tickvals=[0, 25, 50, 75, 100])
    fig.update_yaxes(visible=False, range=[-0.7, 2.8])
    verdict = "Sharp instincts." if avg_gap <= 8 else ("Pretty close." if avg_gap <= 18 else "The data surprised you.")
    k1, k2 = st.columns([1, 2.4])
    with k1:
        html(f'<div class="giant" style="font-size:6rem">{avg_gap:.0f}<small>pts</small></div>'
             f'<div class="giant-cap"><b>{verdict}</b> That is how far off your three guesses were on average.</div>')
    with k2:
        show(style(fig, 400, margin=dict(l=10, r=10, t=10, b=30)))
        html(f'<div class="note"><span class="dot" style="background:{C["card_solid"]};border:2px solid {C["ink"]};border-radius:2px"></span>your guess '
             f'<span class="dot" style="background:{C["forest"]};border:2px solid {C["lime"]}"></span>survey result '
             f'<span class="dot" style="background:{C["lime"]}"></span>within 10 points '
             f'<span class="dot" style="background:{C["warn"]}"></span>more than 10 points off</div>')
    wi = gaps.index(max(gaps))
    w = items[wi]
    insight(f"<b>Your biggest miss:</b> you {'overestimated' if w[1] > w[2] else 'underestimated'} “{w[0].lower()}” ({w[1]}% guessed vs {w[2]:.0f}% in the data).",
            "Using AI, finding it useful and trusting it are three different things. The survey shows all three at once, which is the paradox.")
    _np = nlab(PX["n_prod"])
    html(f'<div class="note">Bases: trust, {nlab(N_ACC)}; productive-but-distrust, {_np}; adoption, {nlab(N_SEL)}.</div>')


# ----------------------------------------------------------------------------
# 15 - The story in 30 seconds (always the FULL survey)
# ----------------------------------------------------------------------------
section("15", "Wrap-up", "story", "The Story in 30 Seconds",
        "Every number below comes from the full survey, so it does not change with the sidebar filters.")
hiF, loF = EXP_FULL.loc[EXP_FULL["adopt"].idxmax()], EXP_FULL.loc[EXP_FULL["adopt"].idxmin()]
rtopF = ROLE_FULL.iloc[-1]
pProdF, pAccF = float(BEN_PCT_FULL.get(PRODUCTIVITY_KEY, float("nan"))), float(BEN_PCT_FULL.get(ACCURACY_KEY, float("nan")))
chTopK = CH_PCT_FULL.idxmax()
chTopV = float(CH_PCT_FULL.max())
secF = DIS_FULL[DIS_FULL["role"] == SECURITY_ROLE]
secRank = int(secF.index[0]) + 1 if len(secF) else None
_us = sal_clean(df)
_gU = group_sal(_us[_us["Country"] == US])
us_ok = all(k in _gU.index for k in ["Yes", NO_PLAN])
cards = [
    (f"{ADOPT:.0f}%", f"use AI ({nlab(N_SEL)})"),
    (f"{hiF['adopt']:.0f}% → {loF['adopt']:.0f}%", f"adoption from {hiF['level']} to {loF['level']} developers"),
    (f"{rtopF['adopt']:.0f}%", f"in the top role: {rtopF['role']} ({nlab(int(rtopF['n']))})"),
    (f"{pProdF:.0f}% vs {pAccF:.0f}%", f"report productivity vs better coding accuracy ({nlab(N_BEN_FULL)})"),
    (f"{chTopV:.0f}%", f"name “{ch_name(chTopK)}” as a challenge, the most common ({nlab(N_CH_FULL)})"),
    (f"{PX['p_both']:.0f}%", f"report productivity gains and distrust AI output ({nlab(PX['n'])})"),
    (f"{P_HIGH_TRUST:.1f}%", f"highly trust AI accuracy; {P_DISTRUST:.0f}% distrust it ({nlab(N_ACC)})"),
]
if len(secF):
    cards.append((f"{secF.iloc[0]['dist']:.0f}%", f"of security professionals distrust AI accuracy (#{secRank} among {len(DIS_FULL)} roles, {nlab(int(secF.iloc[0]['n']))})"))
if us_ok:
    cards.append((f"{_gU.loc['Yes', 'median'] / 1000:.0f}k vs {_gU.loc[NO_PLAN, 'median'] / 1000:.0f}k",
                  "US median salary (USD): AI users vs non-users, association only"))
html('<div class="story">' + "".join(f'<div class="it"><div class="n">{n}</div><div class="l">{l}</div></div>' for n, l in cards) + '</div>')
sec_line_en = (f" Security professionals are the most doubtful role (#{secRank} of {len(DIS_FULL)} with {secF.iloc[0]['dist']:.0f}% distrust)." if secRank == 1
               else (f" Security professionals rank #{secRank} for distrust ({secF.iloc[0]['dist']:.0f}%)." if secRank else ""))
html(f"""
<div class="conclusion">
<h3>The real story, in one paragraph</h3>
<p>AI changed the game for developers: {ADOPT:.0f}% of respondents already use it, and those who do mostly report productivity gains ({pProdF:.0f}%) far more than better accuracy ({pAccF:.0f}%). But adoption is uneven, falling from {hiF['adopt']:.0f}% among {hiF['level']} developers to {loF['adopt']:.0f}% among {loF['level']} ones, and confidence is another matter: only {P_HIGH_TRUST:.1f}% highly trust AI accuracy, the leading challenge is not trusting the output ({chTopV:.0f}%), and {PX['p_dist_given_prod']:.0f}% of those who report productivity gains also struggle to trust it.{sec_line_en} Developers seem to use AI like a quick but fallible colleague: gladly, and with their guard up. These are survey associations, not proof of cause, and they measure opinions, not code quality. In 2024, AI earned a place in the workflow, not yet in developers’ full trust.</p>
</div>
""")
html(f'<div class="note" style="text-align:center;margin-top:30px">{TITLE} · Source: Stack Overflow Developer Survey 2024 · cleaned_data.csv · all figures calculated live from the data.</div>')