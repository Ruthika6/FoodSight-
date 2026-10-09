"""
UI Styling and Enterprise SaaS Design System for FoodSight Dashboard.
Implements crisp light-mode typography, clean card styling, status banners,
dark navy sidebar with high-contrast text and vibrant emerald green active navigation pills.
"""

import textwrap
import streamlit as st


SAAS_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

/* Main Body Typography */
.block-container, .block-container p, .block-container h1, .block-container h2, .block-container h3, .block-container h4, .block-container h5, .block-container h6, .block-container label, .block-container span {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    color: #0F172A;
}

/* Explicitly preserve Material Icons / Streamlit System Icons */
[data-testid="stIconMaterial"], 
[class*="material-symbols"], 
[class*="material-icons"], 
.material-symbols-rounded, 
.material-symbols-outlined,
[data-testid="stSidebarCollapseButton"] button span,
[data-testid="stExpander"] summary span:first-child {
    font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons', sans-serif !important;
    font-feature-settings: 'liga' 1 !important;
    text-transform: none !important;
    letter-spacing: normal !important;
    word-wrap: normal !important;
    white-space: nowrap !important;
    direction: ltr !important;
}

/* Streamlit Header adjustments to eliminate top gap */
header[data-testid="stHeader"] {
    background: transparent !important;
    height: 1.25rem !important;
    min-height: 0 !important;
}

/* Optimize Page Spacing - Move FoodSight banner close to the top and to the left */
.block-container {
    padding-top: 0.15rem !important;
    padding-left: 1.5rem !important;
    padding-right: 2rem !important;
    padding-bottom: 2rem !important;
    max-width: 100% !important;
}

/* Top Global App Header */
.top-navbar-container {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.65rem 1.15rem;
    background: #FFFFFF;
    border-radius: 12px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    margin-top: 0 !important;
    margin-bottom: 1rem;
}

.top-navbar-left {
    display: flex;
    align-items: center;
    gap: 12px;
}

.top-navbar-icon {
    width: 42px;
    height: 42px;
    background: #059669;
    color: #FFFFFF;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.35rem;
    box-shadow: 0 4px 10px rgba(5, 150, 105, 0.25);
    flex-shrink: 0;
}

.top-navbar-title {
    font-size: 1.45rem !important;
    font-weight: 800 !important;
    color: #0F172A !important;
    margin: 0 !important;
    letter-spacing: -0.02em !important;
    line-height: 1.15 !important;
}

.top-navbar-subtitle {
    font-size: 0.82rem !important;
    color: #64748B !important;
    margin-top: 2px !important;
    margin-bottom: 0 !important;
    font-weight: 500 !important;
}

.top-navbar-right {
    display: flex;
    align-items: center;
    gap: 10px;
}

.top-model-badge {
    background: #ECFDF5;
    color: #047857;
    border: 1px solid #A7F3D0;
    padding: 0.4rem 0.85rem;
    border-radius: 8px;
    font-size: 0.8rem;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 6px;
}

.top-health-badge {
    background: #FFFFFF;
    color: #059669;
    border: 1.5px solid #10B981;
    padding: 0.4rem 0.85rem;
    border-radius: 8px;
    font-size: 0.8rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    gap: 6px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}

.top-health-badge-red {
    background: #FEF2F2;
    color: #DC2626;
    border: 1.5px solid #EF4444;
    padding: 0.4rem 0.85rem;
    border-radius: 8px;
    font-size: 0.8rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* Page Section Header */
.main-header-container {
    padding: 0.1rem 0.1rem 0.85rem 0.1rem;
    margin-bottom: 0.75rem;
}

.main-header-title {
    font-size: 1.55rem !important;
    font-weight: 800 !important;
    margin: 0 !important;
    color: #0F172A !important;
    letter-spacing: -0.025em !important;
}

.main-header-subtitle {
    font-size: 0.9rem !important;
    color: #64748B !important;
    margin-top: 0.25rem !important;
    margin-bottom: 0 !important;
    line-height: 1.4 !important;
}

/* System State Green Alert Banner */
.system-state-banner {
    background: #ECFDF5;
    border: 1px solid #A7F3D0;
    border-radius: 12px;
    padding: 1.1rem 1.35rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1.2rem;
    box-shadow: 0 2px 8px rgba(16, 185, 129, 0.08);
}

.system-state-left {
    display: flex;
    align-items: flex-start;
    gap: 14px;
}

.system-state-icon {
    width: 34px;
    height: 34px;
    background: #10B981;
    color: #FFFFFF;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.15rem;
    font-weight: 800;
    flex-shrink: 0;
    box-shadow: 0 2px 6px rgba(16, 185, 129, 0.3);
}

.system-state-title {
    font-size: 1.02rem !important;
    font-weight: 800 !important;
    color: #065F46 !important;
    margin: 0 !important;
    line-height: 1.3 !important;
}

.system-state-sub {
    font-size: 0.86rem !important;
    color: #047857 !important;
    margin-top: 4px !important;
    margin-bottom: 0 !important;
    font-weight: 500 !important;
}

/* KPI White Cards */
.saas-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 1.1rem 1.15rem;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
    margin-bottom: 1rem;
    transition: all 0.2s ease;
    min-height: 140px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}

.saas-card:hover {
    border-color: #CBD5E1;
    box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
    transform: translateY(-2px);
}

.saas-card-label {
    font-size: 0.76rem !important;
    font-weight: 700 !important;
    color: #64748B !important;
    margin-bottom: 0.35rem !important;
    letter-spacing: 0.02em !important;
}

.saas-card-value {
    font-size: 1.6rem !important;
    font-weight: 800 !important;
    color: #0F172A !important;
    margin-bottom: 0.25rem !important;
    line-height: 1.15 !important;
}

.saas-card-sub {
    font-size: 0.76rem !important;
    color: #94A3B8 !important;
    line-height: 1.35 !important;
}

/* Daily Action Directive Dark Banner */
.directive-box {
    background: #0F172A;
    border: 1px solid #1E293B;
    padding: 1.1rem 1.35rem;
    border-radius: 12px;
    margin-top: 1rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.12);
}

.directive-title {
    font-weight: 800 !important;
    color: #34D399 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    margin-bottom: 0.4rem !important;
    display: flex;
    align-items: center;
    gap: 8px;
}

.directive-text {
    font-size: 0.9rem !important;
    color: #E2E8F0 !important;
    line-height: 1.5 !important;
}

/* Modern Badges */
.badge-green {
    background: #ECFDF5;
    color: #059669 !important;
    border: 1px solid #A7F3D0;
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    display: inline-flex;
    align-items: center;
}

.badge-amber {
    background: #FFFBEB;
    color: #D97706 !important;
    border: 1px solid #FDE68A;
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    display: inline-flex;
    align-items: center;
}

.badge-red {
    background: #FEF2F2;
    color: #DC2626 !important;
    border: 1px solid #FECACA;
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    display: inline-flex;
    align-items: center;
}

.badge-blue {
    background: #EFF6FF;
    color: #2563EB !important;
    border: 1px solid #BFDBFE;
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    display: inline-flex;
    align-items: center;
}

/* ============================================================
   SIDEBAR STYLING — HIGH CONTRAST DARK NAVY + BRIGHT WHITE TEXT
   ============================================================ */
[data-testid="stSidebar"] {
    background-color: #0B132B !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
}

/* Eliminate empty top gap in Streamlit sidebar */
[data-testid="stSidebarHeader"] {
    display: none !important;
    height: 0 !important;
    min-height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
}

[data-testid="stSidebar"] section,
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div,
[data-testid="stSidebarContent"],
[data-testid="stSidebarUserContent"],
div[class*="stSidebarUserContent"],
div[class*="stSidebarContent"] {
    padding-top: 0.5rem !important;
    padding-left: 0.75rem !important;
    padding-right: 0.75rem !important;
}

.sidebar-section-heading {
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: #94A3B8 !important;
    margin-top: 0.75rem !important;
    margin-bottom: 0.5rem !important;
    padding-left: 0.35rem !important;
}

/* Hide redundant radio widget label */
[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stWidgetLabel"] {
    display: none !important;
}

/* Sidebar Navigation Items */
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {
    gap: 4px !important;
    display: flex !important;
    flex-direction: column !important;
}

/* Radio button row wrapper */
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 8px !important;
    padding: 0.55rem 0.85rem !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
    margin: 0 !important;
    width: 100% !important;
    display: flex !important;
    align-items: center !important;
}

/* Hide default circle dot icon completely from radio buttons */
[data-testid="stSidebar"] [data-testid="stRadio"] label > div:not(:has([data-testid="stMarkdownContainer"])):not([data-testid="stMarkdownContainer"]),
[data-testid="stSidebar"] [data-testid="stRadio"] label > div:first-child:not([data-testid="stMarkdownContainer"]),
[data-testid="stSidebar"] [data-testid="stRadio"] label > span:not([data-testid="stMarkdownContainer"]),
[data-testid="stSidebar"] [data-testid="stRadio"] label > div:first-child,
[data-testid="stSidebar"] [data-testid="stRadio"] label input,
[data-testid="stSidebar"] [data-testid="stRadio"] label svg {
    display: none !important;
    width: 0 !important;
    height: 0 !important;
    min-width: 0 !important;
    min-height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    opacity: 0 !important;
    visibility: hidden !important;
    position: absolute !important;
}

/* Force visible bright white text for navigation options */
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label div[data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label div[data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label span {
    color: #F8FAFC !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    margin: 0 !important;
    width: 100% !important;
}

/* Hover effect on inactive items */
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:hover {
    background: rgba(255, 255, 255, 0.1) !important;
    border-color: rgba(255, 255, 255, 0.15) !important;
}

/* ACTIVE NAVIGATION PILL — VIBRANT EMERALD GREEN */
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {
    background: #059669 !important;
    border: 1px solid #10B981 !important;
    box-shadow: 0 4px 14px rgba(5, 150, 105, 0.4) !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) div[data-testid="stMarkdownContainer"] p {
    color: #FFFFFF !important;
    font-weight: 700 !important;
}

/* Data source selectbox in sidebar */
[data-testid="stSidebar"] div[data-baseweb="select"] {
    background: #111D3D !important;
    border-radius: 8px !important;
}

[data-testid="stSidebar"] div[data-baseweb="select"] * {
    color: #F8FAFC !important;
}

/* Streamlit Button Styling */
div.stButton > button:first-child {
    border-radius: 8px !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em !important;
    transition: all 0.2s ease !important;
    border: 1px solid #CBD5E1 !important;
    background: #FFFFFF !important;
    color: #0F172A !important;
}

div.stButton > button:first-child:hover {
    transform: translateY(-1px) !important;
    border-color: #059669 !important;
    box-shadow: 0 4px 12px rgba(5, 150, 105, 0.15) !important;
}

div.stButton > button[kind="primary"] {
    background: #059669 !important;
    color: #FFFFFF !important;
    border: 1px solid #059669 !important;
}

div.stButton > button[kind="primary"]:hover {
    background: #047857 !important;
    box-shadow: 0 4px 14px rgba(5, 150, 105, 0.35) !important;
}
</style>
"""


def apply_custom_styles() -> None:
    """Inject custom SaaS styles into Streamlit app."""
    st.markdown(SAAS_CSS, unsafe_allow_html=True)


def render_top_navbar(model_name: str = "XGBoost", is_healthy: bool = True, status_text: str = "MODEL HEALTHY") -> None:
    """Render the top banner bar with FoodSight Logo, Title, and Right Status Badges."""
    badge_health_class = "top-health-badge" if is_healthy else "top-health-badge-red"
    icon = "✓" if is_healthy else "⚠️"
    
    html = f"""<div class="top-navbar-container">
<div class="top-navbar-left">
<div class="top-navbar-icon">🍽️</div>
<div>
<h1 class="top-navbar-title">FoodSight</h1>
<div class="top-navbar-subtitle">Smart Food Demand & Zero-Waste Operational Intelligence</div>
</div>
</div>
<div class="top-navbar-right">
<div class="top-model-badge">
<span>⚡</span>
<span>Active Model: <strong>v1.0 ({model_name.split(' ')[0]})</strong></span>
</div>
<div class="{badge_health_class}">
<span>{icon}</span>
<span>{status_text}</span>
</div>
</div>
</div>"""
    st.markdown(html, unsafe_allow_html=True)


def render_app_header(title: str, subtitle: str) -> None:
    """Render page section title and description."""
    header_html = f"""<div class="main-header-container">
<h2 class="main-header-title">{title}</h2>
<p class="main-header-subtitle">{subtitle}</p>
</div>"""
    st.markdown(header_html, unsafe_allow_html=True)


def render_system_status_banner(is_healthy: bool, model_name: str, rolling_mae: float) -> None:
    """Render the prominent mint-green system state alert banner."""
    status_label = "MODEL HEALTHY — Operations Running Inside Tolerances" if is_healthy else "DRIFT DETECTED — Retraining Recommended"
    bg_color = "#ECFDF5" if is_healthy else "#FEF2F2"
    border_color = "#A7F3D0" if is_healthy else "#FECACA"
    icon_bg = "#10B981" if is_healthy else "#EF4444"
    icon = "✓" if is_healthy else "!"
    title_color = "#065F46" if is_healthy else "#991B1B"
    sub_color = "#047857" if is_healthy else "#B91C1C"

    html = f"""<div class="system-state-banner" style="background: {bg_color}; border-color: {border_color};">
<div class="system-state-left">
<div class="system-state-icon" style="background: {icon_bg};">{icon}</div>
<div>
<div class="system-state-title" style="color: {title_color};">System State: {status_label}</div>
<div class="system-state-sub" style="color: {sub_color};">
Active model version v1.0 ({model_name.split(' ')[0]}) is accurately calibrating daily demand. 7-day rolling MAE is {rolling_mae:.1f} meals.
</div>
</div>
</div>
</div>"""
    st.markdown(html, unsafe_allow_html=True)


def render_kpi_card(label: str, value: str, subtext: str = "", badge: str = "", badge_type: str = "green", value_color: str = "#0F172A") -> None:
    """Render a clean white SaaS metric card without unparsed markdown code blocks."""
    badge_html = f'<span class="badge-{badge_type}">{badge}</span>' if badge else ""
    card_html = f"""<div class="saas-card">
<div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.35rem; gap: 8px;">
<span class="saas-card-label">{label}</span>
{badge_html}
</div>
<div class="saas-card-value" style="color: {value_color};">{value}</div>
<div class="saas-card-sub">{subtext}</div>
</div>"""
    st.markdown(card_html, unsafe_allow_html=True)


def render_directive_box(title: str, text: str) -> None:
    """Render the daily operational directive banner at the bottom of overview."""
    html = f"""<div class="directive-box">
<div class="directive-title">✨ {title}</div>
<div class="directive-text">{text}</div>
</div>"""
    st.markdown(html, unsafe_allow_html=True)


def get_plotly_light_layout(title: str = "", height: int = 340, **kwargs) -> dict:
    """Standardized Plotly crisp light layout config matching modern SaaS themes."""
    layout = dict(
        title=dict(text=title, font=dict(color="#0F172A", size=14, family="Plus Jakarta Sans")),
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#64748B", family="Plus Jakarta Sans"),
        margin=dict(l=20, r=20, t=45 if title else 20, b=25),
        height=height,
        xaxis=dict(gridcolor="#F1F5F9", zerolinecolor="#E2E8F0"),
        yaxis=dict(gridcolor="#F1F5F9", zerolinecolor="#E2E8F0"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#475569", size=11),
            bgcolor="rgba(0,0,0,0)"
        )
    )
    for k, v in kwargs.items():
        if isinstance(v, dict) and k in layout and isinstance(layout[k], dict):
            layout[k] = {**layout[k], **v}
        else:
            layout[k] = v
    return layout


# Alias for compatibility across all pages
get_plotly_dark_layout = get_plotly_light_layout
