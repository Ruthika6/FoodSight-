"""
UI Styling and SaaS Design System for Streamlit Dashboard.
Implements modern dark-mode glassmorphic typography, card styling, badge utilities, and chart themes.
"""

import streamlit as st


SAAS_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"], .stMarkdown, .stText, p, span, label, div {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}

/* App Header Styling */
.main-header-container {
    padding: 1.5rem 1.75rem;
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.95) 0%, rgba(15, 23, 42, 0.95) 100%);
    border-radius: 14px;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5), 0 0 15px rgba(59, 130, 246, 0.1);
    backdrop-filter: blur(16px);
}

.main-header-title {
    font-size: 1.65rem !important;
    font-weight: 800 !important;
    margin: 0 !important;
    background: linear-gradient(135deg, #FFFFFF 0%, #E2E8F0 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.025em !important;
}

.main-header-subtitle {
    font-size: 0.92rem !important;
    color: #94A3B8 !important;
    margin-top: 0.4rem !important;
    margin-bottom: 0 !important;
    line-height: 1.4 !important;
}

/* SaaS KPI Cards */
.saas-card {
    background: rgba(17, 24, 39, 0.85);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 1.15rem 1.25rem;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    margin-bottom: 1rem;
    transition: all 0.2s ease;
    backdrop-filter: blur(12px);
}

.saas-card:hover {
    border-color: rgba(59, 130, 246, 0.4);
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.35), 0 0 12px rgba(59, 130, 246, 0.15);
}

.saas-card-label {
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    color: #94A3B8 !important;
    margin-bottom: 0.35rem !important;
}

.saas-card-value {
    font-size: 1.7rem !important;
    font-weight: 800 !important;
    color: #F8FAFC !important;
    margin-bottom: 0.25rem !important;
    line-height: 1.2 !important;
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
}

.saas-card-sub {
    font-size: 0.78rem !important;
    color: #64748B !important;
}

/* Modern Badges */
.badge-green {
    background: rgba(16, 185, 129, 0.18);
    color: #34D399 !important;
    border: 1px solid rgba(52, 211, 153, 0.35);
    padding: 0.22rem 0.65rem;
    border-radius: 6px;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    display: inline-flex;
    align-items: center;
}

.badge-amber {
    background: rgba(245, 158, 11, 0.18);
    color: #FBBF24 !important;
    border: 1px solid rgba(251, 191, 36, 0.35);
    padding: 0.22rem 0.65rem;
    border-radius: 6px;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    display: inline-flex;
    align-items: center;
}

.badge-red {
    background: rgba(239, 68, 68, 0.18);
    color: #F87171 !important;
    border: 1px solid rgba(248, 113, 113, 0.35);
    padding: 0.22rem 0.65rem;
    border-radius: 6px;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    display: inline-flex;
    align-items: center;
}

.badge-blue {
    background: rgba(59, 130, 246, 0.18);
    color: #60A5FA !important;
    border: 1px solid rgba(96, 165, 250, 0.35);
    padding: 0.22rem 0.65rem;
    border-radius: 6px;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    display: inline-flex;
    align-items: center;
}

/* Callout Directive Box */
.directive-box {
    background: rgba(17, 24, 39, 0.9);
    border-left: 4px solid #3B82F6;
    border-top: 1px solid rgba(255, 255, 255, 0.06);
    border-right: 1px solid rgba(255, 255, 255, 0.06);
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    padding: 1.1rem 1.25rem;
    border-radius: 8px;
    margin-bottom: 1.2rem;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
}

.directive-title {
    font-weight: 700 !important;
    color: #F8FAFC !important;
    font-size: 0.95rem !important;
    margin-bottom: 0.35rem !important;
}

.directive-text {
    font-size: 0.88rem !important;
    color: #CBD5E1 !important;
    line-height: 1.45 !important;
}

/* Sidebar Custom Styling */
.sidebar-header-box {
    padding: 0.75rem 0.5rem;
    text-align: center;
}

.sidebar-title {
    color: #FFFFFF !important;
    font-size: 1.25rem !important;
    font-weight: 800 !important;
    margin: 0 !important;
    letter-spacing: -0.02em !important;
}

.sidebar-subtitle {
    color: #94A3B8 !important;
    font-size: 0.78rem !important;
    margin-top: 0.2rem !important;
}

.sidebar-info-box {
    background: rgba(17, 24, 39, 0.9);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    padding: 0.95rem;
    margin-top: 1.25rem;
    font-size: 0.82rem;
    color: #CBD5E1;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
}

/* Streamlit Button Styling */
div.stButton > button:first-child {
    border-radius: 8px !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em !important;
    transition: all 0.2s ease !important;
}

div.stButton > button:first-child:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3) !important;
}
</style>
"""


def apply_custom_styles() -> None:
    """Inject custom SaaS styles into Streamlit app."""
    st.markdown(SAAS_CSS, unsafe_allow_html=True)


def render_app_header(title: str, subtitle: str, badge_text: str = "Closed-Loop ML System") -> None:
    """Render top SaaS navigation header."""
    header_html = f"""
    <div class="main-header-container">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
            <div>
                <h1 class="main-header-title">{title}</h1>
                <p class="main-header-subtitle">{subtitle}</p>
            </div>
            <div>
                <span class="badge-blue">{badge_text}</span>
            </div>
        </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)


def render_kpi_card(label: str, value: str, subtext: str = "", badge: str = "", badge_type: str = "green") -> None:
    """Render a sleek modern glassmorphic SaaS metric card."""
    badge_html = f'<span class="badge-{badge_type}">{badge}</span>' if badge else ""
    card_html = f"""
    <div class="saas-card">
        <div class="saas-card-label">{label}</div>
        <div class="saas-card-value">
            <span>{value}</span>
            {badge_html}
        </div>
        <div class="saas-card-sub">{subtext}</div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def get_plotly_dark_layout(title: str = "", height: int = 340) -> dict:
    """Standardized Plotly dark layout config for cohesive chart themes."""
    return dict(
        title=dict(text=title, font=dict(color="#F8FAFC", size=14, family="Plus Jakarta Sans")),
        template="plotly_dark",
        paper_bgcolor="rgba(17, 24, 39, 0.7)",
        plot_bgcolor="rgba(17, 24, 39, 0.7)",
        font=dict(color="#94A3B8", family="Plus Jakarta Sans"),
        margin=dict(l=20, r=20, t=45 if title else 20, b=25),
        height=height,
        xaxis=dict(gridcolor="rgba(255, 255, 255, 0.06)", zerolinecolor="rgba(255, 255, 255, 0.08)"),
        yaxis=dict(gridcolor="rgba(255, 255, 255, 0.06)", zerolinecolor="rgba(255, 255, 255, 0.08)"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#CBD5E1", size=11),
            bgcolor="rgba(0,0,0,0)"
        )
    )
