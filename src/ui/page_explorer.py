"""
Page 7: Data Explorer & Historical Analytics Module.
Interactive dataset exploration, multi-dimensional filtering, exploratory distributions,
and processed data export.
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from src.ui.styles import get_plotly_dark_layout, render_app_header, render_kpi_card


def render_explorer_page(df_daily: pd.DataFrame, model_results, metadata) -> None:
    render_app_header(
        title="Interactive Data Explorer & Analytics",
        subtitle="Explore disaggregated day-level demand trends, categorical distributions, and climatic interactions."
    )

    df_work = df_daily.copy()
    df_work["date"] = pd.to_datetime(df_work["date"])

    # Sidebar / Top Filters
    st.markdown("#### Dynamic Multi-Dimensional Filters")
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)

    with f_col1:
        centers_avail = sorted(df_work["center_id"].unique())
        sel_centers = st.multiselect("Fulfilment Centers:", centers_avail, default=centers_avail)
    with f_col2:
        categories_avail = sorted(df_work["category"].dropna().unique())
        sel_cats = st.multiselect("Meal Categories:", categories_avail, default=categories_avail)
    with f_col3:
        cuisines_avail = sorted(df_work["cuisine"].dropna().unique())
        sel_cuisines = st.multiselect("Cuisines:", cuisines_avail, default=cuisines_avail)
    with f_col4:
        min_d = df_work["date"].min().date()
        max_d = df_work["date"].max().date()
        sel_dates = st.date_input("Date Range:", value=(min_d, max_d), min_value=min_d, max_value=max_d)

    # Apply Filters
    if isinstance(sel_dates, (tuple, list)) and len(sel_dates) == 2:
        date_start, date_end = pd.to_datetime(sel_dates[0]), pd.to_datetime(sel_dates[1])
    else:
        date_start, date_end = pd.to_datetime(min_d), pd.to_datetime(max_d)

    filtered_df = df_work[
        (df_work["center_id"].isin(sel_centers)) &
        (df_work["category"].isin(sel_cats)) &
        (df_work["cuisine"].isin(sel_cuisines)) &
        (df_work["date"] >= date_start) &
        (df_work["date"] <= date_end)
    ].copy()

    # Filter KPIs
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi_card("Filtered Records", f"{len(filtered_df):,}", f"Out of {len(df_work):,} total", badge="Slice", badge_type="blue")
    with k2:
        avg_dem = float(filtered_df["num_orders"].mean()) if not filtered_df.empty else 0.0
        render_kpi_card("Average Daily Demand", f"{avg_dem:.1f} Meals", "Mean orders per meal", badge="Volume", badge_type="green")
    with k3:
        avg_price = float(filtered_df["checkout_price"].mean()) if not filtered_df.empty else 0.0
        render_kpi_card("Average Price", f"₹{avg_price:.2f}", "Planned checkout price", badge="Price", badge_type="blue")
    with k4:
        promo_rate = (filtered_df["emailer_for_promotion"].mean() * 100.0) if not filtered_df.empty else 0.0
        render_kpi_card("Promotion Frequency", f"{promo_rate:.1f}%", "Active email campaigns", badge="Campaigns", badge_type="amber")

    st.markdown("---")

    # Visualizations: Category Boxplots and Day-of-Week Patterns
    v_col1, v_col2 = st.columns(2)

    with v_col1:
        st.markdown("#### Demand Volume by Meal Category")
        if not filtered_df.empty:
            fig_box = px.box(
                filtered_df,
                x="category",
                y="num_orders",
                color="category",
                labels={"category": "Category", "num_orders": "Daily Orders"},
                template="plotly_white"
            )
            fig_box.update_layout(
                **get_plotly_dark_layout(height=340),
                showlegend=False
            )
            st.plotly_chart(fig_box, use_container_width=True)

    with v_col2:
        st.markdown("#### Weekly Seasonality (Day-of-Week Pattern)")
        if not filtered_df.empty:
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            df_dow = filtered_df.copy()
            df_dow["day_name"] = df_dow["date"].dt.day_name()
            dow_summary = df_dow.groupby("day_name")["num_orders"].mean().reindex(day_order).reset_index()

            fig_dow = px.bar(
                dow_summary,
                x="day_name",
                y="num_orders",
                text="num_orders",
                labels={"day_name": "Day of Week", "num_orders": "Average Orders"},
                color="num_orders",
                color_continuous_scale="Blues"
            )
            fig_dow.update_traces(texttemplate='%{text:.0f}', textposition='outside')
            fig_dow.update_layout(
                **get_plotly_dark_layout(height=340),
                showlegend=False
            )
            st.plotly_chart(fig_dow, use_container_width=True)

    # Weather vs Demand Scatter
    st.markdown("#### Weather Impact: Rainfall vs Daily Food Demand")
    if not filtered_df.empty and "rainfall_mm" in filtered_df.columns:
        fig_rain = px.scatter(
            filtered_df.sample(min(500, len(filtered_df)), random_state=42),
            x="rainfall_mm",
            y="num_orders",
            color="weather_condition",
            trendline="ols",
            labels={"rainfall_mm": "Rainfall (mm)", "num_orders": "Daily Orders", "weather_condition": "Condition"}
        )
        fig_rain.update_layout(
            **get_plotly_dark_layout(height=320)
        )
        st.plotly_chart(fig_rain, use_container_width=True)

    # Interactive Filtered Data Table & Download
    st.markdown("#### Filtered Dataset Records Preview")
    st.dataframe(filtered_df.head(200), use_container_width=True)

    csv_bytes = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv_bytes,
        file_name="filtered_food_demand_data.csv",
        mime="text/csv"
    )
