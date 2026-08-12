"""Investor-facing Streamlit entry point built only from precomputed results."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.app_data import (
    fact_sheet_figure_key,
    figure_paths,
    historical_allocation_scenarios,
    load_app_data,
)


REPOSITORY_ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Systematic Fund Explorer",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root { --ink: #102a43; --teal: #0f766e; --amber: #d97706; --mist: #f4f7fa; }
    .stApp { background: linear-gradient(180deg, #f7fafc 0, #ffffff 240px); }
    .block-container { max-width: 1240px; padding-top: 4rem; padding-bottom: 4rem; }
    h1, h2, h3 { color: var(--ink); letter-spacing: -0.02em; }
    [data-testid="stMetric"] {
        background: #ffffff; border: 1px solid #d9e2ec; border-radius: 12px;
        padding: 0.9rem 1rem; box-shadow: 0 5px 18px rgba(16,42,67,0.05);
    }
    .hero-kicker { color: var(--teal); font-weight: 700; text-transform: uppercase;
        letter-spacing: .11em; font-size: .78rem; }
    .hero-copy { color: #486581; font-size: 1.08rem; max-width: 850px; }
    .evidence-note { background: #eef8f6; border-left: 4px solid var(--teal);
        border-radius: 6px; padding: .8rem 1rem; color: #243b53; margin: .6rem 0 1.2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading verified historical results…")
def _load_results() -> dict[str, pd.DataFrame]:
    return load_app_data(REPOSITORY_ROOT)


def _percent(value: float) -> str:
    return f"{value:.2%}"


def _currency(value: float) -> str:
    return f"${value:,.2f}"


def _display_performance(frame: pd.DataFrame) -> pd.DataFrame:
    table = frame[
        [
            "fund_name", "annualised_return", "annualised_volatility",
            "sharpe_ratio", "maximum_drawdown", "final_growth_of_1",
        ]
    ].copy()
    table.columns = [
        "Fund", "Annualised return", "Annualised volatility", "Sharpe ratio",
        "Maximum drawdown", "Growth of $1",
    ]
    return table.style.format(
        {
            "Annualised return": "{:.2%}",
            "Annualised volatility": "{:.2%}",
            "Sharpe ratio": "{:.3f}",
            "Maximum drawdown": "{:.2%}",
            "Growth of $1": "${:.3f}",
        }
    )


try:
    data = _load_results()
    figures = figure_paths(REPOSITORY_ROOT)
except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
    st.error("The verified precomputed app results could not be loaded.")
    st.code(str(error))
    st.stop()

family_order = ["Equity only", "Cryptocurrency only", "Combined equity + cryptocurrency"]
method_order = ["Equal Weight", "Risk Parity", "Minimum Variance"]
metrics = data["performance_metrics"].copy()
metrics["asset_family"] = pd.Categorical(metrics["asset_family"], family_order, ordered=True)
metrics["portfolio_method"] = pd.Categorical(metrics["portfolio_method"], method_order, ordered=True)
metrics = metrics.sort_values(["asset_family", "portfolio_method"]).reset_index(drop=True)
sample_start = metrics["evaluation_start_date"].min().date()
sample_end = metrics["evaluation_end_date"].max().date()

st.markdown('<div class="hero-kicker">Historical fund evidence</div>', unsafe_allow_html=True)
st.title("Systematic Fund Explorer")
st.markdown(
    '<div class="hero-copy">Compare nine completed systematic funds across equities, '
    'cryptocurrencies and a combined universe, inspect their fact sheets, test a hypothetical allocation, '
    'and explore equity headline sentiment.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<div class="evidence-note"><strong>Historical evidence only.</strong> Fund results '
    f'are walk-forward out-of-sample and descriptive from {sample_start} to {sample_end}. '
    'They are not forecasts, trading signals, investment advice, or a promise of future returns.</div>',
    unsafe_allow_html=True,
)

tab_compare, tab_facts, tab_allocation, tab_analytics, tab_sentiment, tab_notes = st.tabs(
    [
        "Compare funds", "Fact sheets", "Allocation lab", "Fund analytics",
        "Sentiment & fusion", "Method notes",
    ]
)

with tab_compare:
    st.subheader("Compare completed fund methods")
    st.write(
        "Each asset family is evaluated with Equal Weight, Risk Parity and long-only "
        "Minimum Variance. Choose a family for a like-for-like comparison on its native calendar."
    )
    selected_family = st.selectbox(
        "Asset family",
        family_order,
        format_func=lambda value: value.replace(" equity + cryptocurrency", ""),
        key="comparison_family",
    )
    comparison_metrics = metrics.loc[metrics["asset_family"].eq(selected_family)].copy()
    st.dataframe(_display_performance(comparison_metrics), width="stretch", hide_index=True)
    st.image(figures["growth"], width="stretch")
    st.caption(
        "The lines compound the saved daily OOS fund returns. Returns are gross of "
        "transaction costs, management fees and taxes."
    )
    with st.expander("Open turnover and transaction-cost sensitivity"):
        selected_cost = st.select_slider(
            "Illustrative one-way transaction cost",
            options=sorted(data["fund_cost_check"]["cost_rate_bps_one_way"].unique()),
            value=10.0,
            format_func=lambda value: f"{value:g} bps",
            key="fund_cost_rate",
        )
        cost_rows = data["fund_cost_check"].loc[
            data["fund_cost_check"]["asset_family"].eq(selected_family)
            & data["fund_cost_check"]["cost_rate_bps_one_way"].eq(selected_cost),
            [
                "fund_name", "net_annualised_return", "net_sharpe_ratio",
                "net_final_growth_of_1", "total_one_way_turnover",
            ],
        ].copy()
        cost_rows.columns = [
            "Fund", "Net annualised return", "Net Sharpe ratio",
            "Net growth of $1", "Total one-way turnover",
        ]
        st.dataframe(
            cost_rows.style.format(
                {
                    "Net annualised return": "{:.2%}",
                    "Net Sharpe ratio": "{:.3f}",
                    "Net growth of $1": "${:.3f}",
                    "Total one-way turnover": "{:.3f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

with tab_facts:
    st.subheader("Fund fact sheets and current target holdings")
    selected_fund = st.selectbox(
        "Choose a completed fund",
        metrics["fund_name"].tolist(),
        key="fact_sheet_fund",
    )
    fact = data["fact_sheet_summary"].set_index("fund_name").loc[selected_fund]
    with st.container(horizontal=True, gap="small"):
        st.metric("Growth of $1", f"${fact['final_growth_of_1']:.3f}", width=220)
        st.metric("Annualised return", _percent(fact["annualised_return"]), width=220)
        st.metric(
            "Annualised volatility",
            _percent(fact["annualised_volatility"]),
            width=220,
        )
        st.metric("Sharpe ratio", f"{fact['sharpe_ratio']:.3f}", width=220)
        st.metric("Maximum drawdown", _percent(fact["maximum_drawdown"]), width=220)

    fact_sheet_key = fact_sheet_figure_key(selected_fund)
    st.image(figures[fact_sheet_key], width="stretch")

    holdings = data["fact_sheet_holdings"].loc[
        data["fact_sheet_holdings"]["fund_name"].eq(selected_fund)
    ].sort_values("holding_rank")
    left, right = st.columns([1.05, 1.4])
    with left:
        st.markdown("#### Target mix by asset class")
        mix = (
            holdings.groupby("holding_asset_class", as_index=False)["target_weight"]
            .sum()
            .assign(target_weight=lambda frame: frame["target_weight"] * 100.0)
            .set_index("holding_asset_class")
        )
        st.bar_chart(
            mix,
            y="target_weight",
            x_label="",
            y_label="Target weight (%)",
            color="#0f766e",
            horizontal=True,
        )
        st.caption(
            f"Latest monthly target: {fact['latest_rebalance_date'].date()}. "
            "These are target weights, not live holdings."
        )
    with right:
        st.markdown("#### All latest target holdings")
        holdings_table = holdings[["ticker", "holding_asset_class", "target_weight"]].copy()
        holdings_table.columns = ["Ticker", "Asset class", "Target weight"]
        st.dataframe(
            holdings_table.style.format({"Target weight": "{:.3%}"}),
            width="stretch",
            hide_index=True,
            height=390,
        )

with tab_allocation:
    st.subheader("Historical allocation lab")
    st.write(
        "Choose any two completed funds, enter an amount and split it between them. The comparison "
        "rebases both paths at their first common saved observation and does not "
        "rebalance between funds afterward."
    )
    selector_one, selector_two = st.columns(2)
    fund_names = metrics["fund_name"].astype(str).tolist()
    with selector_one:
        first_fund = st.selectbox(
            "First fund", fund_names, index=fund_names.index("Combined Equal Weight")
        )
    second_options = [fund for fund in fund_names if fund != first_fund]
    default_second = (
        second_options.index("Combined Risk Parity")
        if "Combined Risk Parity" in second_options else 0
    )
    with selector_two:
        second_fund = st.selectbox("Second fund", second_options, index=default_second)
    amount_column, split_column = st.columns([1, 2])
    with amount_column:
        investment_amount = st.number_input(
            "Hypothetical starting amount ($)",
            min_value=100.0,
            max_value=10_000_000.0,
            value=10_000.0,
            step=500.0,
        )
    with split_column:
        first_weight_percent = st.slider(
            f"Allocation to {first_fund}",
            min_value=0,
            max_value=100,
            value=50,
            step=5,
            format="%d%%",
        )
        st.caption(
            f"{first_fund}: {first_weight_percent}% · "
            f"{second_fund}: {100 - first_weight_percent}%"
        )
    custom_weights = {
        first_fund: first_weight_percent / 100.0,
        second_fund: 1.0 - first_weight_percent / 100.0,
    }
    allocation_paths, allocation_summary = historical_allocation_scenarios(
        data["fund_returns"], investment_amount, custom_weights
    )
    summary_for_display = allocation_summary.copy()
    summary_for_display["Starting amount"] = summary_for_display["Starting amount"].map(_currency)
    summary_for_display["Historical ending value"] = summary_for_display[
        "Historical ending value"
    ].map(_currency)
    summary_for_display["Historical change"] = summary_for_display["Historical change"].map(_currency)
    summary_for_display["Historical return"] = summary_for_display["Historical return"].map(_percent)
    st.dataframe(summary_for_display, width="stretch", hide_index=True)
    st.line_chart(allocation_paths, x_label="Common observed date", y_label="Historical value ($)")
    st.caption(
        "Values are hypothetical and use the saved gross OOS growth paths from "
        f"{allocation_paths.index.min().date()} to {allocation_paths.index.max().date()}. "
        "Both selected paths start at the entered amount. No fees, taxes, cash flows, "
        "or fund-level rebalancing are applied."
    )

with tab_analytics:
    st.subheader("Growth, drawdown, risk-return and portfolio weights")
    growth_column, drawdown_column = st.columns(2)
    with growth_column:
        st.image(figures["growth"], width="stretch")
    with drawdown_column:
        st.image(figures["drawdown"], width="stretch")
    risk_column, weight_column = st.columns(2)
    with risk_column:
        st.image(figures["risk_return"], width="stretch")
    with weight_column:
        st.image(figures["weights"], width="stretch")
    st.caption(
        "All figures are generated from verified precomputed artifacts. The weight figure "
        "shows monthly Combined Risk Parity target weights; exact holdings for all nine funds are in the fact sheets."
    )

with tab_sentiment:
    st.subheader("Sector sentiment index and approved fusion comparison")
    sentiment_tab, fusion_tab, robustness_tab = st.tabs(
        ["Sector index", "Approved fusion", "Robustness checks"]
    )
    with sentiment_tab:
        st.write(
            "The index summarises supplied equity headlines only. It is descriptive and is "
            "not applied to cryptocurrencies. A missing sector-day remains missing rather than "
            "being treated as neutral."
        )
        sentiment = data["sector_sentiment"]
        sectors = sorted(sentiment["sector"].unique())
        selected_sectors = st.multiselect(
            "Sectors to display",
            sectors,
            default=[sector for sector in ("Tech", "Financials", "Energy") if sector in sectors],
        )
        if selected_sectors:
            sentiment_chart = (
                sentiment.loc[sentiment["sector"].isin(selected_sectors)]
                .pivot(index="date", columns="sector", values="sentiment_value")
                .sort_index()
            )
            st.line_chart(
                sentiment_chart,
                x_label="Observed equity trading date",
                y_label="VADER compound sector index (-1 to +1)",
            )
        else:
            st.info("Choose at least one sector to draw the historical index.")
        observed_rows = int(sentiment["sentiment_observed"].astype(bool).sum())
        missing_rows = len(sentiment) - observed_rows
        coverage_columns = st.columns(3)
        coverage_columns[0].metric("Equity sectors", f"{len(sectors)}")
        coverage_columns[1].metric("Observed sector-days", f"{observed_rows:,}")
        coverage_columns[2].metric("No-headline sector-days", f"{missing_rows:,}")
        with st.expander("Open the complete ten-sector figure"):
            st.image(figures["sentiment"], width="stretch")
        st.caption(
            "Plain VADER scores supplied headlines, not full articles. An observed score of zero "
            "is different from a day with no headlines."
        )

    with fusion_tab:
        st.markdown("#### Equity Equal Weight before and after sentiment fusion")
        st.write(
            "The approved fusion applies only to the 50-equity strategy. At each monthly "
            "decision it uses the preceding observed equity day's sector index with a 0.5 tilt."
        )
        fusion_comparison = data["fusion_comparison"].sort_values("uses_sentiment")
        fusion_table = fusion_comparison[
            [
                "strategy_name", "annualised_return", "annualised_volatility",
                "sharpe_ratio", "maximum_drawdown", "final_growth_of_1",
                "total_one_way_turnover",
            ]
        ].copy()
        fusion_table.columns = [
            "Strategy", "Annualised return", "Annualised volatility", "Sharpe ratio",
            "Maximum drawdown", "Growth of $1", "Total one-way turnover",
        ]
        st.dataframe(
            fusion_table.style.format(
                {
                    "Annualised return": "{:.2%}",
                    "Annualised volatility": "{:.2%}",
                    "Sharpe ratio": "{:.3f}",
                    "Maximum drawdown": "{:.2%}",
                    "Growth of $1": "${:.3f}",
                    "Total one-way turnover": "{:.3f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )
        fusion_paths = (
            data["fusion_returns"]
            .pivot(index="date", columns="variant", values="growth_of_1")
            .sort_index()
        )
        st.line_chart(
            fusion_paths,
            x_label="Observed equity trading date",
            y_label="Growth of $1",
        )
        base_row = fusion_comparison.loc[fusion_comparison["variant"].eq("Base")].iloc[0]
        fusion_row = fusion_comparison.loc[
            fusion_comparison["variant"].eq("Sentiment fusion")
        ].iloc[0]
        with st.container(horizontal=True, gap="small"):
            st.metric(
                "Return difference vs base",
                f"{(fusion_row['annualised_return'] - base_row['annualised_return']) * 100:.2f} pp",
                width=260,
            )
            st.metric(
                "Volatility difference vs base",
                f"{(fusion_row['annualised_volatility'] - base_row['annualised_volatility']) * 100:.2f} pp",
                width=260,
            )
            st.metric(
                "Sharpe difference vs base",
                f"{fusion_row['sharpe_ratio'] - base_row['sharpe_ratio']:.3f}",
                width=260,
            )
            st.metric(
                "Growth difference vs base",
                f"${fusion_row['final_growth_of_1'] - base_row['final_growth_of_1']:.3f}",
                width=260,
            )
        st.info(
            "In this historical sample, the approved fusion variant did not exceed the base "
            "strategy's annualised return or Sharpe ratio. This is a descriptive comparison, "
            "not a causal conclusion."
        )

    with robustness_tab:
        st.markdown("#### Precomputed turnover, cost and tilt-strength checks")
        st.write(
            "These tables expose the approved design's implementation sensitivity. They do not "
            "select a future cost rate or optimise a future sentiment multiplier."
        )
        selected_cost = st.select_slider(
            "Illustrative one-way transaction cost",
            options=sorted(data["fusion_cost_check"]["cost_rate_bps_one_way"].unique()),
            value=10.0,
            format_func=lambda value: f"{value:g} bps",
        )
        cost_rows = data["fusion_cost_check"].loc[
            data["fusion_cost_check"]["cost_rate_bps_one_way"].eq(selected_cost),
            [
                "strategy_name", "net_annualised_return", "net_sharpe_ratio",
                "net_final_growth_of_1", "total_one_way_turnover",
            ],
        ].copy()
        cost_rows.columns = [
            "Strategy", "Net annualised return", "Net Sharpe ratio",
            "Net growth of $1", "Total one-way turnover",
        ]
        st.dataframe(
            cost_rows.style.format(
                {
                    "Net annualised return": "{:.2%}",
                    "Net Sharpe ratio": "{:.3f}",
                    "Net growth of $1": "${:.3f}",
                    "Total one-way turnover": "{:.3f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )
        sensitivity = data["fusion_sensitivity"][[
            "tilt_strength", "approved_design", "annualised_return", "sharpe_ratio",
            "total_one_way_turnover",
        ]].copy()
        sensitivity.columns = [
            "Tilt strength", "Approved design", "Annualised return", "Sharpe ratio",
            "Total one-way turnover",
        ]
        st.dataframe(
            sensitivity.style.format(
                {
                    "Tilt strength": "{:.2f}",
                    "Annualised return": "{:.2%}",
                    "Sharpe ratio": "{:.3f}",
                    "Total one-way turnover": "{:.3f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

with tab_notes:
    st.subheader("Method notes and limits")
    settings = metrics.loc[metrics["fund_name"].astype(str).eq("Combined Equal Weight")].iloc[0]
    st.markdown(
        f"""
        - **Fund shelf:** nine completed funds: three asset families × three portfolio methods.
        - **Evidence period:** {sample_start} to {sample_end} for equity-calendar and combined funds; cryptocurrency-only funds retain their native seven-day calendar.
        - **Design:** historical walk-forward out-of-sample backtests using rolling 252-observation windows for equity and combined funds, and 365 observations for cryptocurrency-only funds.
        - **Rebalancing:** {settings['rebalance_rule']}.
        - **Constraints:** {settings['constraints']}.
        - **Annualisation:** 252 observations per year for equity and combined funds; 365 for cryptocurrency-only funds.
        - **Saved return basis:** zero annual risk-free rate and gross of transaction costs, management fees and taxes; an illustrative 10 bps one-way cost sensitivity is reported separately.
        - **Sentiment:** supplied equity headlines only; same-day or next-trading-day mapping in the build, and a one-observed-day lag for fusion decisions.
        """
    )
    st.warning(
        "This prototype does not provide forecasts, live prices, trading signals, personalised "
        "recommendations, or investment advice. It does not rerun models or the data pipeline."
    )
    st.caption(
        "All displayed evidence is loaded from committed, precomputed files under results/. "
        "The approved data sample ends no later than 2023-12-31."
    )
