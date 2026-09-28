from datetime import date, timedelta

import pandas as pd
import streamlit as st

from src.portfolio_engine import (
    run_portfolio_analysis
)
from src.web_validation import (
    prepare_portfolio_weights
)
from src.web_portfolio_results import (
    render_portfolio_results
)

from src.web_style import apply_app_style


st.set_page_config(
    page_title="Portfolio Analysis",
    page_icon="📊",
    layout="wide"
)

apply_app_style()


DEFAULT_PORTFOLIO = pd.DataFrame({
    "Ticker": [
        "AAPL",
        "MSFT",
        "JPM"
    ],
    "Weight (%)": [
        40.0,
        35.0,
        25.0
    ]
})


st.markdown(
    ":blue-badge[PORTFOLIO ANALYTICS] "
    ":green-badge[LIVE MARKET DATA] "
    ":gray-badge[HISTORICAL ANALYSIS]"
)

st.title("Portfolio Risk & Optimization")

st.markdown(
    """
    Build a custom equity portfolio and evaluate its historical
    performance, downside risk, benchmark exposure, asset-level
    contributions, and optimized allocations.
    """
)

st.caption(
    "Performance • Risk • Benchmarking • Monte Carlo • "
    "Efficient Frontier • Holdings Analysis"
)

with st.form("portfolio_analysis_form"):
    st.subheader("Portfolio Holdings")

    portfolio_table = st.data_editor(
        DEFAULT_PORTFOLIO,
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        column_config={
            "Ticker": st.column_config.TextColumn(
                "Ticker",
                help="Exchange-listed ticker symbol"
            ),
            "Weight (%)": (
                st.column_config.NumberColumn(
                    "Weight (%)",
                    help=(
                        "Portfolio allocation as a "
                        "percentage"
                    ),
                    min_value=0.0,
                    max_value=100.0,
                    step=1.0,
                    format="%.2f"
                )
            )
        }
    )

    st.caption(
        "Add or delete rows as needed. Portfolio weights "
        "must total 100%."
    )

    st.subheader("Analysis Settings")

    date_column_1, date_column_2 = st.columns(2)

    with date_column_1:
        start_date = st.date_input(
            "Start date",
            value=(
                date.today()
                - timedelta(days=5 * 365)
            ),
            max_value=date.today()
        )

    with date_column_2:
        end_date = st.date_input(
            "End date",
            value=date.today(),
            max_value=date.today()
        )

    settings_column_1, settings_column_2 = (
        st.columns(2)
    )

    with settings_column_1:
        initial_investment = st.number_input(
            "Initial investment ($)",
            min_value=1_000.0,
            value=100_000.0,
            step=1_000.0,
            format="%.2f"
        )

        benchmark_ticker = st.text_input(
            "Benchmark ticker",
            value="SPY",
            help=(
                "The market benchmark used for beta, "
                "alpha, tracking error, and information "
                "ratio."
            )
        )

        risk_free_rate_percent = st.number_input(
            "Annual risk-free rate (%)",
            min_value=-99.0,
            max_value=50.0,
            value=4.0,
            step=0.25,
            format="%.2f"
        )

    with settings_column_2:
        confidence_level = st.selectbox(
            "VaR confidence level",
            options=[
                0.90,
                0.95,
                0.99
            ],
            index=1,
            format_func=lambda value: (
                f"{value:.0%}"
            )
        )

        maximum_position_percent = st.number_input(
            "Maximum optimized position (%)",
            min_value=1.0,
            max_value=100.0,
            value=60.0,
            step=5.0,
            format="%.2f",
            help=(
                "Maximum allocation permitted for any "
                "asset in optimized portfolios."
            )
        )

        number_of_simulations = st.selectbox(
            "Monte Carlo simulations",
            options=[
                5_000,
                10_000,
                25_000
            ],
            index=1,
            format_func=lambda value: f"{value:,}"
        )

    submitted = st.form_submit_button(
        "Run Portfolio Analysis",
        type="primary"
    )


if submitted:
    st.session_state.pop(
        "portfolio_analysis",
        None
    )

    try:
        benchmark_ticker = (
            benchmark_ticker.strip().upper()
        )

        if not benchmark_ticker:
            raise ValueError(
                "Benchmark ticker cannot be empty."
            )

        if start_date >= end_date:
            raise ValueError(
                "The start date must be earlier than "
                "the end date."
            )

        if (end_date - start_date).days < 90:
            raise ValueError(
                "Select at least 90 calendar days of "
                "historical data."
            )

        maximum_position_weight = (
            maximum_position_percent / 100
        )

        tickers, weights = (
            prepare_portfolio_weights(
                portfolio_table,
                maximum_position_weight
            )
        )

        # yfinance treats its end date as exclusive.
        download_end_date = (
            end_date + timedelta(days=1)
        )

        with st.spinner(
            "Downloading market data and running "
            "portfolio analysis..."
        ):
            analysis = run_portfolio_analysis(
                tickers=tickers,
                weights=weights,
                start_date=start_date.isoformat(),
                end_date=(
                    download_end_date.isoformat()
                ),
                initial_investment=(
                    initial_investment
                ),
                benchmark_ticker=(
                    benchmark_ticker
                ),
                risk_free_rate=(
                    risk_free_rate_percent / 100
                ),
                confidence_level=confidence_level,
                maximum_position_weight=(
                    maximum_position_weight
                ),
                number_of_simulations=(
                    number_of_simulations
                ),
                monte_carlo_horizon_days=1,
                monte_carlo_rebalancing_mode=(
                    "daily"
                ),
                frontier_points=50,
                trading_days=252,
                random_seed=42
            )

        st.session_state[
            "portfolio_analysis"
        ] = analysis

        st.session_state[
            "submitted_weights"
        ] = weights

        st.success(
            "Portfolio analysis completed."
        )

    except (TypeError, ValueError) as error:
        st.error(str(error))

    except Exception as error:
        st.error(
            "The analysis could not be completed. "
            f"Details: {error}"
        )


if "portfolio_analysis" in st.session_state:
    st.divider()

    render_portfolio_results(
        st.session_state["portfolio_analysis"]
    )