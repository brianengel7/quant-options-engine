import streamlit as st
from src.web_style import apply_app_style


st.set_page_config(
    page_title="Quantitative Analytics Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_app_style()


st.title("Quantitative Portfolio & Options Analytics")

st.markdown(
    """
    Analyze custom portfolios, measure historical risk,
    compare performance with a benchmark, explore optimized
    allocations, and value equity options using multiple
    quantitative models.
    """
)

st.divider()

performance_column, risk_column, options_column = st.columns(3)

with performance_column:
    st.subheader("Portfolio Analytics")
    st.markdown(
        """
        - Performance and volatility
        - Benchmark attribution
        - Asset-level contributions
        - Buy-and-hold comparison
        """
    )

with risk_column:
    st.subheader("Risk & Optimization")
    st.markdown(
        """
        - Historical, parametric, and Monte Carlo VaR
        - Expected shortfall
        - Minimum-volatility portfolio
        - Maximum-Sharpe portfolio
        """
    )

with options_column:
    st.subheader("Options Analytics")
    st.markdown(
        """
        - Black–Scholes valuation
        - Binomial-tree pricing
        - Monte Carlo pricing
        - Greeks and implied volatility
        """
    )

st.divider()

st.subheader("Project Methodology")

st.markdown(
    """
    The platform uses historical market data to estimate
    portfolio performance, covariance, benchmark exposure,
    downside risk, and constrained optimized allocations.

    Options can be evaluated using closed-form, lattice, and
    simulation-based pricing methods.
    """
)

st.info(
    "Use the navigation pages to configure a portfolio or "
    "evaluate an option. Interactive analysis pages are being "
    "added in the next development steps."
)

st.caption(
    "For educational and demonstration purposes only. "
    "Results are not investment advice."
)