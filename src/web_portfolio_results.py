import pandas as pd
import streamlit as st

from .web_portfolio_charts import (
    create_strategy_growth_chart,
    create_drawdown_chart,
    create_monte_carlo_distribution_chart,
    create_correlation_heatmap,
    create_benchmark_growth_chart,
    create_contribution_chart,
    create_efficient_frontier_chart,
    create_weight_comparison_chart,
    create_holdings_weight_chart
)


CHART_CONFIG = {
    "displaylogo": False,
    "responsive": True
}


def _display_strategy_comparison(
    strategy_comparison
):
    table = strategy_comparison.copy()

    formatters = {
        "Final Value": "${:,.2f}",
        "Total Return": "{:.2%}",
        "Annualized Return": "{:.2%}",
        "Annualized Volatility": "{:.2%}",
        "Sharpe Ratio": "{:.2f}",
        "Maximum Drawdown": "{:.2%}"
    }

    valid_formatters = {
        column: formatter
        for column, formatter in formatters.items()
        if column in table.columns
    }

    st.dataframe(
        table.style.format(valid_formatters),
        width="stretch"
    )


def _create_risk_table(risk_results):
    method_names = {
        "historical": "Historical",
        "parametric": "Parametric",
        "monte_carlo": "Monte Carlo"
    }

    rows = []

    for key, label in method_names.items():
        method = risk_results[key]

        rows.append({
            "Method": label,
            "VaR": method["var"],
            "Dollar VaR": method["dollar_var"],
            "Expected Shortfall": (
                method["expected_shortfall"]
            ),
            "Dollar Expected Shortfall": (
                method["dollar_expected_shortfall"]
            )
        })

    return pd.DataFrame(rows).set_index("Method")

def _render_benchmark_results(analysis):
    benchmark = analysis["benchmark"]
    ticker = benchmark["ticker"]

    st.subheader(
        f"Benchmark Analysis: {ticker}"
    )

    columns = st.columns(4)

    columns[0].metric(
        "Portfolio Beta",
        f"{benchmark['beta']:.2f}"
    )

    columns[1].metric(
        "Annualized Alpha",
        f"{benchmark['alpha']:.2%}"
    )

    columns[2].metric(
        "Tracking Error",
        f"{benchmark['tracking_error']:.2%}"
    )

    columns[3].metric(
        "Information Ratio",
        f"{benchmark['information_ratio']:.2f}"
    )

    benchmark_figure = (
        create_benchmark_growth_chart(
            benchmark["aligned_returns"]
        )
    )

    st.plotly_chart(
        benchmark_figure,
        width="stretch",
        config=CHART_CONFIG,
        key="benchmark_growth_chart"
    )

    st.caption(
        "Alpha adjusts performance for benchmark exposure. "
        "The information ratio measures active return "
        "relative to tracking error."
    )


def _render_contribution_results(analysis):
    contributions = analysis["contributions"]

    return_contributions = pd.Series(
        contributions["return"],
        dtype=float
    )

    volatility_contributions = pd.DataFrame(
        contributions["volatility"]
    )

    st.subheader("Asset Contribution Analysis")

    columns = st.columns(2)

    columns[0].metric(
        "Total Return Contribution",
        f"{return_contributions.sum():.2%}"
    )

    columns[1].metric(
        "Total Volatility Contribution",
        (
            f"{volatility_contributions[
                'Volatility Contribution'
            ].sum():.2%}"
        )
    )

    contribution_figure = create_contribution_chart(
        return_contributions,
        volatility_contributions
    )

    st.plotly_chart(
        contribution_figure,
        width="stretch",
        config=CHART_CONFIG,
        key="contribution_chart"
    )

    contribution_table = pd.DataFrame(
        index=volatility_contributions.index
    )

    contribution_table["Weight"] = (
        volatility_contributions["Weight"]
    )

    contribution_table["Return Contribution"] = (
        return_contributions
    )

    contribution_table["Volatility Contribution"] = (
        volatility_contributions[
            "Volatility Contribution"
        ]
    )

    contribution_table["Percentage of Risk"] = (
        volatility_contributions[
            "Percentage of Portfolio Risk"
        ]
    )

    st.dataframe(
        contribution_table.style.format({
            "Weight": "{:.2%}",
            "Return Contribution": "{:.2%}",
            "Volatility Contribution": "{:.2%}",
            "Percentage of Risk": "{:.2%}"
        }),
        width="stretch"
    )


def _render_optimization_results(analysis):
    optimization = analysis["optimization"]

    current = optimization["current"]
    minimum_volatility = (
        optimization["minimum_volatility"]
    )
    maximum_sharpe = (
        optimization["maximum_sharpe"]
    )

    st.subheader("Portfolio Optimization")

    summary = pd.DataFrame(
        {
            "Current Portfolio": {
                "Expected Return": (
                    current["expected_return"]
                ),
                "Volatility": current["volatility"],
                "Sharpe Ratio": (
                    current["sharpe_ratio"]
                )
            },
            "Minimum Volatility": {
                "Expected Return": (
                    minimum_volatility[
                        "expected_return"
                    ]
                ),
                "Volatility": (
                    minimum_volatility["volatility"]
                ),
                "Sharpe Ratio": (
                    minimum_volatility[
                        "sharpe_ratio"
                    ]
                )
            },
            "Maximum Sharpe": {
                "Expected Return": (
                    maximum_sharpe[
                        "expected_return"
                    ]
                ),
                "Volatility": (
                    maximum_sharpe["volatility"]
                ),
                "Sharpe Ratio": (
                    maximum_sharpe[
                        "sharpe_ratio"
                    ]
                )
            }
        }
    ).T

    st.dataframe(
        summary.style.format({
            "Expected Return": "{:.2%}",
            "Volatility": "{:.2%}",
            "Sharpe Ratio": "{:.2f}"
        }),
        width="stretch"
    )

    frontier_figure = create_efficient_frontier_chart(
        optimization["efficient_frontier"],
        current,
        minimum_volatility,
        maximum_sharpe
    )

    st.plotly_chart(
        frontier_figure,
        width="stretch",
        config=CHART_CONFIG,
        key="efficient_frontier_chart"
    )

    weight_figure = create_weight_comparison_chart(
        current,
        minimum_volatility,
        maximum_sharpe
    )

    st.plotly_chart(
        weight_figure,
        width="stretch",
        config=CHART_CONFIG,
        key="optimized_weight_chart"
    )

    st.caption(
        "Optimization is based on historical expected "
        "returns and covariance. It is not a forecast or "
        "investment recommendation."
    )


def _render_holdings_results(analysis):
    buy_and_hold = analysis["buy_and_hold"]

    st.subheader("Buy-and-Hold Portfolio")

    first_row = st.columns(3)

    first_row[0].metric(
        "Final Value",
        f"${buy_and_hold['final_value']:,.2f}"
    )

    first_row[1].metric(
        "Total Return",
        f"{buy_and_hold['total_return']:.2%}"
    )

    first_row[2].metric(
        "Annualized Return",
        f"{buy_and_hold['annualized_return']:.2%}"
    )

    second_row = st.columns(3)

    second_row[0].metric(
        "Annualized Volatility",
        (
            f"{buy_and_hold[
                'annualized_volatility'
            ]:.2%}"
        )
    )

    second_row[1].metric(
        "Sharpe Ratio",
        f"{buy_and_hold['sharpe_ratio']:.2f}"
    )

    second_row[2].metric(
        "Maximum Drawdown",
        f"{buy_and_hold['maximum_drawdown']:.2%}"
    )

    holdings_figure = create_holdings_weight_chart(
        analysis["inputs"]["weights"],
        buy_and_hold["ending_weights"]
    )

    st.plotly_chart(
        holdings_figure,
        width="stretch",
        config=CHART_CONFIG,
        key="holdings_weight_chart"
    )

    shares = pd.Series(
        buy_and_hold["shares"],
        name="Shares"
    )

    holdings_table = pd.concat(
        [
            shares,
            pd.Series(
                analysis["inputs"]["weights"],
                name="Initial Weight"
            ),
            pd.Series(
                buy_and_hold["ending_weights"],
                name="Ending Weight"
            )
        ],
        axis=1
    )

    st.subheader("Holdings Detail")

    st.dataframe(
        holdings_table.style.format({
            "Shares": "{:,.4f}",
            "Initial Weight": "{:.2%}",
            "Ending Weight": "{:.2%}"
        }),
        width="stretch"
    )

    st.caption(
        "Weight drift occurs because buy-and-hold positions "
        "are not periodically returned to their original "
        "target weights."
    )

def render_portfolio_results(analysis):
    (
        overview_tab,
        risk_tab,
        benchmark_tab,
        contribution_tab,
        optimization_tab,
        holdings_tab
    ) = st.tabs([
        "Overview",
        "Risk",
        "Benchmark",
        "Contributions",
        "Optimization",
        "Holdings"
    ])

    with overview_tab:
        performance = analysis["performance"]

        st.subheader("Performance Summary")

        columns = st.columns(4)

        columns[0].metric(
            "Current Value",
            f"${performance['current_value']:,.2f}"
        )

        columns[1].metric(
            "Total Return",
            f"{performance['total_return']:.2%}"
        )

        columns[2].metric(
            "Annualized Return",
            (
                f"{performance[
                    'annualized_return'
                ]:.2%}"
            )
        )

        columns[3].metric(
            "Annualized Volatility",
            (
                f"{performance[
                    'annualized_volatility'
                ]:.2%}"
            )
        )

        strategy_figure = (
            create_strategy_growth_chart(
                analysis["strategy_value_history"]
            )
        )

        st.plotly_chart(
            strategy_figure,
            width="stretch",
            config=CHART_CONFIG,
            key="strategy_growth_chart"
        )

        st.subheader("Strategy Comparison")

        _display_strategy_comparison(
            analysis["strategy_comparison"]
        )

    with risk_tab:
        risk_adjusted = analysis["risk_adjusted"]
        risk = analysis["risk"]

        st.subheader("Risk-Adjusted Performance")

        columns = st.columns(4)

        columns[0].metric(
            "Sharpe Ratio",
            f"{risk_adjusted['sharpe_ratio']:.2f}"
        )

        columns[1].metric(
            "Sortino Ratio",
            f"{risk_adjusted['sortino_ratio']:.2f}"
        )

        columns[2].metric(
            "Maximum Drawdown",
            (
                f"{risk_adjusted[
                    'maximum_drawdown'
                ]:.2%}"
            )
        )

        columns[3].metric(
            "Calmar Ratio",
            f"{risk_adjusted['calmar_ratio']:.2f}"
        )

        st.plotly_chart(
            create_drawdown_chart(
                risk_adjusted["drawdowns"]
            ),
            width="stretch",
            config=CHART_CONFIG,
            key="drawdown_chart"
        )

        st.subheader("Value at Risk")

        risk_table = _create_risk_table(risk)

        st.dataframe(
            risk_table.style.format({
                "VaR": "{:.2%}",
                "Dollar VaR": "${:,.2f}",
                "Expected Shortfall": "{:.2%}",
                "Dollar Expected Shortfall": "${:,.2f}"
            }),
            width="stretch"
        )

        monte_carlo = risk["monte_carlo"]

        st.plotly_chart(
            create_monte_carlo_distribution_chart(
                monte_carlo["simulated_returns"],
                monte_carlo["var"],
                monte_carlo["expected_shortfall"],
                analysis["inputs"][
                    "confidence_level"
                ]
            ),
            width="stretch",
            config=CHART_CONFIG,
            key="monte_carlo_distribution_chart"
        )

        st.plotly_chart(
            create_correlation_heatmap(
                analysis["matrices"]["correlation"]
            ),
            width="stretch",
            config=CHART_CONFIG,
            key="correlation_heatmap"
        )

    with benchmark_tab:
        _render_benchmark_results(analysis)

    with contribution_tab:
        _render_contribution_results(analysis)

    with optimization_tab:
        _render_optimization_results(analysis)

    with holdings_tab:
        _render_holdings_results(analysis)