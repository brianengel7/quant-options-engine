import pandas as pd
import plotly.graph_objects as go


BLUE = "#38BDF8"
ORANGE = "#F59E0B"
RED = "#FB7185"
GREEN = "#34D399"
PURPLE = "#A78BFA"


def _apply_layout(
    figure,
    title,
    xaxis_title,
    yaxis_title
):
    figure.update_layout(
        template="plotly_dark",
        height=430,
        title={
            "text": title,
            "x": 0.01,
            "xanchor": "left",
            "font": {
                "size": 20,
                "color": "#F8FAFC"
            }
        },
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="#0F172A",
        font={
            "family": "Arial, sans-serif",
            "size": 13,
            "color": "#CBD5E1"
        },
        hovermode="x unified",
        hoverlabel={
            "bgcolor": "#020617",
            "font_color": "#F8FAFC",
            "bordercolor": "#334155"
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "bgcolor": "rgba(0, 0, 0, 0)"
        },
        margin={
            "l": 30,
            "r": 25,
            "t": 75,
            "b": 35
        }
    )

    figure.update_xaxes(
        showgrid=False,
        showline=True,
        linecolor="#334155",
        tickcolor="#475569",
        title_font={
            "color": "#94A3B8"
        }
    )

    figure.update_yaxes(
        showgrid=True,
        gridcolor="#1E293B",
        zeroline=False,
        title_font={
            "color": "#94A3B8"
        }
    )

    return figure


def create_strategy_growth_chart(
    strategy_value_history
):
    history = (
        strategy_value_history
        .dropna(how="all")
        .sort_index()
    )

    figure = go.Figure()
    colors = [BLUE, ORANGE, GREEN]

    for index, column in enumerate(history.columns):
        series = history[column].dropna()

        figure.add_trace(
            go.Scatter(
                x=series.index,
                y=series,
                mode="lines",
                name=str(column),
                line={
                    "width": 2.5,
                    "color": colors[index % len(colors)]
                },
                hovertemplate=(
                    "%{x|%Y-%m-%d}<br>"
                    "Value: $%{y:,.2f}"
                    f"<extra>{column}</extra>"
                )
            )
        )

    _apply_layout(
        figure,
        title="Portfolio Growth",
        xaxis_title="Date",
        yaxis_title="Portfolio Value"
    )

    figure.update_yaxes(
        tickprefix="$",
        separatethousands=True
    )

    return figure


def create_drawdown_chart(drawdowns):
    drawdown_series = (
        pd.Series(drawdowns)
        .dropna()
        .sort_index()
    )

    figure = go.Figure(
        go.Scatter(
            x=drawdown_series.index,
            y=drawdown_series,
            mode="lines",
            name="Drawdown",
            line={
                "color": RED,
                "width": 2
            },
            fill="tozeroy",
            fillcolor="rgba(220, 38, 38, 0.20)",
            hovertemplate=(
                "%{x|%Y-%m-%d}<br>"
                "Drawdown: %{y:.2%}"
                "<extra></extra>"
            )
        )
    )

    _apply_layout(
        figure,
        title="Historical Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown"
    )

    figure.update_yaxes(tickformat=".1%")

    return figure


def create_monte_carlo_distribution_chart(
    simulated_returns,
    portfolio_var,
    expected_shortfall,
    confidence_level
):
    simulated_returns = (
        pd.Series(simulated_returns, dtype=float)
        .dropna()
    )

    figure = go.Figure(
        go.Histogram(
            x=simulated_returns,
            nbinsx=60,
            histnorm="probability",
            marker={
                "color": BLUE,
                "opacity": 0.80
            },
            hovertemplate=(
                "Return: %{x:.2%}<br>"
                "Probability: %{y:.2%}"
                "<extra></extra>"
            )
        )
    )

    figure.add_vline(
        x=-float(portfolio_var),
        line_color=RED,
        line_dash="dash",
        annotation_text=(
            f"{confidence_level:.0%} VaR"
        ),
        annotation_position="top left"
    )

    figure.add_vline(
        x=-float(expected_shortfall),
        line_color=ORANGE,
        line_dash="dot",
        annotation_text="Expected Shortfall",
        annotation_position="top right"
    )

    _apply_layout(
        figure,
        title="Monte Carlo Return Distribution",
        xaxis_title="Simulated One-Day Return",
        yaxis_title="Probability"
    )

    figure.update_xaxes(tickformat=".1%")
    figure.update_yaxes(tickformat=".1%")

    return figure


def create_correlation_heatmap(
    correlation_matrix
):
    matrix = pd.DataFrame(correlation_matrix)

    figure = go.Figure(
        go.Heatmap(
            z=matrix.to_numpy(),
            x=matrix.columns.astype(str),
            y=matrix.index.astype(str),
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            text=matrix.round(2).to_numpy(),
            texttemplate="%{text:.2f}",
            hovertemplate=(
                "%{y} / %{x}<br>"
                "Correlation: %{z:.2f}"
                "<extra></extra>"
            ),
            colorbar={
                "title": "Correlation"
            }
        )
    )

    _apply_layout(
        figure,
        title="Asset Correlation Matrix",
        xaxis_title="Asset",
        yaxis_title="Asset"
    )

    figure.update_layout(
        height=500,
        hovermode="closest"
    )

    figure.update_yaxes(autorange="reversed")

    return figure

def create_benchmark_growth_chart(
    aligned_returns
):
    returns = (
        pd.DataFrame(aligned_returns)
        .dropna()
        .sort_index()
    )

    cumulative_returns = (
        (1 + returns).cumprod() - 1
    )

    figure = go.Figure()

    colors = {
        "Portfolio": BLUE,
        "Benchmark": ORANGE
    }

    for column in cumulative_returns.columns:
        figure.add_trace(
            go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns[column],
                mode="lines",
                name=str(column),
                line={
                    "width": 2.5,
                    "color": colors.get(
                        column,
                        GREEN
                    )
                },
                hovertemplate=(
                    f"{column}<br>"
                    "%{x|%Y-%m-%d}<br>"
                    "Cumulative return: %{y:.2%}"
                    "<extra></extra>"
                )
            )
        )

    _apply_layout(
        figure,
        title="Portfolio vs Benchmark",
        xaxis_title="Date",
        yaxis_title="Cumulative Return"
    )

    figure.update_yaxes(tickformat=".1%")

    return figure


def create_contribution_chart(
    return_contributions,
    volatility_contributions
):
    return_series = pd.Series(
        return_contributions,
        dtype=float
    )

    volatility_table = pd.DataFrame(
        volatility_contributions
    )

    tickers = return_series.index.union(
        volatility_table.index
    )

    return_series = return_series.reindex(
        tickers,
        fill_value=0.0
    )

    risk_percentages = volatility_table[
        "Percentage of Portfolio Risk"
    ].reindex(
        tickers,
        fill_value=0.0
    )

    figure = go.Figure()

    figure.add_trace(
        go.Bar(
            x=tickers,
            y=return_series,
            name="Return Contribution",
            marker_color=BLUE,
            hovertemplate=(
                "%{x}<br>"
                "Return contribution: %{y:.2%}"
                "<extra></extra>"
            )
        )
    )

    figure.add_trace(
        go.Bar(
            x=tickers,
            y=risk_percentages,
            name="Percentage of Risk",
            marker_color=ORANGE,
            hovertemplate=(
                "%{x}<br>"
                "Portfolio risk: %{y:.2%}"
                "<extra></extra>"
            )
        )
    )

    _apply_layout(
        figure,
        title="Asset Return and Risk Contributions",
        xaxis_title="Asset",
        yaxis_title="Contribution"
    )

    figure.update_layout(barmode="group")
    figure.update_yaxes(tickformat=".1%")

    return figure


def create_efficient_frontier_chart(
    efficient_frontier,
    current_portfolio,
    minimum_volatility_portfolio,
    maximum_sharpe_portfolio
):
    frontier = pd.DataFrame(
        efficient_frontier
    )

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=frontier["Volatility"],
            y=frontier["Expected Return"],
            mode="lines+markers",
            name="Efficient Frontier",
            marker={
                "size": 7,
                "color": frontier["Sharpe Ratio"],
                "colorscale": "Viridis",
                "showscale": True,
                "colorbar": {
                    "title": "Sharpe"
                }
            },
            line={
                "color": BLUE,
                "width": 2
            },
            customdata=frontier[
                ["Sharpe Ratio"]
            ].to_numpy(),
            hovertemplate=(
                "Volatility: %{x:.2%}<br>"
                "Expected return: %{y:.2%}<br>"
                "Sharpe ratio: %{customdata[0]:.2f}"
                "<extra></extra>"
            )
        )
    )

    portfolio_points = [
        (
            "Current Portfolio",
            current_portfolio,
            "#F8FAFC",
            "circle"
        ),
        (
            "Minimum Volatility",
            minimum_volatility_portfolio,
            GREEN,
            "diamond"
        ),
        (
            "Maximum Sharpe",
            maximum_sharpe_portfolio,
            RED,
            "star"
        )
    ]

    for label, portfolio, color, symbol in portfolio_points:
        figure.add_trace(
            go.Scatter(
                x=[portfolio["volatility"]],
                y=[portfolio["expected_return"]],
                mode="markers",
                name=label,
                marker={
                    "size": 15,
                    "color": color,
                    "symbol": symbol,
                    "line": {
                        "color": "white",
                        "width": 1
                    }
                },
                customdata=[
                    [portfolio["sharpe_ratio"]]
                ],
                hovertemplate=(
                    f"{label}<br>"
                    "Volatility: %{x:.2%}<br>"
                    "Expected return: %{y:.2%}<br>"
                    "Sharpe ratio: "
                    "%{customdata[0]:.2f}"
                    "<extra></extra>"
                )
            )
        )

    _apply_layout(
        figure,
        title="Efficient Frontier",
        xaxis_title="Annualized Volatility",
        yaxis_title="Expected Annual Return"
    )

    figure.update_layout(hovermode="closest")
    figure.update_xaxes(tickformat=".1%")
    figure.update_yaxes(tickformat=".1%")

    return figure


def create_weight_comparison_chart(
    current_portfolio,
    minimum_volatility_portfolio,
    maximum_sharpe_portfolio
):
    portfolio_weights = {
        "Current": pd.Series(
            current_portfolio["weights"],
            dtype=float
        ),
        "Minimum Volatility": pd.Series(
            minimum_volatility_portfolio["weights"],
            dtype=float
        ),
        "Maximum Sharpe": pd.Series(
            maximum_sharpe_portfolio["weights"],
            dtype=float
        )
    }

    weight_table = pd.concat(
        portfolio_weights,
        axis=1
    ).fillna(0.0)

    figure = go.Figure()

    colors = [BLUE, GREEN, RED]

    for index, column in enumerate(
        weight_table.columns
    ):
        figure.add_trace(
            go.Bar(
                x=weight_table.index,
                y=weight_table[column],
                name=column,
                marker_color=colors[index],
                hovertemplate=(
                    "%{x}<br>"
                    "Weight: %{y:.2%}"
                    f"<extra>{column}</extra>"
                )
            )
        )

    _apply_layout(
        figure,
        title="Optimized Allocation Comparison",
        xaxis_title="Asset",
        yaxis_title="Portfolio Weight"
    )

    figure.update_layout(barmode="group")
    figure.update_yaxes(tickformat=".0%")

    return figure


def create_holdings_weight_chart(
    initial_weights,
    ending_weights
):
    weight_table = pd.concat(
        {
            "Initial Weight": pd.Series(
                initial_weights,
                dtype=float
            ),
            "Ending Weight": pd.Series(
                ending_weights,
                dtype=float
            )
        },
        axis=1
    ).fillna(0.0)

    figure = go.Figure()

    figure.add_trace(
        go.Bar(
            x=weight_table.index,
            y=weight_table["Initial Weight"],
            name="Initial Weight",
            marker_color=BLUE,
            hovertemplate=(
                "%{x}<br>"
                "Initial weight: %{y:.2%}"
                "<extra></extra>"
            )
        )
    )

    figure.add_trace(
        go.Bar(
            x=weight_table.index,
            y=weight_table["Ending Weight"],
            name="Ending Weight",
            marker_color=ORANGE,
            hovertemplate=(
                "%{x}<br>"
                "Ending weight: %{y:.2%}"
                "<extra></extra>"
            )
        )
    )

    _apply_layout(
        figure,
        title="Buy-and-Hold Weight Drift",
        xaxis_title="Asset",
        yaxis_title="Portfolio Weight"
    )

    figure.update_layout(barmode="group")
    figure.update_yaxes(tickformat=".0%")

    return figure