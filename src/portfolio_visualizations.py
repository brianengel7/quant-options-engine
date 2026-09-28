from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt

from matplotlib.ticker import PercentFormatter
from matplotlib.ticker import (
    PercentFormatter,
    StrMethodFormatter
)


def plot_efficient_frontier(
    efficient_frontier,
    current_portfolio,
    minimum_volatility_portfolio,
    maximum_sharpe_portfolio,
    show=True,
    save_path=None
):
    required_columns = {
        "Expected Return",
        "Volatility",
        "Sharpe Ratio"
    }

    missing_columns = (
        required_columns
        - set(efficient_frontier.columns)
    )

    if missing_columns:
        raise ValueError(
            "Efficient frontier is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if efficient_frontier.empty:
        raise ValueError(
            "Efficient frontier cannot be empty."
        )

    figure, axis = plt.subplots(
        figsize=(10, 7)
    )

    frontier_scatter = axis.scatter(
        efficient_frontier["Volatility"],
        efficient_frontier["Expected Return"],
        c=efficient_frontier["Sharpe Ratio"],
        cmap="viridis",
        s=45,
        alpha=0.85,
        label="Efficient frontier"
    )

    axis.plot(
        efficient_frontier["Volatility"],
        efficient_frontier["Expected Return"],
        color="gray",
        linewidth=1,
        alpha=0.5
    )

    axis.scatter(
        current_portfolio["volatility"],
        current_portfolio["expected_return"],
        marker="o",
        s=160,
        color="royalblue",
        edgecolor="black",
        label="Current portfolio",
        zorder=5
    )

    axis.scatter(
        minimum_volatility_portfolio["volatility"],
        minimum_volatility_portfolio[
            "expected_return"
        ],
        marker="X",
        s=190,
        color="orange",
        edgecolor="black",
        label="Minimum volatility",
        zorder=5
    )

    axis.scatter(
        maximum_sharpe_portfolio["volatility"],
        maximum_sharpe_portfolio[
            "expected_return"
        ],
        marker="*",
        s=260,
        color="red",
        edgecolor="black",
        label="Maximum Sharpe",
        zorder=5
    )

    color_bar = figure.colorbar(
        frontier_scatter,
        ax=axis
    )

    color_bar.set_label("Sharpe ratio")

    axis.set_title(
        "Portfolio Efficient Frontier"
    )

    axis.set_xlabel(
        "Annualized volatility"
    )

    axis.set_ylabel(
        "Expected annual return"
    )

    axis.xaxis.set_major_formatter(
        PercentFormatter(1.0)
    )

    axis.yaxis.set_major_formatter(
        PercentFormatter(1.0)
    )

    axis.grid(alpha=0.25)
    axis.legend()

    figure.tight_layout()

    if save_path is not None:
        output_path = Path(save_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        figure.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight"
        )

    if show:
        plt.show()
    else:
        plt.close(figure)

    return figure, axis

def plot_strategy_growth(
    strategy_value_history,
    show=True,
    save_path=None
):
    if strategy_value_history.empty:
        raise ValueError(
            "Strategy value history cannot be empty."
        )

    figure, axis = plt.subplots(
        figsize=(11, 7)
    )

    for strategy in strategy_value_history.columns:
        axis.plot(
            strategy_value_history.index,
            strategy_value_history[strategy],
            linewidth=2,
            label=strategy
        )

    axis.set_title(
        "Portfolio Strategy Growth"
    )

    axis.set_xlabel("Date")
    axis.set_ylabel("Portfolio value")

    axis.yaxis.set_major_formatter(
        StrMethodFormatter("${x:,.0f}")
    )

    axis.grid(alpha=0.25)
    axis.legend()

    figure.tight_layout()

    if save_path is not None:
        output_path = Path(save_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        figure.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight"
        )

    if show:
        plt.show()
    else:
        plt.close(figure)

    return figure, axis

def plot_monte_carlo_distribution(
    simulated_returns,
    monte_carlo_var,
    monte_carlo_expected_shortfall,
    confidence_level=0.95,
    number_of_bins=60,
    show=True,
    save_path=None
):
    simulated_returns = np.asarray(
        simulated_returns,
        dtype=float
    ).reshape(-1)

    if simulated_returns.size == 0:
        raise ValueError(
            "Simulated returns cannot be empty."
        )

    if not np.isfinite(
        simulated_returns
    ).all():
        raise ValueError(
            "Simulated returns must be finite."
        )

    if monte_carlo_var < 0:
        raise ValueError(
            "Monte Carlo VaR cannot be negative."
        )

    if monte_carlo_expected_shortfall < 0:
        raise ValueError(
            "Expected Shortfall cannot be negative."
        )

    if not 0 < confidence_level < 1:
        raise ValueError(
            "Confidence level must be between 0 and 1."
        )

    if number_of_bins <= 0:
        raise ValueError(
            "Number of bins must be greater than zero."
        )

    var_return_threshold = (
        -monte_carlo_var
    )

    expected_shortfall_return = (
        -monte_carlo_expected_shortfall
    )

    figure, axis = plt.subplots(
        figsize=(11, 7)
    )

    counts, bin_edges, patches = axis.hist(
        simulated_returns,
        bins=number_of_bins,
        edgecolor="white",
        linewidth=0.5,
        alpha=0.85
    )

    for left_edge, patch in zip(
        bin_edges[:-1],
        patches
    ):
        if left_edge <= var_return_threshold:
            patch.set_facecolor("crimson")
        else:
            patch.set_facecolor("steelblue")

    axis.axvline(
        var_return_threshold,
        color="darkred",
        linestyle="--",
        linewidth=2,
        label=(
            f"{confidence_level:.0%} VaR: "
            f"{monte_carlo_var:.2%}"
        )
    )

    axis.axvline(
        expected_shortfall_return,
        color="black",
        linestyle=":",
        linewidth=2,
        label=(
            "Expected Shortfall: "
            f"{monte_carlo_expected_shortfall:.2%}"
        )
    )

    axis.set_title(
        "Correlated Monte Carlo Return Distribution"
    )

    axis.set_xlabel(
        "Simulated portfolio return"
    )

    axis.set_ylabel(
        "Simulation count"
    )

    axis.xaxis.set_major_formatter(
        PercentFormatter(1.0)
    )

    axis.grid(
        axis="y",
        alpha=0.25
    )

    axis.legend()

    figure.tight_layout()

    if save_path is not None:
        output_path = Path(save_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        figure.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight"
        )

    if show:
        plt.show()
    else:
        plt.close(figure)

    return figure, axis