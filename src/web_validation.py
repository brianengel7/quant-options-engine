from math import isclose

import pandas as pd


def prepare_portfolio_weights(
    portfolio_table,
    maximum_position_weight,
    minimum_assets=2,
    maximum_assets=10
):
    if not isinstance(portfolio_table, pd.DataFrame):
        raise TypeError(
            "Portfolio input must be a pandas DataFrame."
        )

    required_columns = {
        "Ticker",
        "Weight (%)"
    }

    if not required_columns.issubset(
        portfolio_table.columns
    ):
        raise ValueError(
            "Portfolio input must contain ticker and "
            "weight columns."
        )

    portfolio = portfolio_table[
        ["Ticker", "Weight (%)"]
    ].copy()

    portfolio["Ticker"] = (
        portfolio["Ticker"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    portfolio["Weight (%)"] = pd.to_numeric(
        portfolio["Weight (%)"],
        errors="coerce"
    )

    fully_blank_rows = (
        portfolio["Ticker"].eq("")
        & portfolio["Weight (%)"].isna()
    )

    portfolio = portfolio.loc[
        ~fully_blank_rows
    ].copy()

    if portfolio.empty:
        raise ValueError(
            "Enter at least two portfolio assets."
        )

    if portfolio["Ticker"].eq("").any():
        raise ValueError(
            "Every portfolio row must contain a ticker."
        )

    if portfolio["Weight (%)"].isna().any():
        raise ValueError(
            "Every portfolio row must contain a weight."
        )

    number_of_assets = len(portfolio)

    if number_of_assets < minimum_assets:
        raise ValueError(
            f"Enter at least {minimum_assets} assets."
        )

    if number_of_assets > maximum_assets:
        raise ValueError(
            f"Enter no more than {maximum_assets} assets."
        )

    if portfolio["Ticker"].duplicated().any():
        duplicate_tickers = (
            portfolio.loc[
                portfolio["Ticker"].duplicated(),
                "Ticker"
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            "Duplicate tickers are not allowed: "
            f"{duplicate_tickers}"
        )

    if (portfolio["Weight (%)"] <= 0).any():
        raise ValueError(
            "Every portfolio weight must be greater "
            "than zero."
        )

    total_weight = portfolio["Weight (%)"].sum()

    if not isclose(
        total_weight,
        100.0,
        abs_tol=0.01
    ):
        raise ValueError(
            "Portfolio weights must total 100%. "
            f"Current total: {total_weight:.2f}%."
        )

    if not 0 < maximum_position_weight <= 1:
        raise ValueError(
            "Maximum position weight must be between "
            "0% and 100%."
        )

    if (
        maximum_position_weight
        * number_of_assets
        < 1.0 - 1e-9
    ):
        minimum_feasible_weight = (
            1 / number_of_assets
        )

        raise ValueError(
            "The optimization constraint is infeasible. "
            "With "
            f"{number_of_assets} assets, the maximum "
            "position weight must be at least "
            f"{minimum_feasible_weight:.2%}."
        )

    decimal_weights = (
        portfolio["Weight (%)"] / 100
    )

    weights = dict(
        zip(
            portfolio["Ticker"],
            decimal_weights
        )
    )

    tickers = portfolio["Ticker"].tolist()

    return tickers, weights