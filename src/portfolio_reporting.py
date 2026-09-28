def _print_section(title):
    print(f"\n{title}")
    print("-" * len(title))


def _print_weights(weights):
    for ticker, weight in weights.items():
        print(f"{ticker}: {weight:.2%}")


def _print_risk_method(
    title,
    risk_results,
    confidence_level
):
    _print_section(title)

    print(
        f"VaR ({confidence_level:.0%}): "
        f"{risk_results['var']:.2%}"
    )

    print(
        f"Dollar VaR: "
        f"${risk_results['dollar_var']:,.2f}"
    )

    print(
        f"Expected Shortfall "
        f"({confidence_level:.0%}): "
        f"{risk_results['expected_shortfall']:.2%}"
    )

    print(
        f"Dollar Expected Shortfall: "
        f"${risk_results['dollar_expected_shortfall']:,.2f}"
    )


def print_portfolio_report(report):
    inputs = report["inputs"]

    _print_section("PORTFOLIO INPUTS")

    print(
        f"Tickers: "
        f"{', '.join(inputs['tickers'])}"
    )

    print(f"Start date: {inputs['start_date']}")

    print(
        f"Initial investment: "
        f"${inputs['initial_investment']:,.2f}"
    )

    print(
        f"Risk-free rate: "
        f"{inputs['risk_free_rate']:.2%}"
    )

    print(
        f"Maximum position weight: "
        f"{inputs['maximum_position_weight']:.0%}"
    )

    print("\nWeights:")
    _print_weights(inputs["weights"])

    previews = report["previews"]

    _print_section("PORTFOLIO RETURNS")
    print(previews["portfolio_returns"].head())

    _print_section("CUMULATIVE RETURNS")
    print(previews["cumulative_returns"].head())

    _print_section("PORTFOLIO VALUE HISTORY")
    print(previews["portfolio_value"].head())

    performance = report["performance"]

    _print_section("PERFORMANCE SUMMARY")

    print(
        f"Current portfolio value: "
        f"${performance['current_value']:,.2f}"
    )

    print(
        f"Total return: "
        f"{performance['total_return']:.2%}"
    )

    print(
        f"Annualized return: "
        f"{performance['annualized_return']:.2%}"
    )

    print(
        f"Annualized volatility: "
        f"{performance['annualized_volatility']:.2%}"
    )

    print(
        f"Covariance-based volatility: "
        f"{performance['covariance_volatility']:.2%}"
    )

    risk_adjusted = report["risk_adjusted"]

    _print_section("RISK-ADJUSTED PERFORMANCE")

    print(
        f"Sharpe ratio: "
        f"{risk_adjusted['sharpe_ratio']:.2f}"
    )

    print(
        f"Sortino ratio: "
        f"{risk_adjusted['sortino_ratio']:.2f}"
    )

    print(
        f"Maximum drawdown: "
        f"{risk_adjusted['maximum_drawdown']:.2%}"
    )

    print(
        f"Calmar ratio: "
        f"{risk_adjusted['calmar_ratio']:.2f}"
    )

    benchmark = report["benchmark"]

    _print_section("BENCHMARK ANALYSIS")

    print(f"Benchmark: {benchmark['ticker']}")

    print(
        f"Portfolio beta: "
        f"{benchmark['beta']:.2f}"
    )

    print(
        f"Annualized alpha: "
        f"{benchmark['alpha']:.2%}"
    )

    print(
        f"Tracking error: "
        f"{benchmark['tracking_error']:.2%}"
    )

    print(
        f"Information ratio: "
        f"{benchmark['information_ratio']:.2f}"
    )

    risk = report["risk"]

    _print_risk_method(
        "HISTORICAL RISK",
        risk["historical"],
        inputs["confidence_level"]
    )

    _print_risk_method(
        "PARAMETRIC RISK",
        risk["parametric"],
        inputs["confidence_level"]
    )

    _print_risk_method(
        "CORRELATED MONTE CARLO RISK",
        risk["monte_carlo"],
        inputs["confidence_level"]
    )

    matrices = report["matrices"]

    _print_section("ANNUALIZED COVARIANCE MATRIX")
    print(matrices["covariance"])

    _print_section("CORRELATION MATRIX")
    print(matrices["correlation"])

    _print_section(
        "ALIGNED PORTFOLIO AND BENCHMARK RETURNS"
    )

    print(
        benchmark["aligned_returns"].head()
    )

    contributions = report["contributions"]

    _print_section(
        "ANNUALIZED RETURN CONTRIBUTIONS"
    )

    print(
        contributions["return"].apply(
            lambda value: f"{value:.2%}"
        )
    )

    print(
        "Total arithmetic return contribution: "
        f"{contributions['return'].sum():.2%}"
    )

    formatted_risk_contributions = (
        contributions["volatility"].copy()
    )

    for column in (
        formatted_risk_contributions.columns
    ):
        formatted_risk_contributions[column] = (
            formatted_risk_contributions[column]
            .apply(lambda value: f"{value:.2%}")
        )

    _print_section(
        "ASSET-LEVEL RISK CONTRIBUTIONS"
    )

    print(formatted_risk_contributions)

    _print_section("MOST RECENT DRAWDOWNS")
    print(risk_adjusted["drawdowns"].tail())

    optimization = report["optimization"]

    _print_section("PORTFOLIO OPTIMIZATION")

    portfolio_labels = {
        "current": "Current portfolio",
        "minimum_volatility": (
            "Minimum-volatility portfolio"
        ),
        "maximum_sharpe": (
            "Maximum-Sharpe portfolio"
        )
    }

    for key, label in portfolio_labels.items():
        portfolio = optimization[key]

        print(f"\n{label} weights:")
        _print_weights(portfolio["weights"])

        print(
            f"Expected return: "
            f"{portfolio['expected_return']:.2%}"
        )

        print(
            f"Volatility: "
            f"{portfolio['volatility']:.2%}"
        )

        print(
            f"Sharpe ratio: "
            f"{portfolio['sharpe_ratio']:.2f}"
        )

    print("\nEfficient frontier:")

    print(
        optimization["efficient_frontier"][
            [
                "Expected Return",
                "Volatility",
                "Sharpe Ratio"
            ]
        ].head(10).to_string(
            index=False,
            formatters={
                "Expected Return": (
                    lambda value: f"{value:.2%}"
                ),
                "Volatility": (
                    lambda value: f"{value:.2%}"
                ),
                "Sharpe Ratio": (
                    lambda value: f"{value:.2f}"
                )
            }
        )
    )

    buy_and_hold = report["buy_and_hold"]

    _print_section("BUY-AND-HOLD PORTFOLIO")

    print("\nInitial shares:")

    for ticker, shares in (
        buy_and_hold["shares"].items()
    ):
        print(f"{ticker}: {shares:,.4f}")

    print(
        f"\nFinal portfolio value: "
        f"${buy_and_hold['final_value']:,.2f}"
    )

    print(
        f"Total return: "
        f"{buy_and_hold['total_return']:.2%}"
    )

    print(
        f"Annualized return: "
        f"{buy_and_hold['annualized_return']:.2%}"
    )

    print(
        f"Annualized volatility: "
        f"{buy_and_hold['annualized_volatility']:.2%}"
    )

    print(
        f"Sharpe ratio: "
        f"{buy_and_hold['sharpe_ratio']:.2f}"
    )

    print(
        f"Maximum drawdown: "
        f"{buy_and_hold['maximum_drawdown']:.2%}"
    )

    print("\nEnding weights:")
    _print_weights(
        buy_and_hold["ending_weights"]
    )

    strategy_comparison = report[
        "strategy_comparison"
    ]

    _print_section("STRATEGY COMPARISON")

    print(
        strategy_comparison.to_string(
            formatters={
                "Final Value": (
                    lambda value: f"${value:,.2f}"
                ),
                "Total Return": (
                    lambda value: f"{value:.2%}"
                ),
                "Annualized Return": (
                    lambda value: f"{value:.2%}"
                ),
                "Annualized Volatility": (
                    lambda value: f"{value:.2%}"
                ),
                "Sharpe Ratio": (
                    lambda value: f"{value:.2f}"
                ),
                "Maximum Drawdown": (
                    lambda value: f"{value:.2%}"
                )
            }
        )
    )

    charts = report["charts"]

    _print_section("GENERATED CHARTS")

    print(
        "Efficient frontier: "
        f"{charts['efficient_frontier']}"
    )

    print(
        "Strategy growth: "
        f"{charts['strategy_growth']}"
    )