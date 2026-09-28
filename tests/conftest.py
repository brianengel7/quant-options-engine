import pandas as pd
import pytest


@pytest.fixture
def sample_asset_returns():
    dates = pd.date_range(
        "2026-01-01",
        periods=6,
        freq="B"
    )

    return pd.DataFrame(
        {
            "AAPL": [
                0.010,
                -0.020,
                0.015,
                0.005,
                -0.010,
                0.012
            ],
            "JPM": [
                0.005,
                -0.010,
                0.010,
                0.002,
                -0.004,
                0.006
            ],
            "MSFT": [
                0.015,
                -0.025,
                0.020,
                0.006,
                -0.012,
                0.014
            ]
        },
        index=dates
    )


@pytest.fixture
def sample_weights():
    return {
        "AAPL": 0.40,
        "JPM": 0.25,
        "MSFT": 0.35
    }

@pytest.fixture
def sample_prices():
    dates = pd.date_range(
        "2026-01-01",
        periods=4,
        freq="B"
    )

    return pd.DataFrame(
        {
            "AAPL": [100.0, 110.0, 105.0, 120.0],
            "JPM": [50.0, 51.0, 53.0, 54.0],
            "MSFT": [200.0, 198.0, 210.0, 220.0]
        },
        index=dates
    )


@pytest.fixture
def sample_benchmark_returns(
    sample_asset_returns
):
    return pd.Series(
        [
            0.008,
            -0.015,
            0.012,
            0.004,
            -0.008,
            0.010
        ],
        index=sample_asset_returns.index,
        name="SPY"
    )