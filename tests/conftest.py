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