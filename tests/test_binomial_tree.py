import pytest

from src.binomial_tree import calculate_binomial_price
from src.black_scholes import (
    calculate_call_price,
    calculate_put_price,
)


def test_european_call_converges_to_black_scholes():
    S = 100
    K = 100
    T = 1
    r = 0.05
    sigma = 0.20

    binomial_price = calculate_binomial_price(
        S=S,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        steps=1000,
        type="call",
        style="european",
    )

    black_scholes_price = calculate_call_price(
        S=S,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
    )

    assert binomial_price == pytest.approx(
        black_scholes_price,
        abs=0.01,
    )


def test_european_put_converges_to_black_scholes():
    S = 100
    K = 100
    T = 1
    r = 0.05
    sigma = 0.20

    binomial_price = calculate_binomial_price(
        S=S,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        steps=1000,
        type="put",
        style="european",
    )

    black_scholes_price = calculate_put_price(
        S=S,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
    )

    assert binomial_price == pytest.approx(
        black_scholes_price,
        abs=0.01,
    )


def test_american_put_early_exercise_adds_value():
    common_inputs = {
        "S": 80,
        "K": 100,
        "T": 1,
        "r": 0.05,
        "sigma": 0.20,
        "steps": 100,
        "type": "put",
    }

    american_price = calculate_binomial_price(
        **common_inputs,
        style="american",
    )

    european_price = calculate_binomial_price(
        **common_inputs,
        style="european",
    )

    assert american_price > european_price
    assert american_price == pytest.approx(20.00, abs=0.01)


def test_american_call_equals_european_call_without_dividends():
    common_inputs = {
        "S": 100,
        "K": 100,
        "T": 1,
        "r": 0.05,
        "sigma": 0.20,
        "steps": 500,
        "type": "call",
    }

    american_price = calculate_binomial_price(
        **common_inputs,
        style="american",
    )

    european_price = calculate_binomial_price(
        **common_inputs,
        style="european",
    )

    assert american_price == pytest.approx(
        european_price,
        abs=0.0001,
    )


def test_american_option_is_not_below_intrinsic_value():
    option_price = calculate_binomial_price(
        S=80,
        K=100,
        T=1,
        r=0.05,
        sigma=0.20,
        steps=100,
        type="put",
        style="american",
    )

    intrinsic_value = max(100 - 80, 0)

    assert option_price >= intrinsic_value


@pytest.mark.parametrize(
    "parameter_name, invalid_value, expected_message",
    [
        ("S", 0, "Stock price must be greater than zero."),
        ("K", 0, "Strike price must be greater than zero."),
        ("T", 0, "Time to expiration must be greater than zero."),
        ("sigma", 0, "Volatility must be greater than zero."),
    ],
)
def test_nonpositive_inputs_are_rejected(
    parameter_name,
    invalid_value,
    expected_message,
):
    inputs = {
        "S": 100,
        "K": 100,
        "T": 1,
        "r": 0.05,
        "sigma": 0.20,
        "steps": 100,
        "type": "call",
        "style": "american",
    }

    inputs[parameter_name] = invalid_value

    with pytest.raises(ValueError, match=expected_message):
        calculate_binomial_price(**inputs)


@pytest.mark.parametrize(
    "invalid_steps",
    [0, -1, 2.5, True],
)
def test_invalid_steps_are_rejected(invalid_steps):
    with pytest.raises(
        ValueError,
        match="Steps must be a positive integer.",
    ):
        calculate_binomial_price(
            S=100,
            K=100,
            T=1,
            r=0.05,
            sigma=0.20,
            steps=invalid_steps,
            type="call",
            style="american",
        )


def test_invalid_option_type_is_rejected():
    with pytest.raises(
        ValueError,
        match="Option type must be either 'call' or 'put'.",
    ):
        calculate_binomial_price(
            S=100,
            K=100,
            T=1,
            r=0.05,
            sigma=0.20,
            steps=100,
            type="invalid",
            style="american",
        )


def test_invalid_exercise_style_is_rejected():
    with pytest.raises(
        ValueError,
        match="Exercise style must be either 'american' or 'european'.",
    ):
        calculate_binomial_price(
            S=100,
            K=100,
            T=1,
            r=0.05,
            sigma=0.20,
            steps=100,
            type="call",
            style="invalid",
        )