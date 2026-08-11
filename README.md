# quant-options-engine
Options Pricing and Risk Analytics Engine

## Features

- European call and put pricing using the Black–Scholes model
- Greeks: delta, gamma, vega, theta, and rho
- Live stock and option-chain data retrieving
- Historical volatility calculated from annualized log returns
- Implied-volatility calculation using the bisection method
- Call and put volatility smile
- Validation for option prices
- Automated testing with pytest

## Implied Volatility Methodology

The engine calculates implied volatility by finding the volatility input that makes the model price match the observed market price.

Because the Black–Scholes equation cannot be rearranged to solve directly for volatility, the engine uses the bisection method:

1. Establish a volatility search interval.
2. Calculate the midpoint volatility.
3. Use Black–Scholes to calculate an option price at that volatility.
4. Compare the model price with the observed market price.
5. Narrow the search interval based on the comparison.
6. Repeat until the price difference is within the specified tolerance.

The implementation supports both calls and puts and rejects prices outside their European no-arbitrage bounds before beginning the numerical search.

## Volatility Smile

The engine builds call and put volatility smiles from live option-chain data by:

1. Removing contracts with missing or invalid bid-ask quotes.
2. Filtering contracts to a configurable strike range around the current stock price.
3. Calculating each contract's market midpoint:

  Midpoint = (Bid + Ask)/2

4. Using the midpoint as the market price in the implied-volatility solver.
5. Skipping contracts whose prices are outside no-arbitrage bounds or whose implied volatility cannot be calculated.
6. Sorting the valid results by strike price.
7. Plotting implied volatility against strike price for calls and puts.

The strike-range filter focuses the analysis on contracts near the current stock price, where quotes are generally more relevant and liquid.

## Testing

The project uses `pytest` to test the implied-volatility implementation.

The test suite verifies that:

- Call and put solvers recover a known volatility from a Black–Scholes price.
- Non-positive option prices are rejected.
- Prices at or above the applicable upper bounds are rejected.
- Prices below the European no-arbitrage lower bounds are rejected.
- Validation raises the expected exception type and error message.

To run the complete test suite:

python -m pytest -v


The recovery tests for correctness under controlled inputs, while the validation tests impossible inputs are rejected before the solver runs.
