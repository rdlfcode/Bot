import numpy as np
from scipy.stats import norm

def bs_price(rate, underlying_p, strike_p, maturity, impl_vol, type='C'):
    """
    Calculate option price for a call or put

    rate:           risk-free interest rate (3-month T-Bill)
    underlying_p:   underlying stock spot price
    strike_p:       strike price of option
    maturity:       time to maturity in days
    impl_vol:       implied volatility of the asset

    Assumptions:
        -European options that can only be exercised at expiration
        -No dividends paid out during option's life
        -No transaction/commission costs in buying the option
        -Normally distributed returns on the underlying asset
    """
    d1 = (np.log(underlying_p/strike_p) + (rate + impl_vol**2/2)*maturity) / (impl_vol*np.sqrt(maturity))
    d2 = d1 - impl_vol*np.sqrt(maturity)
    if type == 'C':
        price = underlying_p * norm.cdf(d1, 0, 1) - strike_p * np.exp(-rate*maturity) * norm.cdf(d2, 0, 1)
    elif type == 'P':
        price = strike_p * np.exp(-rate*maturity) * norm.cdf(-d2, 0, 1) - underlying_p * norm.cdf(-d1, 0, 1)
    return round(price, 3)

print(bs_price(.0472, 7.68, 5, 400, 64.25, 'C'))