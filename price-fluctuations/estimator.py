import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

from scipy.optimize import fmin
from scipy.special import erf


class Modes:

    NORMAL = "normal"
    EMPIRICAL = "empirical"


def estimate_prices(prices, lookforward, probability, type=Modes.NORMAL, lookback=63):

    returns = preprocess_data(prices=prices, lookback=lookback)

    # Last seen price
    last = prices.values[-1]

    if type == Modes.NORMAL:
        price_min, price_max = estimate_normal(
            price=last,
            returns=returns,
            probability=probability,
            lookforward=lookforward,
        )

    return price_min, price_max


def preprocess_data(prices, lookback):
    """Transform prices into logarithmic returns.

    R_t := log(1 + r_t) = log(P_t / P_{t-1})

    where log is the natural logarithm.

    Parameters
    ----------
    prices : pandas.Series
        [description]
    lookback : int
        How many returns to keep since the last price

    Returns
    -------
    returns : pandas.Series
        [description]
    """

    # Enforce order
    prices = prices.sort_index(ascending=True)

    # Compute log returns
    returns = prices.pct_change().add(1.0)
    returns = returns.apply(np.log)
    returns = returns.tail(lookback)

    return returns


def estimate_normal(price, returns, probability, lookforward):
    """Estimate price interval with normal distribution.

    Parameters
    ----------
    price : float
        Last price.
    returns : pandas.Series
        Log-returns.
    probability : float
        Confidence interval.
    lookforward : int
        How many days in the future.

    Returns
    -------
    price_min : float
    price_max : float
        Minimum and maximum estimated prices.
    """

    mean = np.mean(returns)
    sigma = np.std(returns)

    # -------------------------------------------------------------------------
    # Compute sum of normal distributions parameters
    mean_sum = lookforward * mean
    sigma_sum = np.sqrt(lookforward) * sigma

    # -------------------------------------------------------------------------
    # Find sigma width according to probability
    func = lambda x: (probability - erf(x / np.sqrt(2))) ** 2.0

    num = fmin(func=func, x0=0.0)
    num = num[0]

    # -------------------------------------------------------------------------
    # Compute minimum and maximum total return
    minimum = mean_sum - num * sigma_sum
    maximum = mean_sum + num * sigma_sum

    # -------------------------------------------------------------------------
    # Compute final prices by accumulation
    price_min = price * np.exp(minimum)
    price_max = price * np.exp(maximum)

    return price_min, price_max


if __name__ == "__main__":

    prices = pd.read_csv("eurgbp.csv", index_col="Date", parse_dates=True, squeeze=True)

    price_min, price_max = estimate_prices(
        prices=prices, lookforward=5, probability=0.954499736104
    )
