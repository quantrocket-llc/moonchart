# Copyright 2017 QuantRocket LLC - All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pandas as pd
from .exceptions import InsufficientData
from .utils import (
    trim_outliers as trim_outliers_func,
    get_sharpe,
    get_rolling_sharpe,
    get_cagr,
    get_cum_returns,
    get_drawdowns)

from quantrocket.moonshot import read_moonshot_csv, intraday_to_daily

class DailyPerformance(object):
    """
    Class for storing performance attributes and calculating derived statistics.

    Parameters
    ----------
    returns : DataFrame, required
        a Dataframe of pct returns

    pnl : DataFrame, optional
        a DataFrame of pnl

    net_exposures : DataFrame, optional
        a Dataframe of net (hedged) exposure

    abs_exposures : DataFrame, optional
        a Dataframe of absolute exposure (ignoring hedging)

    total_holdings : DataFrame, optional
        a Dataframe of the number of holdings

    trades : DataFrame, optional
        a DataFrame of trades, that is, changes to positions

    commissions : DataFrame, optional
        a DataFrame of commissions, in the base currency

    commissions_pct : DataFrame, optional
        a DataFrame of commissions, in percentages

    slippages : DataFrame, optional
        a DataFrame of slippages, in percentages

    benchmark : Series, optional
        a Series of prices for a benchmark

    riskfree : float, optional
        the riskfree rate (default 0)

    compound : bool
         True for compound/geometric returns, False for arithmetic returns (default True)

    rolling_sharpe_window : int, optional
        compute rolling Sharpe over this many periods (default 200)

    trim_outliers: int or float, optional
        discard returns that are more than this many standard deviations from the mean
    """

    def __init__(
        self,
        returns,
        pnl=None,
        net_exposures=None,
        abs_exposures=None,
        total_holdings=None,
        trades=None,
        commissions=None,
        commissions_pct=None,
        slippages=None,
        benchmark=None,
        riskfree=0,
        compound=True,
        rolling_sharpe_window=200,
        trim_outliers=None
        ):

        self.returns = returns
        if len(self.returns.index) < 2:
            raise InsufficientData(
                "Moonchart needs at least 2 dates to analyze performance, "
                "but returns DataFrame has length {0}".format(len(self.returns.index)))
        self._trim_outliers = trim_outliers
        if trim_outliers:
            self.returns = trim_outliers_func(returns, z_score=trim_outliers)
        self.pnl = pnl
        self.net_exposures = net_exposures
        self.abs_exposures = abs_exposures
        self.total_holdings = total_holdings
        self.trades = trades
        self.commissions = commissions
        self.commissions_pct = commissions_pct
        self.slippages = slippages
        self.riskfree = riskfree
        self.compound = compound
        self.rolling_sharpe_window = rolling_sharpe_window
        self._benchmark_prices = benchmark
        self._benchmark_returns = None
        self._cum_returns = None
        self._sharpe = None
        self._rolling_sharpe = None
        self._cagr = None
        self._drawdowns = None
        self._max_drawdown = None
        self._cum_pnl = None

    @classmethod
    def _from_moonshot(cls, results,
                       trim_outliers=None,
                       riskfree=0,
                       compound=True,
                       rolling_sharpe_window=200):
        """
        Creates a DailyPerformance instance from a moonshot backtest results DataFrame.
        """
        if "Time" in results.index.names:
            results = intraday_to_daily(results)

        fields = results.index.get_level_values("Field").unique()
        kwargs = {}
        kwargs["returns"] = results.loc["Return"]
        if "Position" in fields:
            kwargs["net_exposures"] = results.loc["Position"]
        if "AbsPosition" in fields:
            kwargs["abs_exposures"] = results.loc["AbsPosition"]
        if "TotalHoldings" in fields:
            kwargs["total_holdings"] = results.loc["TotalHoldings"]
        if "Trade" in fields:
            kwargs["trades"] = results.loc["Trade"]
        if "Commission" in fields:
            kwargs["commissions_pct"] = results.loc["Commission"]
        if "Slippage" in fields:
            kwargs["slippages"] = results.loc["Slippage"]
        if "Benchmark" in fields:
            kwargs["benchmark"] = results.loc["Benchmark"]

        return cls(**kwargs)

    @classmethod
    def from_moonshot_csv(cls, filepath_or_buffer,
                          trim_outliers=None,
                          riskfree=0,
                          compound=True,
                          rolling_sharpe_window=200):
        """
        Creates a DailyPerformance instance from a moonshot backtest results
        CSV.

        Parameters
        ----------
        filepath_or_buffer : str or file-like object
            filepath or file-like object of the CSV

        trim_outliers: int or float, optional
            discard returns that are more than this many standard deviations from the mean

        riskfree : float, optional
            the riskfree rate (default 0)

        compound : bool
             True for compound/geometric returns, False for arithmetic returns (default True)

        rolling_sharpe_window : int, optional
            compute rolling Sharpe over this many periods (default 200)

        Returns
        -------
        DailyPerformance
        """
        results = read_moonshot_csv(filepath_or_buffer)

        return cls._from_moonshot(
            results, trim_outliers=trim_outliers,
            riskfree=riskfree,
            compound=compound,
            rolling_sharpe_window=rolling_sharpe_window)

    @classmethod
    def _from_pnl(cls, results):
        """
        Creates a DailyPerformance instance from a PNL results DataFrame.
        """
        fields = results.index.get_level_values("Field").unique()
        kwargs = {}
        kwargs["returns"] = results.loc["Return"].astype(np.float64)
        kwargs["pnl"] = results.loc["Pnl"].astype(np.float64)
        if "NetExposure" in fields:
            kwargs["net_exposures"] = results.loc["NetExposure"].astype(np.float64)
        if "AbsExposure" in fields:
            kwargs["abs_exposures"] = results.loc["AbsExposure"].astype(np.float64)
        if "TotalHoldings" in fields:
            kwargs["total_holdings"] = results.loc["TotalHoldings"].astype(np.float64)
        if "Commission" in fields:
            kwargs["commissions_pct"] = results.loc["Commission"].astype(np.float64)
        if "CommissionAmount" in fields:
            kwargs["commissions"] = results.loc["CommissionAmount"].astype(np.float64)
        if "Benchmark" in fields:
            kwargs["benchmark"] = results.loc["Benchmark"].astype(np.float64)

        return cls(**kwargs)

    @property
    def cum_returns(self):

        if self._cum_returns is None:
            self._cum_returns = get_cum_returns(self.returns, compound=self.compound)

        return self._cum_returns

    @property
    def cagr(self):
        if self._cagr is None:
            self._cagr = get_cagr(self.cum_returns, compound=self.compound)

        return self._cagr

    @property
    def sharpe(self):
        if self._sharpe is None:
            self._sharpe = get_sharpe(self.returns, riskfree=self.riskfree)

        return self._sharpe

    @property
    def rolling_sharpe(self):
        if self._rolling_sharpe is None:
            self._rolling_sharpe = get_rolling_sharpe(
                self.returns,
                window=self.rolling_sharpe_window,
                riskfree=self.riskfree)

        return self._rolling_sharpe

    @property
    def drawdowns(self):
        if self._drawdowns is None:
            self._drawdowns = get_drawdowns(self.cum_returns)

        return self._drawdowns

    @property
    def max_drawdown(self):
        return self.drawdowns.min()

    @property
    def cum_pnl(self):
        if self._cum_pnl is None and self.pnl is not None:
            self._cum_pnl = self.pnl.cumsum()

        return self._cum_pnl

    @property
    def benchmark_returns(self):
        """
        Returns a Series of benchmark returns from the DataFrame of benchmark
        prices, if any. If more than one strategy/column has benchmark
        prices, uses the first to compute returns.
        """
        if self._benchmark_returns is not None:
            return self._benchmark_returns

        if self._benchmark_prices is None:
            return None

        have_benchmarks = self._benchmark_prices.notnull().any(axis=0)
        have_benchmarks = have_benchmarks[have_benchmarks]
        if have_benchmarks.empty:
            return None

        col = have_benchmarks.index[0]
        if len(have_benchmarks.index) > 1:
            import warnings
            warnings.warn("Multiple benchmarks found, only using first ({0})".format(col))

        benchmark_prices = benchmark[col]

        self._benchmark_returns = benchmark_prices.pct_change()
        self._benchmark_returns.name = "benchmark"

        return self._benchmark_returns

class AggregateDailyPerformance(DailyPerformance):

    def __init__(self, performance):

        super(AggregateDailyPerformance, self).__init__(
            performance.returns.sum(axis=1),
            riskfree=performance.riskfree,
            compound=performance.compound,
            rolling_sharpe_window=performance.rolling_sharpe_window,
            benchmark=performance._benchmark_prices,
            trim_outliers=performance._trim_outliers
        )
        if performance.pnl is not None:
            self.pnl = performance.pnl.sum(axis=1)

        if performance.commissions is not None:
            self.commissions = performance.commissions.sum(axis=1)

        if performance.commissions_pct is not None:
            self.commissions_pct = performance.commissions_pct.sum(axis=1)

        if performance.slippages is not None:
            self.slippages = performance.slippages.sum(axis=1)

        if performance.net_exposures is not None:
            self.net_exposures = performance.net_exposures.sum(axis=1)

        if performance.abs_exposures is not None:
            self.abs_exposures = performance.abs_exposures.sum(axis=1)

        if performance.total_holdings is not None:
            self.total_holdings = performance.total_holdings.sum(axis=1)

        if performance.trades is not None:
            self.trades = performance.trades.sum(axis=1)
