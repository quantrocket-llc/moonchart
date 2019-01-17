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

    compound_returns : bool
         True for compound/geometric returns, False for arithmetic returns (default True)

    rolling_sharpe_window : int, optional
        compute rolling Sharpe over this many periods (default 200)
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
        compound_returns=True,
        rolling_sharpe_window=200
        ):

        self.returns = returns
        if len(self.returns.index) < 2:
            raise InsufficientData(
                "Moonchart needs at least 2 dates to analyze performance, "
                "but returns DataFrame has length {0}".format(len(self.returns.index)))
        self.pnl = pnl
        self.net_exposures = net_exposures
        self.abs_exposures = abs_exposures
        self.total_holdings = total_holdings
        self.trades = trades
        self.commissions = commissions
        self.commissions_pct = commissions_pct
        self.slippages = slippages
        self.benchmark = benchmark
        self.riskfree = riskfree
        self.rolling_sharpe_window = rolling_sharpe_window
        self.compound_returns = compound_returns
        self._returns_with_baseline = None
        self._cum_returns = None
        self._cum_returns_with_baseline = None
        self._sharpe = None
        self._rolling_sharpe = None
        self._cagr = None
        self._drawdowns = None
        self._max_drawdown = None
        self._cum_pnl = None

    @classmethod
    def from_moonshot(cls, results):
        """
        Creates a DailyPerformance instance from a moonshot backtest results DataFrame.
        """
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
    def from_moonshot_csv(cls, filepath_or_buffer):
        """
        Creates a DailyPerformance instance from a moonshot backtest results
        CSV.

        Parameters
        ----------
        filepath_or_buffer : str or file-like object
            filepath or file-like object of the CSV

        Returns
        -------
        DailyPerformance
        """
        results = read_moonshot_csv(filepath_or_buffer)

        if "Time" in results.index.names:
            results = intraday_to_daily(results)

        return cls.from_moonshot(results)

    @classmethod
    def from_pnl(cls, results):
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
            self._cum_returns = self.get_cum_returns(self.returns)

        return self._cum_returns

    @property
    def cum_returns_with_baseline(self):
        """
        Cumulative returns, with a leading 0 return.
        """
        if self._cum_returns_with_baseline is None:
            self._cum_returns_with_baseline = self.get_cum_returns(
                self.with_baseline(self.returns))

        return self._cum_returns_with_baseline

    @property
    def cagr(self):
        if self._cagr is None:
            self._cagr = self.get_cagr(self.cum_returns)

        return self._cagr

    @property
    def sharpe(self):
        if self._sharpe is None:
            self._sharpe = self.get_sharpe(self.returns)

        return self._sharpe

    @property
    def rolling_sharpe(self):
        if self._rolling_sharpe is None:
            self._rolling_sharpe = self.get_rolling_sharpe(self.returns)

        return self._rolling_sharpe

    @property
    def drawdowns(self):
        if self._drawdowns is None:
            self._drawdowns = self.get_drawdowns(self.cum_returns)

        return self._drawdowns

    @property
    def max_drawdown(self):
        if self._max_drawdown is None:
            self._max_drawdown = self.get_max_drawdown(self.drawdowns)

        return self._max_drawdown

    @property
    def cum_pnl(self):
        if self._cum_pnl is None and self.pnl is not None:
            self._cum_pnl = self.pnl.cumsum()

        return self._cum_pnl

    def with_baseline(self, data):
        """
        Adds an initial period with a return of 0, as a baseline.
        """
        period_length = data.index[1] - data.index[0]
        prior_period = data.index[0] - period_length
        if isinstance(data, pd.DataFrame):
            baseline_row = pd.DataFrame(0, index=[prior_period], columns=data.columns)
        else:
            baseline_row = pd.Series(0, index=[prior_period], name=data.name)
        try:
            data_with_baseline = pd.concat((baseline_row, data), sort=False)
        except TypeError:
            # sort was introduced in pandas 0.23
            data_with_baseline = pd.concat((baseline_row, data))
        return data_with_baseline

    def get_sharpe(self, returns):
        """
        Returns the Sharpe ratio of the provided returns (which should be a
        DataFrame or Series).
        """
        mean = (returns - self.riskfree).mean()
        if isinstance(mean, float) and mean == 0:
            return 0
        std = (returns - self.riskfree).std()
        # Returns are assumed to represent daily returns, so annualize the Sharpe ratio
        return mean/std * np.sqrt(252)

    def get_rolling_sharpe(self, returns):
        """
        Computes rolling Sharpe ratios for the returns. Returns should be a
        DataFrame.
        """
        rolling_returns = returns.fillna(0).rolling(
            self.rolling_sharpe_window, min_periods=self.rolling_sharpe_window)
        try:
            return rolling_returns.apply(self.get_sharpe, raw=True)
        except TypeError as e:
            # handle pandas<0.23
            if "apply() got an unexpected keyword argument 'raw'" in repr(e):
                return rolling_returns.apply(self.get_sharpe)
            else:
                raise

    def get_cum_returns(self, returns, compound=None):
        """
        Computes the cumulative returns of the provided Series or DataFrame.
        """
        if compound is None:
            compound = self.compound_returns
        if compound:
            cum_returns = (1 + returns).cumprod()
        else:
            cum_returns = returns.cumsum() + 1

        cum_returns.index.name = "Date"
        return cum_returns

    def get_cagr(self, cum_returns):
        """
        Computes the CAGR of the cum_returns (a DataFrame or Series).
        """
        # For DataFrames, apply this method to each Series.
        if isinstance(cum_returns, pd.DataFrame):
            return cum_returns.apply(self.get_cagr, axis=0)

        # Ignore nulls when compting CAGR
        cum_returns = cum_returns[cum_returns.notnull()]

        if cum_returns.empty:
            return 0

        # Compute the CAGR of the Series
        min_date = cum_returns.index.min()
        max_date = cum_returns.index.max()
        years = ((max_date - min_date).days or 1)/365.0
        ending_value = cum_returns.iloc[-1]
        # Since we are computing CAGR on cumulative returns, the beginning
        # value is always 1.
        beginning_value = 1
        if self.compound_returns:
            cagr = (ending_value/beginning_value)**(1/years) - 1
        else:
            # Compound annual growth rate doesn't apply to arithmetic
            # returns, so just divide the cum_returns by the number of years
            # to get the annual return
            cagr = (ending_value/beginning_value - 1)/years

        return cagr

    def get_avg_exposure(self, exposures):
        """
        Calculates the avg exposure.
        """
        return exposures.mean()

    def get_avg_total_holdings(self, total_holdings):
        """
        Calculates the avg exposure.
        """
        return total_holdings.mean()

    def get_normalized_cagr(self, cagr, exposure):
        """
        Returns the CAGR per 1x exposure, a measure of the strategy's
        efficiency.
        """
        return cagr / exposure

    def get_drawdowns(self, cum_returns):
        """
        Computes the drawdowns of the cum_returns (a Series or DataFrame).
        """
        cum_returns = cum_returns[cum_returns.notnull()]
        highwater_marks = cum_returns.expanding().max()
        drawdowns = cum_returns/highwater_marks - 1
        return drawdowns

    def get_max_drawdown(self, drawdowns):
        """
        Returns the max drawdown.
        """
        return drawdowns.min()

    def get_benchmark_returns(self):
        """
        Returns a Series of benchmark prices, if any. If more than one strategy/column has
        benchmark prices, returns the first.
        """
        if self.benchmark is None:
            return None
        have_benchmarks = self.benchmark.notnull().any(axis=0)
        have_benchmarks = have_benchmarks[have_benchmarks]
        if have_benchmarks.empty:
            return None

        col = have_benchmarks.index[0]
        if len(have_benchmarks.index) > 1:
            import warnings
            warnings.warn("Multiple benchmarks found, only using first ({0})".format(col))

        benchmark_prices = self.benchmark[col]
        benchmark_prices.name = "benchmark"
        return benchmark_prices.pct_change()

    def get_top_movers(self, returns, top_n=10):
        """
        Returns the biggest gainers and losers in the returns.
        """

        if isinstance(returns, pd.DataFrame):
            returns = returns.stack()

        returns = returns.sort_values()

        try:
            top_movers = pd.concat((returns.head(top_n), returns.tail(top_n)), sort=True)
        except TypeError:
            # sort was introduced in pandas 0.23
            top_movers = pd.concat((returns.head(top_n), returns.tail(top_n)))

        return top_movers

class AggregateDailyPerformance(DailyPerformance):

    def __init__(self, performance):

        super(AggregateDailyPerformance, self).__init__(
            performance.returns.sum(axis=1),
            riskfree=performance.riskfree,
            compound_returns=performance.compound_returns,
            rolling_sharpe_window=performance.rolling_sharpe_window,
            benchmark=performance.benchmark
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
