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
import math
import pandas as pd
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib import cycler
from matplotlib.backends.backend_pdf import PdfPages

class Performance(object):
    """
    Class for storing performance attributes and calculating derived statistics.

    Parameters
    ----------
    returns
        a Dataframe of pct returns

    pnl
        a DataFrame of pnl

    order_exposures
        a Dataframe of absolute order exposures (i.e. including non-fills)

    net_exposures
        a Dataframe of net (hedged) exposure

    abs_exposures
        a Dataframe of absolute exposure (ignoring hedging)

    commissions
        a DataFrame of commissions, in the base currency

    commissions_pct
        a DataFrame of commissions, in percentages

    benchmark
        a Series of prices for a benchmark

    riskfree
        the riskfree rate, as a scalar value.

    compound_returns
         True for compound/geometric returns, False for arithmetic returns (default True)

    rolling_sharpe_window
        default 200
    """

    def __init__(
        self,
        returns,
        pnl=None,
        order_exposures=None,
        net_exposures=None,
        abs_exposures=None,
        commissions=None,
        commissions_pct=None,
        benchmark=None,
        riskfree=0,
        compound_returns=True,
        rolling_sharpe_window=200
        ):

        self.returns = returns
        self.pnl = pnl
        self.order_exposures = order_exposures
        self.net_exposures = net_exposures
        self.abs_exposures = abs_exposures
        self.commissions = commissions
        self.commissions_pct = commissions_pct
        self.benchmark = benchmark
        self.riskfree = riskfree
        self.rolling_sharpe_window = rolling_sharpe_window
        self.compound_returns = compound_returns
        self.returns_with_baseline = None
        self.cum_returns = None
        self.cum_returns_with_baseline = None
        self.sharpe = None
        self.rolling_sharpe = None
        self.cagr = None
        self.drawdowns = None
        self.max_drawdown = None
        self.cum_pnl = None
        self._performance_cache_filled = False

    @classmethod
    def from_moonshot_backtest(cls, filepath_or_buffer):
        """
        Creates a Performance instance from a moonshot backtest results CSV.
        """
        results = pd.read_csv(filepath_or_buffer,
                              parse_dates=["Date"],
                              index_col=["Field","Date"])
        fields = results.index.get_level_values("Field").unique()
        kwargs = {}
        kwargs["returns"] = results.loc["Return"]
        if "OrderExposure" in fields:
            kwargs["order_exposures"] = results.loc["OrderExposure"]
        if "NetExposure" in fields:
            kwargs["net_exposures"] = results.loc["NetExposure"]
        if "AbsExposure" in fields:
            kwargs["abs_exposures"] = results.loc["AbsExposure"]
        if "Commission" in fields:
            kwargs["commissions_pct"] = results.loc["Commission"]
        if "Benchmark" in fields:
            kwargs["benchmark"] = results.loc["Benchmark"]

        return cls(**kwargs)

    def fill_performance_cache(self):
        if self._performance_cache_filled:
            return

        self.cum_returns = self.get_cum_returns(self.returns)
        self.cum_returns_with_baseline = self.get_cum_returns(
            self.with_baseline(self.returns))
        self.cagr = self.get_cagr(self.cum_returns)
        self.sharpe = self.get_sharpe(self.returns)
        self.rolling_sharpe = self.get_rolling_sharpe(self.returns)
        self.drawdowns = self.get_drawdowns(self.cum_returns)
        self.max_drawdown = self.get_max_drawdown(self.drawdowns)
        if self.pnl is not None:
            self.cum_pnl = self.pnl.cumsum()

        self._performance_cache_filled = True

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
        data_with_baseline = pd.concat((baseline_row, data))
        return data_with_baseline

    def get_sharpe(self, returns):
        """
        Returns the Sharpe ratio of the provided returns (which should be a
        DataFrame or Series).
        """
        mean = (returns - self.riskfree).mean()
        std = (returns - self.riskfree).std()
        # Returns are assumed to represent daily returns, so annualize the Sharpe ratio
        return mean/std * np.sqrt(252)

    def get_rolling_sharpe(self, returns):
        """
        Computes rolling Sharpe ratios for the returns. Returns should be a
        DataFrame.
        """
        return returns.fillna(0).rolling(
            self.rolling_sharpe_window, min_periods=self.rolling_sharpe_window).apply(
                self.get_sharpe)

    def get_cum_returns(self, returns, compound=None):
        """
        Computes the cumulative returns of the provided Series or DataFrame.
        """
        if compound is None:
            compound = self.compound_returns
        if compound:
            return (1 + returns).cumprod()
        else:
            return returns.cumsum() + 1

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

    def get_efficiency(self, cagr, exposure):
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

    def get_top_movers(self, returns, top_n=10):
        """
        Returns the biggest gainers and losers in the returns.
        """

        if isinstance(returns, pd.DataFrame):
            returns = returns.stack()

        returns = returns.sort_values()

        return pd.concat((returns.head(top_n), returns.tail(top_n)))

class AggregatePerformance(Performance):

    def __init__(self, performance):

        super(AggregatePerformance, self).__init__(
            performance.returns.sum(axis=1),
            riskfree=performance.riskfree,
            compound_returns=performance.compound_returns,
            rolling_sharpe_window=performance.rolling_sharpe_window
        )
        if performance.pnl is not None:
            self.pnl = performance.pnl.sum(axis=1)

        if performance.commissions is not None:
            self.commissions = performance.commissions.sum(axis=1)

        if performance.commissions_pct is not None:
            self.commissions_pct = performance.commissions_pct.sum(axis=1)

        if performance.order_exposures is not None:
            self.order_exposures = performance.order_exposures.sum(axis=1)

        if performance.net_exposures is not None:
            self.net_exposures = performance.net_exposures.sum(axis=1)

        if performance.abs_exposures is not None:
            self.abs_exposures = performance.abs_exposures.sum(axis=1)

class Tearsheet(object):
    """
    Generates a tear sheet of performance stats and graphs.
    """

    def __init__(self, pdf_filename=None, window_size=None):
        self.window_size = window_size or (12.0, 7.5) # width, height in inches
        plt.rc("legend", fontsize="xx-small")
        plt.rc("axes",
               prop_cycle=cycler("color", [
                   "b", "g", "r", "c", "m", "y", "k",
                   "sienna", "chartreuse", "darkorange", "springgreen", "gray",
                   "powderblue", "cornflowerblue", "maroon", "indigo", "deeppink",
                   "salmon", "darkseagreen", "rosybrown", "slateblue", "darkgoldenrod",
                   "deepskyblue",
               ]),
               facecolor="#e1e1e6",
               edgecolor="#aaaaaa",
               grid=True,
               axisbelow=True)
        plt.rc("grid", linestyle="-", color="#ffffff")
        plt.rc("figure", autolayout=True)
        if pdf_filename:
            self.pdf = PdfPages(pdf_filename, keep_empty=True)
        else:
            self.pdf = None

        self.title = "Performance tear sheet"

    def _save_or_show(self):
        """
        Saves the fig to the multi-page PDF, or shows it.
        """
        if self.pdf:
            for fignum in plt.get_fignums():
                self.pdf.savefig(fignum)
            plt.close("all")
            self.pdf.close()
        else:
            plt.show()

    def set_title_from_performance(self, performance):
        """
        Sets a title like "<start date> - <end date>: <securities/strategies/columns>
        """
        min_date = performance.returns.index.min().date().isoformat()
        max_date = performance.returns.index.max().date().isoformat()
        cols = list(performance.returns.columns)
        self.title = "{0} - {1}: {2}".format(
            min_date, max_date, ", ".join(cols))

    def from_moonshot_backtest(self, filepath_or_buffer, *args, **kwargs):
        performance = Performance.from_moonshot_backtest(filepath_or_buffer)
        return self.create_full_tearsheet(performance, *args, **kwargs)

    def create_full_tearsheet(
        self,
        performance,
        montecarlo_n=None,
        montecarlo_preaggregate=True,
        title=None,
        include_exposures_tearsheet=True,
        include_annual_breakdown_tearsheet=True):
        """
        Parameters
        ----------
        performance : instance
            Performance instance

        montecarlo_n : int
            how many Montecarlo simulations to run on the returns, if any

        montecarlo_preaggregate : bool
            whether Montecarlo simulator should preaggregate returns;
            ignored unless montecarlo_n is nonzero

        include_exposures : bool
            whether to include a tear sheet of market exposure

        include_annual_breakdown_tearsheet : bool
            whether to include an annual breakdown of Sharpe and CAGR
        """
        if title:
            self.title = title
        else:
            self.set_title_from_performance(performance)

        agg_performance = AggregatePerformance(performance)

        self.create_performance_tearsheet(performance, agg_performance)

        if include_annual_breakdown_tearsheet:
            self.create_annual_breakdown_tearsheet(performance, agg_performance)

        if include_exposures_tearsheet and any([exposures is not None for exposures in (
            performance.order_exposures, performance.net_exposures, performance.abs_exposures)]):
            self.create_exposures_tearsheet(performance, agg_performance)

        if montecarlo_n:
            self.montecarlo_simulate(
                performance, n=montecarlo_n, preaggregate=montecarlo_preaggregate)

        self._save_or_show()

    def create_performance_tearsheet(self, performance, agg_performance):
        """
        Creates a performance tearsheet.
        """
        agg_performance.fill_performance_cache()

        show_details = len(performance.returns.columns) > 1
        if show_details:
            performance.fill_performance_cache()

        self._create_agg_performance_textbox(agg_performance)

        self._create_performance_plots(
            agg_performance,
            subplot=211 if show_details else 111,
            extra_label="(Aggregate)" if show_details else "")

        if agg_performance.pnl is not None and agg_performance.commissions is not None:
            self._create_gross_and_net_pnl_plot(
                agg_performance,
                extra_label="(Aggregate)" if show_details else "")

        if show_details:
            self._create_performance_plots(performance, subplot=212, extra_label="(Details)")
            self._create_detailed_performance_bar_charts(performance, extra_label="(Details)")

    def _create_performance_plots(self, performance, subplot, extra_label):
        """
        Creates agg/details plots for cumulative returns, drawdowns, rolling
        Sharpe, and possibly pnl.
        """
        if subplot == 111:
            tight_layout = self._tight_layout_clear_suptitle
        else:
            tight_layout = None

        fig = plt.figure("Cumulative Returns", figsize=self.window_size, tight_layout=tight_layout)
        fig.suptitle(self.title)
        axis = fig.add_subplot(subplot)
        max_return = performance.cum_returns.max(axis=0)
        if isinstance(max_return, pd.Series):
            max_return = max_return.max(axis=0)
        # If the price more than doubled, use a log scale
        if max_return >= 2:
            axis.set_yscale("log", basey=2)

        # a 212 subplot means a detailed plot, which isn't compatible with
        # showing commissions
        if performance.commissions_pct is not None and subplot != 212:
            commissions_pct = performance.with_baseline(performance.commissions_pct)
            cum_commissions_pct = performance.get_cum_returns(commissions_pct)
            cum_commissions_pct.name = "commissions"
            performance.cum_returns_with_baseline.name = "returns"
            cum_gross_returns = performance.cum_returns_with_baseline.div(cum_commissions_pct.abs())
            cum_gross_returns.name = "gross returns"
            returns_breakdown = pd.concat((performance.cum_returns_with_baseline, cum_gross_returns, cum_commissions_pct), axis=1)
            plot = returns_breakdown.plot(ax=axis, title="Cumulative Returns {0}".format(extra_label))
            if isinstance(returns_breakdown, pd.DataFrame):
                self._clear_legend(plot)
        else:
            plot = performance.cum_returns_with_baseline.plot(ax=axis, title="Cumulative Returns {0}".format(extra_label))
            if isinstance(performance.cum_returns_with_baseline, pd.DataFrame):
                self._clear_legend(plot)

        fig = plt.figure("Drawdowns", figsize=self.window_size, tight_layout=tight_layout)
        fig.suptitle(self.title)
        axis = fig.add_subplot(subplot)
        plot = performance.drawdowns.plot(ax=axis, title="Drawdowns {0}".format(extra_label))
        if isinstance(performance.drawdowns, pd.DataFrame):
            self._clear_legend(plot)

        if performance.cum_pnl is not None:
            # Cumulative pnl will generally be very similar to cumulative
            # return, but they might be different if cumulative return is
            # calcuated based on an account balance that changes due to
            # causes other than the strategies being charted (e.g. other
            # strategies, contributions/withdrawals, etc.)
            fig = plt.figure("Cumulative PNL", figsize=self.window_size, tight_layout=tight_layout)
            fig.suptitle(self.title)
            axis = fig.add_subplot(subplot)
            plot = performance.cum_pnl.plot(ax=axis, title="Cumulative PNL{0}".format(extra_label))
            if isinstance(performance.cum_pnl, pd.DataFrame):
                self._clear_legend(plot)

        if len(performance.rolling_sharpe.index) > performance.rolling_sharpe_window:
            fig = plt.figure("Rolling Sharpe", figsize=self.window_size, tight_layout=tight_layout)
            fig.suptitle(self.title)
            axis = fig.add_subplot(subplot)
            plot = performance.rolling_sharpe.plot(ax=axis, title="Rolling Sharpe ({0}-day) {1}".format(
                performance.rolling_sharpe_window, extra_label))
            if isinstance(performance.rolling_sharpe, pd.DataFrame):
                self._clear_legend(plot)

    def _create_detailed_performance_bar_charts(self, performance, extra_label):
        if performance.pnl is not None:
            fig = plt.figure("PNL {0}".format(extra_label), figsize=self.window_size,
                             tight_layout=self._tight_layout_clear_suptitle)
            fig.suptitle(self.title)
            axis = fig.add_subplot(111)
            pnl = performance.pnl.sum().sort_values(inplace=False)
            if performance.commissions is not None:
                pnl.name = "pnl"
                commissions = performance.commissions.sum()
                commissions.name = "commissions"
                gross_pnl = pnl + commissions.abs()
                gross_pnl.name = "gross pnl"
                pnl = pd.concat((pnl, gross_pnl, commissions), axis=1)
            pnl.plot(
                ax=axis, kind="bar", title="PNL {0}".format(extra_label))

        fig = plt.figure("CAGR {0}".format(extra_label), figsize=self.window_size,
                         tight_layout=self._tight_layout_clear_suptitle)
        fig.suptitle(self.title)
        axis = fig.add_subplot(111)
        performance.cagr.sort_values(inplace=False).plot(
            ax=axis, kind="bar", title="CAGR {0}".format(extra_label))

        fig = plt.figure("Sharpe {0}".format(extra_label), figsize=self.window_size,
                         tight_layout=self._tight_layout_clear_suptitle)
        fig.suptitle(self.title)
        axis = fig.add_subplot(111)
        performance.sharpe.sort_values(inplace=False).plot(
            ax=axis, kind="bar", title="Sharpe {0}".format(extra_label))

        fig = plt.figure("Max Drawdown {0}".format(extra_label), figsize=self.window_size,
                         tight_layout=self._tight_layout_clear_suptitle)
        fig.suptitle(self.title)
        axis = fig.add_subplot(111)
        performance.max_drawdown.sort_values(inplace=False).plot(
            ax=axis, kind="bar", title="Max drawdown {0}".format(extra_label))

    def _create_agg_performance_textbox(self, agg_performance):
        agg_stats = OrderedDict()
        agg_stats_text = ""

        if agg_performance.pnl is not None:
            agg_stats["PNL"] = agg_performance.pnl.sum()
        if agg_performance.commissions is not None:
            agg_stats["Commissions"] = agg_performance.commissions.sum()

        agg_stats["CAGR"] = agg_performance.cagr
        agg_stats["Sharpe"] = agg_performance.sharpe
        agg_stats["Max Drawdown"] = agg_performance.max_drawdown

        agg_stats_text = self._get_agg_stats_text(agg_stats)
        fig = plt.figure("Aggregate Performance", figsize=self.window_size)
        fig.suptitle(self.title)
        fig.text(.3, .4, agg_stats_text,
                 bbox=dict(color="gray",alpha=0.25),
                 family="monospace",
                 fontsize="xx-large"
                 )

    def _create_gross_and_net_pnl_plot(self, agg_performance, extra_label):
        cum_commissions = agg_performance.commissions.cumsum()
        cum_commissions.name = "commissions"
        cum_pnl = agg_performance.pnl.cumsum()
        cum_pnl.name = "pnl"
        cum_gross_pnl = cum_pnl + cum_commissions.abs()
        cum_gross_pnl.name = "gross pnl"
        pnl_breakdown = pd.concat((cum_pnl, cum_gross_pnl, cum_commissions), axis=1)
        fig = plt.figure("Gross and Net PNL", figsize=self.window_size,
                         tight_layout=self._tight_layout_clear_suptitle)
        fig.suptitle(self.title)
        axis = fig.add_subplot(111)
        pnl_breakdown.plot(ax=axis, title="Gross and Net PNL {0}".format(extra_label))

    def create_exposures_tearsheet(self, performance, agg_performance):
        """
        Create a tearsheet of market exposure.
        """
        agg_performance.fill_performance_cache()

        show_details = len(performance.returns.columns) > 1
        if show_details:
            performance.fill_performance_cache()

        self._create_agg_exposures_textbox(agg_performance)

        self._create_exposures_plots(
            agg_performance,
            subplot=211 if show_details else 111,
            extra_label="(Aggregate)" if show_details else "")

        if show_details:
            self._create_exposures_plots(performance, subplot=212, extra_label="(Details)")
            self._create_detailed_exposures_bar_charts(performance, extra_label="(Details)")

    def _create_agg_exposures_textbox(self, agg_performance):

        agg_stats = OrderedDict()
        agg_stats_text = ""

        if agg_performance.order_exposures is not None:
            avg_order_exposures = agg_performance.get_avg_exposure(agg_performance.order_exposures)
            agg_stats["Avg Order Exposure"] = round(avg_order_exposures, 3)

        if agg_performance.net_exposures is not None:
            avg_net_exposures = agg_performance.get_avg_exposure(agg_performance.net_exposures)
            agg_stats["Avg Net Exposure"] = round(avg_net_exposures, 3)

        if agg_performance.abs_exposures is not None:
            avg_abs_exposures = agg_performance.get_avg_exposure(agg_performance.abs_exposures)
            efficiency = agg_performance.get_efficiency(agg_performance.cagr, avg_abs_exposures)
            agg_stats["Avg Absolute Exposure"] = round(avg_abs_exposures, 3)
            agg_stats["Efficiency (CAGR/Exposure)"] = round(efficiency, 3)

        agg_stats_text = self._get_agg_stats_text(agg_stats, title="Aggregate Exposure")
        fig = plt.figure("Aggregate Exposure", figsize=self.window_size)
        fig.suptitle(self.title)
        fig.text(.3, .4, agg_stats_text,
                 bbox=dict(color="gray",alpha=0.25),
                 family="monospace",
                 fontsize="xx-large")

    def _create_exposures_plots(self, performance, subplot, extra_label):
        if subplot == 111:
            tight_layout = self._tight_layout_clear_suptitle
        else:
            tight_layout = None

        if performance.order_exposures is not None:
            fig = plt.figure("Order Exposures", figsize=self.window_size, tight_layout=tight_layout)
            fig.suptitle(self.title)
            axis = fig.add_subplot(subplot)
            plot = performance.order_exposures.plot(ax=axis, title="Order Exposures {0}".format(extra_label))
            if isinstance(performance.order_exposures, pd.DataFrame):
                self._clear_legend(plot)

        if performance.net_exposures is not None:
            fig = plt.figure("Net Exposures", figsize=self.window_size, tight_layout=tight_layout)
            fig.suptitle(self.title)
            axis = fig.add_subplot(subplot)
            plot = performance.net_exposures.plot(ax=axis, title="Net Exposures {0}".format(extra_label))
            if isinstance(performance.net_exposures, pd.DataFrame):
                self._clear_legend(plot)

        if performance.abs_exposures is not None:
            fig = plt.figure("Absolute Exposures", figsize=self.window_size, tight_layout=tight_layout)
            fig.suptitle(self.title)
            axis = fig.add_subplot(subplot)
            plot = performance.abs_exposures.plot(ax=axis, title="Absolute Exposures {0}".format(extra_label))
            if isinstance(performance.abs_exposures, pd.DataFrame):
                self._clear_legend(plot)

    def _create_detailed_exposures_bar_charts(self, performance, extra_label):

        fig = plt.figure("Avg Exposure {0}".format(extra_label), figsize=self.window_size,
                         tight_layout=self._tight_layout_clear_suptitle)
        fig.suptitle(self.title)

        if performance.order_exposures is not None:
            avg_order_exposures = performance.get_avg_exposure(performance.order_exposures)
            axis = fig.add_subplot(2,2,1)
            avg_order_exposures.sort_values(inplace=False).plot(
                ax=axis, kind="bar", title="Avg Order Exposure {0}".format(extra_label))

        if performance.abs_exposures is not None:
            avg_abs_exposures = performance.get_avg_exposure(performance.abs_exposures)
            axis = fig.add_subplot(2,2,2)
            avg_abs_exposures.sort_values(inplace=False).plot(
                ax=axis, kind="bar", title="Avg Absolute Exposure {0}".format(extra_label))

        if performance.net_exposures is not None:
            avg_net_exposures = performance.get_avg_exposure(performance.net_exposures)
            axis = fig.add_subplot(2,2,3)
            avg_net_exposures.sort_values(inplace=False).plot(
            ax=axis, kind="bar", title="Avg Net Exposure {0}".format(extra_label))

        if performance.abs_exposures is not None:
            efficiencies = performance.get_efficiency(performance.cagr, avg_abs_exposures)
            fig = plt.figure("Efficiency (CAGR/Exposure) {0}".format(extra_label), figsize=self.window_size,
                             tight_layout=self._tight_layout_clear_suptitle)
            fig.suptitle(self.title)
            axis = fig.add_subplot(111)
            efficiencies.sort_values(inplace=False).plot(
                ax=axis, kind="bar", title="Efficiency (CAGR/Exposure) {0}".format(extra_label))

    def create_annual_breakdown_tearsheet(self, performance, agg_performance):
        """
        Creates agg/detailed bar charts showing CAGR and Sharpe by year.
        """
        agg_performance.fill_performance_cache()

        show_details = len(performance.returns.columns) > 1
        if show_details:
            performance.fill_performance_cache()

        self._create_annual_breakdown_plots(
            agg_performance,
            subplot=211 if show_details else 111,
            extra_label="(Aggregate)" if show_details else "")

        if show_details:
            self._create_annual_breakdown_plots(performance, subplot=212, extra_label="(Details)")

    def _create_annual_breakdown_plots(self, performance, subplot, extra_label):
        if subplot == 111:
            tight_layout = self._tight_layout_clear_suptitle
        else:
            tight_layout = None

        grouped_returns = performance.returns.groupby(performance.returns.index.year)
        cagrs_by_year = grouped_returns.apply(lambda x: performance.get_cagr(
            performance.get_cum_returns(x)))
        sharpes_by_year = grouped_returns.apply(performance.get_sharpe)

        fig = plt.figure("CAGR by Year", figsize=self.window_size, tight_layout=tight_layout)
        fig.suptitle(self.title)
        axis = fig.add_subplot(subplot)
        plot = cagrs_by_year.plot(ax=axis, kind="bar", title="CAGR by Year {0}".format(extra_label))
        if isinstance(cagrs_by_year, pd.DataFrame):
            self._clear_legend(plot)

        fig = plt.figure("Sharpe by Year", figsize=self.window_size, tight_layout=tight_layout)
        fig.suptitle(self.title)
        axis = fig.add_subplot(subplot)
        plot = sharpes_by_year.plot(ax=axis, kind="bar", title="Sharpe by Year {0}".format(extra_label))
        if isinstance(sharpes_by_year, pd.DataFrame):
            self._clear_legend(plot)

    def _get_agg_stats_text(self, agg_stats, title="Aggregate Performance"):
        """
        From a dict of aggregate stats, formats a text block.
        """
        # Create pd.Series from agg_stats to nice repr
        agg_stats = pd.Series(agg_stats)
        # Split lines
        lines = repr(agg_stats).split("\n")
        width = len(lines[0])
        # Strip last line (dtype)
        agg_stats_text = "\n".join(lines[:-1])
        agg_stats_text = "{0}\n{1}\n{2}".format(title, "="*width, agg_stats_text)
        return agg_stats_text

    def _get_plot_dimensions(self, plot_count):
        """
        Returns a tuple of rows, cols needed to accomodate the plot_count.
        """
        rows = math.ceil(math.sqrt(plot_count))
        cols = math.ceil(plot_count/rows)
        return rows, cols

    def create_shortfall_tearsheet(self, live_returns, simulated_returns):
        """
        Shows drawdown and equity curves comparing the live and simulated
        returns. DataFrames should include a column for each strategy being
        analyzed.
        """
        raise NotImplementedError()

        # TODO: be more specific
        self.title = "Shortfall Analysis"

        strategies = live_returns.columns.union(simulated_returns.columns)
        strategy_count = len(strategies)

        rows, cols = self._get_plot_dimensions(strategy_count)

        for i, strategy in enumerate(strategies):
            if strategy in live_returns.columns:
                strategy_live_returns = live_returns[strategy]
            else:
                strategy_live_returns = pd.Series(0, index=live_returns.index)
            strategy_live_returns.name = "live"
            strategy_simulated_returns = simulated_returns[strategy]
            strategy_simulated_returns.name = "simulated"
            strategy_returns = pd.concat((strategy_live_returns, strategy_simulated_returns), axis=1).fillna(0)
            cum_returns = self.cum_returns(self._with_baseline(strategy_returns))
            drawdowns = self.drawdowns(cum_returns)
            shortfall = cum_returns.live - cum_returns.simulated

            fig = plt.figure("Cumulative Returns", figsize=self.window_size)
            fig.suptitle(self.title)
            axis = fig.add_subplot(rows, cols, i+1)
            cum_returns.plot(ax=axis, title=strategy)

            fig = plt.figure("Drawdowns", figsize=self.window_size)
            fig.suptitle(self.title)
            axis = fig.add_subplot(rows, cols, i+1)
            drawdowns.plot(ax=axis, title=strategy)

            fig = plt.figure("Shortfall", figsize=self.window_size)
            fig.suptitle(self.title)
            axis = fig.add_subplot(rows, cols, i+1)
            shortfall.plot(ax=axis, title=strategy, kind="area", stacked=False)

        self._save_or_show()

    def plot_arbitrary(self, series, kind="line", title=None):
        """
        Plots an arbitrary list of Series or DataFrames. The Series or
        DataFrame names are used as the chart titles.
        """
        series_list = series
        rows, cols = self._get_plot_dimensions(len(series))
        fig = plt.figure(figsize=self.window_size)
        if title:
            self.title = title

        fig.suptitle(self.title)
        for i, series in enumerate(series_list):
            axis = fig.add_subplot(rows, cols, i + 1)
            plot = series.plot(ax=axis, kind=kind, title=series.name, fontsize="small")
            if isinstance(series, pd.DataFrame):
                self._clear_legend(plot)

        self._save_or_show()

    @property
    def _tight_layout_clear_suptitle(self):
        # leave room at top for suptitle
        return dict(rect=[0,0,1,.9])

    def _clear_legend(self, plot):
        """
        Anchors the legend to the outside right of the plot area so you can
        see the plot.
        """
        plot.figure.set_tight_layout({"pad":10, "h_pad":1, "w_pad":1})
        plot.legend(
            loc='center left', bbox_to_anchor=(1, 0.5), fontsize="x-small")

    def montecarlo_simulate(self, performance, n=5, preaggregate=True):
        """
        Runs a Montecarlo simulation by shuffling the dataframe of returns n
        number of times and graphing the cum_returns and drawdowns overlaid
        by the original returns. If preaggregate is True, aggregates the
        returns before the simulation, otherwise after the simulation.
        Preaggregation only randomizes by day (assuming each row is a day),
        while not preaggregating randomizes each value.
        """

        all_simulations = []

        returns = performance.returns

        if preaggregate:
            returns = returns.sum(axis=1)

        for i in range(n):
            if preaggregate:
                sim_returns = pd.Series(np.random.permutation(returns), index=returns.index)
            else:
                sim_returns = returns.apply(np.random.permutation).sum(axis=1)
            all_simulations.append(sim_returns)

        sim_returns = pd.concat(all_simulations, axis=1)

        if not preaggregate:
            returns = returns.sum(axis=1)

        cum_sim_returns = performance.get_cum_returns(performance.with_baseline(sim_returns))
        cum_returns = performance.get_cum_returns(performance.with_baseline(returns))
        fig = plt.figure("Montecarlo Simulation", figsize=self.window_size,
                         tight_layout=self._tight_layout_clear_suptitle)
        fig.suptitle(self.title)
        axis = fig.add_subplot(211)
        cum_sim_returns.plot(ax=axis, title="Montecarlo Cumulative Returns (n={0})".format(n), legend=False)
        cum_returns.plot(ax=axis, linewidth=4, color="black")
        axis = fig.add_subplot(212)
        performance.get_drawdowns(cum_sim_returns).plot(ax=axis, title="Montecarlo Drawdowns (n={0})".format(n), legend=False)
        performance.get_drawdowns(cum_returns).plot(ax=axis, linewidth=4, color="black")
