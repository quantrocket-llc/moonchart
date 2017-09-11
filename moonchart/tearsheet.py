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

import datetime
import numpy as np
import math
import pandas as pd
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib import cycler
from matplotlib.backends.backend_pdf import PdfPages

class Tearsheet(object):
    """
    Generates a tear sheet of performance stats and graphs.
    """

    WINDOW_SIZE = (16,10) # width, height in inches

    def __init__(self, compound_returns=True, pdf_filename=None):
        # cum_returns can be calculated geometrically (compounding) or
        # arithmetically (no compounding)
        self.compound_returns = compound_returns
        plt.rc("legend", fontsize="xx-small")
        plt.rc("axes",
               prop_cycle=cycler("color", [
                   "b", "g", "r", "c", "m", "y", "k",
                   "sienna", "chartreuse", "darkorange", "springgreen", "gray",
                   "powderblue", "cornflowerblue", "maroon", "indigo", "deeppink",
                   "salmon", "darkseagreen", "rosybrown", "slateblue", "darkgoldenrod",
                   "deepskyblue",
               ]))
        plt.rc("figure", autolayout=True)
        if pdf_filename:
            self.pdf = PdfPages(pdf_filename, keep_empty=True)
        else:
            self.pdf = None

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

    def _with_baseline(self, data):
        """
        Adds an initial period with a return of 0, as a baseline.
        """
        min_date = data.index.min()
        prior_date = min_date - datetime.timedelta(days=1)
        if isinstance(data, pd.DataFrame):
            baseline_row = pd.DataFrame(0, index=[prior_date], columns=data.columns)
        else:
            baseline_row = pd.Series(0, index=[prior_date], name=data.name)
        data_with_baseline = pd.concat((baseline_row, data))
        return data_with_baseline

    def sharpe(self, returns, riskfree=0):
        """
        Returns the Sharpe ratio of the provided returns (which should be a
        DataFrame or Series).
        """
        mean = (returns - riskfree).mean()
        std = (returns - riskfree).std()
        # Returns are assumed to represent daily returns, so annualize the Sharpe ratio
        return mean/std * np.sqrt(252)

    def rolling_sharpe(self, returns, window=200, min_periods=None, riskfree=0):
        """
        Computes rolling Sharpe ratios for the returns. Returns should be a
        DataFrame.
        """
        return returns.fillna(0).rolling(window, min_periods=min_periods).apply(
            self.sharpe, kwargs=dict(riskfree=riskfree))

    def top_movers(self, returns, top_n=10):
        """
        Returns the biggest gainers and losers in the returns.
        """

        if isinstance(returns, pd.DataFrame):
            returns = returns.stack()

        returns = returns.sort_values()

        return pd.concat((returns.head(top_n), returns.tail(top_n)))

    def cum_returns(self, returns, compound=None):
        """
        Computes the cumulative returns of the provided Series or DataFrame.
        """
        if compound is None:
            compound = self.compound_returns
        if compound:
            return (1 + returns).cumprod()
        else:
            return returns.cumsum() + 1

    def mean_return(self, returns, exposures=None):
        """
        Computes the mean return of the provided Series of DataFrame. If
        exposures is provided, mean returns will be normalized by exposures.
        """
        returns = returns.where(returns.abs() > 0)

        if exposures is not None:
            exposures = exposures.where(returns.notnull())
            exposures = exposures.where(exposures != 0)
            returns = returns / exposures.abs()

        return returns.mean()

    def cagr(self, cum_returns):
        """
        Computes the CAGR of the cum_returns (a DataFrame or Series).
        """
        # For DataFrames, apply this method to each Series.
        if isinstance(cum_returns, pd.DataFrame):
            return cum_returns.apply(self.cagr, axis=0)

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

    def avg_exposure(self, exposures):
        """
        Calculates the avg exposure.
        """
        return exposures.mean()

    def efficiency(self, cagr, exposure):
        """
        Returns the CAGR per 1x exposure, a measure of the strategy's
        efficiency.
        """
        return cagr / exposure

    def drawdowns(self, cum_returns):
        """
        Computes the drawdowns of the cum_returns (a Series or DataFrame).
        """
        cum_returns = cum_returns[cum_returns.notnull()]
        highwater_marks = cum_returns.expanding().max()
        drawdowns = cum_returns/highwater_marks - 1
        return drawdowns

    def max_drawdown(self, drawdowns):
        """
        Returns the max drawdown.
        """
        return drawdowns.min()

    def create_tearsheet_from_moonshot_backtest_csv(self, filepath_or_buffer, **kwargs):
        """
        Creates a tear sheet from moonshot backtest results.
        """
        results = pd.read_csv(filepath_or_buffer,
                              parse_dates=["Date"],
                              index_col=["Field","Date"])
        fields = results.index.get_level_values("Field").unique()
        _kwargs = {}
        _kwargs["returns"] = results.loc["Return"]
        if "OrderExposure" in fields:
            _kwargs["order_exposures"] = results.loc["OrderExposure"]
        if "NetExposure" in fields:
            _kwargs["net_exposures"] = results.loc["NetExposure"]
        if "AbsExposure" in fields:
            _kwargs["abs_exposures"] = results.loc["AbsExposure"]
        if "Commission" in fields:
            _kwargs["commissions_pct"] = results.loc["Commission"]
        if "Benchmark" in fields:
            _kwargs["benchmark"] = results.loc["Benchmark"]

        _kwargs.update(kwargs)

        return self.create_full_tearsheet(**_kwargs)

    def create_full_tearsheet(
        self,
        returns,
        pnl=None,
        order_exposures=None,
        net_exposures=None,
        abs_exposures=None,
        benchmark=None,
        commissions=None,
        commissions_pct=None,
        montecarlo_n=None,
        montecarlo_preaggregate=True,
        riskfree=0,
        aggregate=True,
        details=True,
        title=None,
        show_exposures=False,
        show_mean=False,
        show_annual=False,
        show_monthly=False,
        show_summaries=True):
        """

        Parameters
        ----------
        returns
            a Dataframe of pct returns
        pnl
            a DataFrame of pnl
        net_exposures
            a Dataframe of net (hedged) exposure
        abs_exposures
            a Dataframe of absolute exposure (ignoring hedging)
        order_exposures
            a Dataframe of absolute order exposures (i.e. including non-fills)
        benchmark
            a Series of prices for a benchmark
        commissions
            a DataFrame of commissions, in dollars
        commissions_pct
            a DataFrame of commissions, in pcts
        montecarlo_n
            how many Montecarlo simulations to run on the returns, if any
        montecarlo_preaggregate
            whether Montecarlo simulator should preaggregate returns;
            ignored unless montecarlo_n is nonzero
        riskfree
            the riskfree rate, as a scalar value.
        aggregate
            whether to compute and plot aggregate statistics
        details
            whether to compute and plot detailed statistics
        show_annual
            whether to show annual breakdown of Sharpe and CAGR
        show_monthly
            whether to show monthly breakdown of Sharpe and CAGR
        """

        if (
            any((net_exposures is not None, abs_exposures is not None, order_exposures is not None))
            and not all((net_exposures is not None, abs_exposures is not None, order_exposures is not None))):
            raise ValueError(
                "Must provide all of order_exposures, net_exposures, and abs_exposures, or none")

        if not title:
            min_date = returns.index.min().date().isoformat()
            max_date = returns.index.max().date().isoformat()
            cols = list(returns.columns)
            title = "{0} - {1}: {2}".format(
                min_date, max_date, ", ".join(cols))

        if aggregate:
            aggregate_returns = returns.sum(axis=1)
            if pnl is not None:
                aggregate_pnl = pnl.sum(axis=1)
            else:
                aggregate_pnl = None
            if commissions is not None:
                aggregate_commissions = commissions.sum(axis=1)
            else:
                aggregate_commissions = None
            if commissions_pct is not None:
                aggregate_commissions_pct = commissions_pct.sum(axis=1)
            else:
                aggregate_commissions_pct = None
            if net_exposures is not None:
                aggregate_net_exposures = net_exposures.sum(axis=1)
                aggregate_abs_exposures = abs_exposures.sum(axis=1)
                aggregate_order_exposures = order_exposures.sum(axis=1)
            else:
                aggregate_net_exposures = aggregate_abs_exposures = aggregate_order_exposures = None
            self._show(aggregate_returns, aggregate_pnl, aggregate_order_exposures, aggregate_net_exposures,
                       aggregate_abs_exposures, aggregate_commissions, aggregate_commissions_pct,
                       riskfree, extra_label="(Aggregate)", title=title, subplot=211 if details else 111,
                       show_exposures=show_exposures, show_mean=show_mean, show_annual=show_annual,
                       show_monthly=show_monthly, show_summaries=show_summaries, benchmark=benchmark)

        if details:
            self._show(returns, pnl, order_exposures, net_exposures, abs_exposures, commissions,
                       riskfree=riskfree, extra_label="(Details)", title=title,
                       subplot=212 if aggregate else 111,
                       show_exposures=show_exposures,
                       show_mean=show_mean, show_annual=show_annual,
                       show_summaries=show_summaries,
                       benchmark=benchmark)

        if montecarlo_n:
            self.montecarlo_simulate(
                returns, n=montecarlo_n, preaggregate=montecarlo_preaggregate, title=title)

        self._save_or_show()

    def _show(self, returns, pnl=None, order_exposures=None, net_exposures=None, abs_exposures=None,
              commissions=None, commissions_pct=None, riskfree=0, extra_label="", title=None,
              subplot=111, show_exposures=False, show_mean=False, show_annual=False,
              show_monthly=False, show_summaries=True, benchmark=None):
        """
        Computes and plots performance statistics.
        """
        title = title or "Performance Results"

        rolling_sharpe_window = 200

        agg_stats = OrderedDict()
        agg_stats_text = ""

        # We will use _with_baseline for graphing cum_returns and
        # drawdowns, so that the first day's return/drawdown appears on the
        # chart. But we'll use returns without baseline for calculating
        # Sharpe and CAGR since the baseline row doesn't represent an actual
        # trading day
        returns_with_baseline = self._with_baseline(returns)
        cum_returns = self.cum_returns(returns)
        cum_returns_with_baseline = self.cum_returns(returns_with_baseline)
        sharpe = self.sharpe(returns, riskfree)
        rolling_sharpe = self.rolling_sharpe(returns, window=rolling_sharpe_window,
                                             riskfree=riskfree)
        cagr = self.cagr(cum_returns)
        drawdowns = self.drawdowns(cum_returns_with_baseline)
        max_drawdown = self.max_drawdown(drawdowns)
        if show_mean:
            mean_returns = self.mean_return(returns, abs_exposures)
        if pnl is not None:
            cum_pnl = pnl.cumsum()

        if show_summaries and isinstance(cagr, pd.Series):
            if pnl is not None:
                fig = plt.figure("PNL {0}".format(extra_label), figsize=self.WINDOW_SIZE,
                                 tight_layout=self._tight_layout_clear_suptitle)
                fig.suptitle(title)
                axis = fig.add_subplot(111)
                pnl = pnl.sum().sort_values(inplace=False)
                if commissions is not None:
                    pnl.name = "pnl"
                    commissions = commissions.sum()
                    commissions.name = "commissions"
                    gross_pnl = pnl + commissions.abs()
                    gross_pnl.name = "gross pnl"
                    pnl = pd.concat((pnl, gross_pnl, commissions), axis=1)
                pnl.plot(
                    ax=axis, kind="bar", title="PNL {0}".format(extra_label))
            fig = plt.figure("CAGR {0}".format(extra_label), figsize=self.WINDOW_SIZE,
                             tight_layout=self._tight_layout_clear_suptitle)
            fig.suptitle(title)
            axis = fig.add_subplot(111)
            cagr.sort_values(inplace=False).plot(
                ax=axis, kind="bar", title="CAGR {0}".format(extra_label))
            fig = plt.figure("Sharpe {0}".format(extra_label), figsize=self.WINDOW_SIZE,
                             tight_layout=self._tight_layout_clear_suptitle)
            fig.suptitle(title)
            axis = fig.add_subplot(111)
            sharpe.sort_values(inplace=False).plot(
                ax=axis, kind="bar", title="Sharpe {0}".format(extra_label))
            fig = plt.figure("Max Drawdown {0}".format(extra_label), figsize=self.WINDOW_SIZE,
                             tight_layout=self._tight_layout_clear_suptitle)
            fig.suptitle(title)
            axis = fig.add_subplot(111)
            max_drawdown.sort_values(inplace=False).plot(
                ax=axis, kind="bar", title="Max drawdown {0}".format(extra_label))
            if show_exposures and order_exposures is not None and net_exposures is not None and abs_exposures is not None:
                avg_order_exposures = self.avg_exposure(order_exposures)
                avg_net_exposures = self.avg_exposure(net_exposures)
                avg_abs_exposures = self.avg_exposure(abs_exposures)
                fig = plt.figure("Avg Exposure {0}".format(extra_label), figsize=self.WINDOW_SIZE,
                                 tight_layout=self._tight_layout_clear_suptitle)
                fig.suptitle(title)
                axis = fig.add_subplot(2,2,1)
                avg_order_exposures.sort_values(inplace=False).plot(
                    ax=axis, kind="bar", title="Avg Order Exposure {0}".format(extra_label))
                axis = fig.add_subplot(2,2,2)
                avg_abs_exposures.sort_values(inplace=False).plot(
                    ax=axis, kind="bar", title="Avg Absolute Exposure {0}".format(extra_label))
                axis = fig.add_subplot(2,2,3)
                avg_net_exposures.sort_values(inplace=False).plot(
                    ax=axis, kind="bar", title="Avg Net Exposure {0}".format(extra_label))
                efficiencies = self.efficiency(cagr, avg_abs_exposures)
                fig = plt.figure("Efficiency (CAGR/Exposure) {0}".format(extra_label), figsize=self.WINDOW_SIZE,
                                 tight_layout=self._tight_layout_clear_suptitle)
                fig.suptitle(title)
                axis = fig.add_subplot(111)
                efficiencies.sort_values(inplace=False).plot(
                    ax=axis, kind="bar", title="Efficiency (CAGR/Exposure) {0}".format(extra_label))
            if show_mean:
                _title = "{0}ean return {1}".format(
                    "Normalized m" if abs_exposures is not None else "M", extra_label)
                fig = plt.figure(_title, figsize=self.WINDOW_SIZE,
                                 tight_layout=self._tight_layout_clear_suptitle)
                fig.suptitle(title)
                axis = fig.add_subplot(111)
                mean_returns.sort_values(inplace=False).plot(ax=axis, kind="bar", title=_title)

        elif show_summaries:
            if pnl is not None:
                agg_stats["PNL"] = pnl.sum()
            if commissions is not None:
                agg_stats["Commissions"] = commissions.sum()
            agg_stats["CAGR"] = cagr
            agg_stats["Sharpe"] = sharpe
            if rolling_sharpe.notnull().any():
                agg_stats["Rolling Sharpe Std Dev"] = rolling_sharpe.std()
            agg_stats["Max Drawdown"] = max_drawdown
            if show_exposures and order_exposures is not None and net_exposures is not None and abs_exposures is not None:
                avg_order_exposures = self.avg_exposure(order_exposures)
                avg_net_exposures = self.avg_exposure(net_exposures)
                avg_abs_exposures = self.avg_exposure(abs_exposures)
                efficiency = self.efficiency(cagr, avg_abs_exposures)
                agg_stats["Avg Order Exposure"] = round(avg_order_exposures, 3)
                agg_stats["Avg Net Exposure"] = round(avg_net_exposures, 3)
                agg_stats["Avg Absolute Exposure"] = round(avg_abs_exposures, 3)
                agg_stats["Efficiency (CAGR/Exposure)"] = round(efficiency, 3)

            if show_mean:
                agg_stats["{0}ean return".format(
                    "Normalized m" if abs_exposures is not None else "M")] = mean_returns

            agg_stats_text = self._get_agg_stats_text(agg_stats)
            fig = plt.figure("Aggregate Performance", figsize=self.WINDOW_SIZE)
            fig.suptitle(title)
            fig.text(.3, .3, agg_stats_text, bbox=dict(color="gray",alpha=0.25), family="monospace")

            if pnl is not None and commissions is not None:
                cum_commissions = commissions.cumsum()
                cum_commissions.name = "commissions"
                cum_pnl = pnl.cumsum()
                cum_pnl.name = "pnl"
                cum_gross_pnl = cum_pnl + cum_commissions.abs()
                cum_gross_pnl.name = "gross pnl"
                pnl_breakdown = pd.concat((cum_pnl, cum_gross_pnl, cum_commissions), axis=1)
                fig = plt.figure("Gross and Net PNL", figsize=self.WINDOW_SIZE,
                                 tight_layout=self._tight_layout_clear_suptitle)
                fig.suptitle(title)
                axis = fig.add_subplot(111)
                pnl_breakdown.plot(ax=axis, title="Gross and Net PNL {0}".format(extra_label))

        if show_annual:
            grouped_returns = returns.groupby(returns.index.year)
            cagrs_by_year = grouped_returns.apply(lambda x: self.cagr(
                self.cum_returns(x)))
            sharpes_by_year = grouped_returns.apply(self.sharpe)
            fig = plt.figure("CAGR by Year", figsize=self.WINDOW_SIZE)
            fig.suptitle(title)
            axis = fig.add_subplot(subplot)
            plot = cagrs_by_year.plot(ax=axis, kind="bar", title="CAGR by Year {0}".format(extra_label))
            if isinstance(cagrs_by_year, pd.DataFrame):
                self._clear_legend(plot)
            fig = plt.figure("Sharpe by Year", figsize=self.WINDOW_SIZE)
            fig.suptitle(title)
            axis = fig.add_subplot(subplot)
            plot = sharpes_by_year.plot(ax=axis, kind="bar", title="Sharpe by Year {0}".format(extra_label))
            if isinstance(sharpes_by_year, pd.DataFrame):
                self._clear_legend(plot)
        if show_monthly:
            grouped_returns = returns.groupby((returns.index.year, returns.index.month))
            cagrs_by_month = grouped_returns.apply(lambda x: self.cagr(
                self.cum_returns(x)))
            sharpes_by_month = grouped_returns.apply(self.sharpe)
            fig = plt.figure("Performance by Month", figsize=self.WINDOW_SIZE)
            fig.suptitle(title)
            axis = fig.add_subplot(211)
            plot = cagrs_by_month.plot(ax=axis, kind="bar", title="CAGR by Month {0}".format(extra_label))
            if isinstance(cagrs_by_month, pd.DataFrame):
                self._clear_legend(plot)
            axis = fig.add_subplot(212)
            plot = sharpes_by_month.plot(ax=axis, kind="bar", title="Sharpe by Month {0}".format(extra_label))
            if isinstance(sharpes_by_month, pd.DataFrame):
                self._clear_legend(plot)

        fig = plt.figure("Cumulative Returns", figsize=self.WINDOW_SIZE)
        fig.suptitle(title)
        axis = fig.add_subplot(subplot)
        max_return = cum_returns.max(axis=0)
        if isinstance(max_return, pd.Series):
            max_return = max_return.max(axis=0)
        # If the price more than doubled, use a log scale
        if max_return >= 2:
            axis.set_yscale("log", basey=2)

        if commissions_pct is not None:
            commissions_pct = self._with_baseline(commissions_pct)
            cum_commissions_pct = self.cum_returns(commissions_pct)
            cum_commissions_pct.name = "commissions"
            cum_returns_with_baseline.name = "returns"
            cum_gross_returns = cum_returns_with_baseline.div(cum_commissions_pct.abs())
            cum_gross_returns.name = "gross returns"
            returns_breakdown = pd.concat((cum_returns_with_baseline, cum_gross_returns, cum_commissions_pct), axis=1)
            plot = returns_breakdown.plot(ax=axis, title="Cumulative Returns {0}".format(extra_label))
            if isinstance(returns_breakdown, pd.DataFrame):
                self._clear_legend(plot)
        else:
            plot = cum_returns_with_baseline.plot(ax=axis, title="Cumulative Returns {0}".format(extra_label))
            if isinstance(cum_returns_with_baseline, pd.DataFrame):
                self._clear_legend(plot)

        if benchmark is not None:
            benchmark.name = "Benchmark"
            benchmark_cum_returns = self.cum_returns(self._with_baseline(benchmark.pct_change()))
            plot = benchmark_cum_returns.plot(ax=axis)
            if isinstance(benchmark_cum_returns, pd.DataFrame):
                self._clear_legend(plot)

        if pnl is not None:
            # Cumulative pnl will generally be very similar to cumulative
            # return, but they might be different if cumulative return is
            # calcuated based on an account balance that changes due to
            # causes other than the strategies being charted (e.g. other
            # strategies, contributions/withdrawals, etc.)
            fig = plt.figure("Cumulative PNL", figsize=self.WINDOW_SIZE)
            fig.suptitle(title)
            axis = fig.add_subplot(subplot)
            plot = cum_pnl.plot(ax=axis, title="Cumulative PNL{0}".format(extra_label))
            if isinstance(cum_pnl, pd.DataFrame):
                self._clear_legend(plot)

        if show_exposures and order_exposures is not None and net_exposures is not None and abs_exposures is not None:
            fig = plt.figure("Order Exposures", figsize=self.WINDOW_SIZE)
            fig.suptitle(title)
            axis = fig.add_subplot(subplot)
            plot = order_exposures.plot(ax=axis, title="Order Exposures {0}".format(extra_label))
            if isinstance(order_exposures, pd.DataFrame):
                self._clear_legend(plot)
            fig = plt.figure("Net Exposures", figsize=self.WINDOW_SIZE)
            fig.suptitle(title)
            axis = fig.add_subplot(subplot)
            plot = net_exposures.plot(ax=axis, title="Net Exposures {0}".format(extra_label))
            if isinstance(net_exposures, pd.DataFrame):
                self._clear_legend(plot)
            fig = plt.figure("Absolute Exposures", figsize=self.WINDOW_SIZE)
            fig.suptitle(title)
            axis = fig.add_subplot(subplot)
            plot = abs_exposures.plot(ax=axis, title="Absolute Exposures {0}".format(extra_label))
            if isinstance(abs_exposures, pd.DataFrame):
                self._clear_legend(plot)

        fig = plt.figure("Drawdowns", figsize=self.WINDOW_SIZE)
        fig.suptitle(title)
        axis = fig.add_subplot(subplot)
        plot = drawdowns.plot(ax=axis, title="Drawdowns {0}".format(extra_label))
        if isinstance(drawdowns, pd.DataFrame):
            self._clear_legend(plot)
        if len(rolling_sharpe.index) > rolling_sharpe_window:
            fig = plt.figure("Rolling Sharpe", figsize=self.WINDOW_SIZE)
            fig.suptitle(title)
            axis = fig.add_subplot(subplot)
            plot = rolling_sharpe.plot(ax=axis, title="Rolling Sharpe ({0}-day) {1}".format(
                rolling_sharpe_window, extra_label))
            if isinstance(rolling_sharpe, pd.DataFrame):
                self._clear_legend(plot)

    def _get_agg_stats_text(self, agg_stats):
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
        agg_stats_text = "Aggregate Performance\n{0}\n{1}".format("="*width, agg_stats_text)
        return agg_stats_text

    def _get_plot_dimensions(self, plot_count):
        """
        Returns a tuple of rows, cols needed to accomodate the plot_count.
        """
        rows = math.ceil(math.sqrt(plot_count))
        cols = math.ceil(plot_count/rows)
        return rows, cols

    def show_shortfall(self, live_returns, simulated_returns):
        """
        Shows drawdown and equity curves comparing the live and simulated
        returns. DataFrames should include a column for each strategy being
        analyzed.
        """
        # TODO: be more specific
        title = "Shortfall Analysis"

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

            fig = plt.figure("Cumulative Returns", figsize=self.WINDOW_SIZE)
            fig.suptitle(title)
            axis = fig.add_subplot(rows, cols, i+1)
            cum_returns.plot(ax=axis, title=strategy)

            fig = plt.figure("Drawdowns", figsize=self.WINDOW_SIZE)
            fig.suptitle(title)
            axis = fig.add_subplot(rows, cols, i+1)
            drawdowns.plot(ax=axis, title=strategy)

            fig = plt.figure("Shortfall", figsize=self.WINDOW_SIZE)
            fig.suptitle(title)
            axis = fig.add_subplot(rows, cols, i+1)
            shortfall.plot(ax=axis, title=strategy, kind="area", stacked=False)

        self._save_or_show()

    def plot_arbitrary(self, series, kind="line", title=None, show=True):
        """
        Plots an arbitrary list of Series or DataFrames. The Series or
        DataFrame names are used as the chart titles.
        """
        series_list = series
        rows, cols = self._get_plot_dimensions(len(series))
        fig = plt.figure(figsize=self.WINDOW_SIZE)
        if title:
            fig.suptitle(title)
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

    def montecarlo_simulate(self, returns, n=5, preaggregate=True, title=None):
        """
        Runs a Montecarlo simulation by shuffling the dataframe of returns n
        number of times and graphing the cum_returns and drawdowns overlaid
        by the original returns. If preaggregate is True, aggregates the
        returns before the simulation, otherwise after the simulation.
        Preaggregation only randomizes by day (assuming each row is a day),
        while not preaggregating randomizes each value.
        """

        all_simulations = []

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

        cum_sim_returns = self.cum_returns(self._with_baseline(sim_returns))
        cum_returns = self.cum_returns(self._with_baseline(returns))
        fig = plt.figure("Montecarlo Simulation", figsize=self.WINDOW_SIZE,
                         tight_layout=self._tight_layout_clear_suptitle)
        if title:
            fig.suptitle(title)
        axis = fig.add_subplot(211)
        cum_sim_returns.plot(ax=axis, title="Montecarlo Cumulative Returns (n={0})".format(n), legend=False)
        cum_returns.plot(ax=axis, linewidth=4, color="black")
        axis = fig.add_subplot(212)
        self.drawdowns(cum_sim_returns).plot(ax=axis, title="Montecarlo Drawdowns (n={0})".format(n), legend=False)
        self.drawdowns(cum_returns).plot(ax=axis, linewidth=4, color="black")
