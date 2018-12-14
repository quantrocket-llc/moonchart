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
import seaborn as sns
from collections import OrderedDict
import matplotlib.pyplot as plt
import warnings
from .perf import Performance, AggregatePerformance
from .base import BaseTearsheet

class Tearsheet(BaseTearsheet):
    """
    Generates a tear sheet of performance stats and graphs.
    """

    def _set_title_from_performance(self, performance):
        """
        Sets a title like "<start date> - <end date>: <securities/strategies/columns>"
        """
        min_date = performance.returns.index.min().date().isoformat()
        max_date = performance.returns.index.max().date().isoformat()
        cols = list(performance.returns.columns)
        cols = ", ".join([str(col) for col in cols])
        if len(cols) > 70:
            cols = cols[:70] + "..."
        self.suptitle = "{0} - {1}: {2}".format(
            min_date, max_date, cols)

    def from_moonshot(self, results, **kwargs):
        """
        Creates a full tear sheet from a moonshot backtest results DataFrame.

        Parameters
        ----------
        results : DataFrame
            multiindex (Field, Date) DataFrame of backtest results

        Returns
        -------
        None
        """
        performance = Performance.from_moonshot(results)
        return self.create_full_tearsheet(performance, **kwargs)

    def from_moonshot_csv(self, filepath_or_buffer, **kwargs):
        """
        Creates a full tear sheet from a moonshot backtest results CSV.

        Parameters
        ----------
        filepath_or_buffer : str or file-like object
            filepath or file-like object of the CSV

        Returns
        -------
        None
        """
        results = pd.read_csv(filepath_or_buffer,
                              parse_dates=["Date"],
                              index_col=["Field","Date"])
        return self.from_moonshot(results, **kwargs)

    def from_pnl(self, results, **kwargs):
        """
        Creates a full tear sheet from a pnl DataFrame.

        Parameters
        ----------
        results : DataFrame
            multiindex (Field, Date) DataFrame of performance results

        Returns
        -------
        None
        """
        performance = Performance.from_pnl(results)
        return self.create_full_tearsheet(performance, **kwargs)

    def from_pnl_csv(self, filepath_or_buffer, **kwargs):
        """
        Creates a full tear sheet from a pnl CSV.

        Parameters
        ----------
        filepath_or_buffer : str or file-like object
            filepath or file-like object of the CSV

        Returns
        -------
        None
        """
        results = pd.read_csv(filepath_or_buffer,
                              parse_dates=["Date"],
                              index_col=["Field","Date"])
        return self.from_pnl(results, **kwargs)

    def create_full_tearsheet(
        self,
        performance,
        include_exposures_tearsheet=True,
        include_annual_breakdown_tearsheet=True,
        montecarlo_n=None,
        montecarlo_preaggregate=True,
        title=None
        ):
        """
        Create a full tear sheet of performance results and market exposure.

        Parameters
        ----------
        performance : instance
            Performance instance

        include_exposures : bool
            whether to include a tear sheet of market exposure

        include_annual_breakdown_tearsheet : bool
            whether to include an annual breakdown of Sharpe and CAGR

        montecarlo_n : int
            how many Montecarlo simulations to run on the returns, if any

        montecarlo_preaggregate : bool
            whether Montecarlo simulator should preaggregate returns;
            ignored unless montecarlo_n is nonzero

        Returns
        -------
        None
        """
        if title:
            self.suptitle = title
        else:
            self._set_title_from_performance(performance)

        agg_performance = AggregatePerformance(performance)

        num_cols = len(performance.returns.columns)
        if num_cols > self.max_cols_for_details:
            warnings.warn("Suppressing details because there are more than {0} columns".format(
                self.max_cols_for_details))

        self.create_performance_tearsheet(performance, agg_performance)

        if include_annual_breakdown_tearsheet:
            self.create_annual_breakdown_tearsheet(performance, agg_performance)

        if include_exposures_tearsheet and any([exposures is not None for exposures in (
            performance.net_exposures, performance.abs_exposures)]):
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

        num_cols = len(performance.returns.columns)
        show_details = num_cols > 1 and num_cols <= self.max_cols_for_details

        if show_details:
            performance.fill_performance_cache()

        self._create_agg_performance_textbox(agg_performance)

        width, height = self.figsize
        # cut height in half if not showing details
        if not show_details:
            height /= 2

        self._create_performance_plots(
            agg_performance,
            subplot=211 if show_details else 111,
            extra_label="(Aggregate)" if show_details else "",
            figsize=(width, height)
        )

        if show_details:
            self._create_performance_plots(performance, subplot=212, extra_label="(Details)")
            self._create_detailed_performance_bar_charts(
                performance, extra_label="(Details)")

    def _create_detailed_performance_bar_charts(self, performance, extra_label):

        # cut height in half since only one chart per figure
        width, height = self.figsize
        figsize = width, height/2

        if performance.pnl is not None:
            fig = plt.figure("PNL {0}".format(extra_label), figsize=figsize)
            fig.suptitle(self.suptitle, **self.suptitle_kwargs)
            axis = fig.add_subplot(111)
            pnl = performance.pnl.sum().sort_values(inplace=False)
            if performance.commissions is not None:
                pnl.name = "pnl"
                commissions = performance.commissions.sum()
                commissions.name = "commissions"
                gross_pnl = pnl + commissions
                gross_pnl.name = "gross pnl"
                try:
                    pnl = pd.concat((pnl, gross_pnl, commissions), axis=1, sort=True)
                except TypeError:
                    # sort was introduced in pandas 0.23
                    pnl = pd.concat((pnl, gross_pnl, commissions), axis=1)
            pnl.plot(
                ax=axis, kind="bar", title="PNL {0}".format(extra_label))

        fig = plt.figure("CAGR {0}".format(extra_label), figsize=figsize)
        fig.suptitle(self.suptitle, **self.suptitle_kwargs)
        axis = fig.add_subplot(111)
        performance.cagr.sort_values(inplace=False).plot(
            ax=axis, kind="bar", title="CAGR {0}".format(extra_label))

        fig = plt.figure("Sharpe {0}".format(extra_label), figsize=figsize)
        fig.suptitle(self.suptitle, **self.suptitle_kwargs)
        axis = fig.add_subplot(111)
        performance.sharpe.sort_values(inplace=False).plot(
            ax=axis, kind="bar", title="Sharpe {0}".format(extra_label))

        fig = plt.figure("Max Drawdown {0}".format(extra_label), figsize=figsize)
        fig.suptitle(self.suptitle, **self.suptitle_kwargs)
        axis = fig.add_subplot(111)
        performance.max_drawdown.sort_values(inplace=False).plot(
            ax=axis, kind="bar", title="Max drawdown {0}".format(extra_label))

    def _create_agg_performance_textbox(self, agg_performance):
        agg_stats = OrderedDict()
        agg_stats_text = ""

        if agg_performance.pnl is not None:
            agg_stats["PNL"] = round(agg_performance.pnl.sum(), 4)
        if agg_performance.commissions is not None:
            agg_stats["Commissions"] = round(agg_performance.commissions.sum(), 4)

        agg_stats["CAGR"] = agg_performance.cagr
        agg_stats["Sharpe"] = agg_performance.sharpe
        agg_stats["Max Drawdown"] = agg_performance.max_drawdown

        agg_stats_text = self._get_agg_stats_text(agg_stats)
        fig = plt.figure("Aggregate Performance")
        fig.suptitle(self.suptitle, **self.suptitle_kwargs)
        self.plot_textbox(fig, agg_stats_text)

    def plot_textbox(self, fig, text):
        with sns.axes_style("white", {'axes.linewidth': 0}):
            axis = fig.add_subplot(111)
            axis.get_xaxis().set_visible(False)
            axis.get_yaxis().set_visible(False)
            axis.text(0.1, 0.4, text,
                 family="monospace",
                 fontsize="xx-large"
                 )
    def create_exposures_tearsheet(self, performance, agg_performance):
        """
        Create a tearsheet of market exposure.
        """
        agg_performance.fill_performance_cache()

        num_cols = len(performance.returns.columns)
        show_details = num_cols > 1 and num_cols <= self.max_cols_for_details
        if show_details:
            performance.fill_performance_cache()

        self._create_agg_exposures_textbox(agg_performance)

        width, height = self.figsize
        # cut height in half if not showing details
        if not show_details:
            height /= 2

        self._create_exposures_plots(
            agg_performance,
            subplot=211 if show_details else 111,
            extra_label="(Aggregate)" if show_details else "",
            figsize=(width, height))

        if show_details:
            self._create_exposures_plots(performance, subplot=212, extra_label="(Details)")
            self._create_detailed_exposures_bar_charts(performance, extra_label="(Details)")

    def _create_agg_exposures_textbox(self, agg_performance):

        agg_stats = OrderedDict()
        agg_stats_text = ""

        if agg_performance.net_exposures is not None:
            avg_net_exposures = agg_performance.get_avg_exposure(agg_performance.net_exposures)
            agg_stats["Avg Net Exposure"] = round(avg_net_exposures, 3)

        if agg_performance.abs_exposures is not None:
            avg_abs_exposures = agg_performance.get_avg_exposure(agg_performance.abs_exposures)
            norm_cagr = agg_performance.get_normalized_cagr(agg_performance.cagr, avg_abs_exposures)
            agg_stats["Avg Absolute Exposure"] = round(avg_abs_exposures, 3)
            agg_stats["Normalized CAGR (CAGR/Exposure)"] = round(norm_cagr, 3)

        agg_stats_text = self._get_agg_stats_text(agg_stats, title="Aggregate Exposure")
        fig = plt.figure("Aggregate Exposure")
        fig.suptitle(self.suptitle, **self.suptitle_kwargs)
        self.plot_textbox(fig, agg_stats_text)

    def _create_exposures_plots(self, performance, subplot, extra_label, figsize=None):

        figsize = figsize or self.figsize

        if isinstance(performance.returns, pd.DataFrame):
            num_series = len(performance.returns.columns)
            if num_series > 6:
                sns.set_palette(sns.color_palette("hls", num_series))

        if performance.net_exposures is not None:
            fig = plt.figure("Net Exposures", figsize=figsize)
            fig.suptitle(self.suptitle, **self.suptitle_kwargs)
            axis = fig.add_subplot(subplot)
            plot = performance.net_exposures.round(2).plot(ax=axis, title="Net Exposures {0}".format(extra_label))
            if isinstance(performance.net_exposures, pd.DataFrame):
                self._clear_legend(plot)

        if performance.abs_exposures is not None:
            fig = plt.figure("Absolute Exposures", figsize=figsize)
            fig.suptitle(self.suptitle, **self.suptitle_kwargs)
            axis = fig.add_subplot(subplot)
            plot = performance.abs_exposures.round(2).plot(ax=axis, title="Absolute Exposures {0}".format(extra_label))
            if isinstance(performance.abs_exposures, pd.DataFrame):
                self._clear_legend(plot)

        if performance.total_holdings is not None:
            fig = plt.figure("Total Holdings", figsize=figsize)
            fig.suptitle(self.suptitle, **self.suptitle_kwargs)
            axis = fig.add_subplot(subplot)
            plot = performance.total_holdings.plot(ax=axis, title="Total Holdings {0}".format(extra_label))
            if isinstance(performance.total_holdings, pd.DataFrame):
                self._clear_legend(plot)

        if isinstance(performance.returns, pd.DataFrame) and num_series > 6:
            sns.set()

    def _create_detailed_exposures_bar_charts(self, performance, extra_label):

        # cut height in half since only one chart per figure
        width, height = self.figsize
        figsize = width, height/2

        if performance.abs_exposures is not None:
            fig = plt.figure("Avg Absolute Exposure {0}".format(extra_label), figsize=figsize)
            fig.suptitle(self.suptitle, **self.suptitle_kwargs)
            avg_abs_exposures = performance.get_avg_exposure(performance.abs_exposures)
            axis = fig.add_subplot(111)
            avg_abs_exposures.sort_values(inplace=False).plot(
                ax=axis, kind="bar", title="Avg Absolute Exposure {0}".format(extra_label))

        if performance.net_exposures is not None:
            fig = plt.figure("Avg Net Exposure {0}".format(extra_label), figsize=figsize)
            fig.suptitle(self.suptitle, **self.suptitle_kwargs)
            avg_net_exposures = performance.get_avg_exposure(performance.net_exposures)
            axis = fig.add_subplot(111)
            avg_net_exposures.sort_values(inplace=False).plot(
            ax=axis, kind="bar", title="Avg Net Exposure {0}".format(extra_label))

        if performance.abs_exposures is not None:
            norm_cagrs = performance.get_normalized_cagr(performance.cagr, avg_abs_exposures)
            fig = plt.figure("Normalized CAGR (CAGR/Exposure) {0}".format(extra_label), figsize=figsize)
            fig.suptitle(self.suptitle, **self.suptitle_kwargs)
            axis = fig.add_subplot(111)
            norm_cagrs.sort_values(inplace=False).plot(
                ax=axis, kind="bar", title="Normalized CAGR (CAGR/Exposure) {0}".format(extra_label))

        if performance.total_holdings is not None:
            fig = plt.figure("Avg Total Holdings {0}".format(extra_label), figsize=figsize)
            fig.suptitle(self.suptitle, **self.suptitle_kwargs)
            avg_total_holdings = performance.get_avg_total_holdings(performance.total_holdings)
            axis = fig.add_subplot(111)
            avg_total_holdings.sort_values(inplace=False).plot(
                ax=axis, kind="bar", title="Avg Total Holdings {0}".format(extra_label))

    def create_annual_breakdown_tearsheet(self, performance, agg_performance):
        """
        Creates agg/detailed bar charts showing CAGR and Sharpe by year.
        """
        agg_performance.fill_performance_cache()

        num_cols = len(performance.returns.columns)
        show_details = num_cols > 1 and num_cols <= self.max_cols_for_details
        if show_details:
            performance.fill_performance_cache()

        width, height = self.figsize
        # cut height in half if not showing details
        if not show_details:
            height /= 2

        self._create_annual_breakdown_plots(
            agg_performance,
            subplot=211 if show_details else 111,
            extra_label="(Aggregate)" if show_details else "",
            figsize=(width, height))

        if show_details:
            self._create_annual_breakdown_plots(performance, subplot=212, extra_label="(Details)")

    def _create_annual_breakdown_plots(self, performance, subplot, extra_label, figsize=None):

        figsize = figsize or self.figsize

        if isinstance(performance.returns, pd.DataFrame):
            num_series = len(performance.cum_returns.columns)
            if num_series > 6:
                sns.set_palette(sns.color_palette("hls", num_series))

        grouped_returns = performance.returns.groupby(performance.returns.index.year)
        cagrs_by_year = grouped_returns.apply(lambda x: performance.get_cagr(
            performance.get_cum_returns(x)))
        sharpes_by_year = grouped_returns.apply(performance.get_sharpe)

        fig = plt.figure("CAGR by Year", figsize=figsize)
        fig.suptitle(self.suptitle, **self.suptitle_kwargs)
        axis = fig.add_subplot(subplot)
        plot = cagrs_by_year.plot(ax=axis, kind="bar", title="CAGR by Year {0}".format(extra_label))
        if isinstance(cagrs_by_year, pd.DataFrame):
            self._clear_legend(plot)

        fig = plt.figure("Sharpe by Year", figsize=figsize)
        fig.suptitle(self.suptitle, **self.suptitle_kwargs)
        axis = fig.add_subplot(subplot)
        plot = sharpes_by_year.plot(ax=axis, kind="bar", title="Sharpe by Year {0}".format(extra_label))
        if isinstance(sharpes_by_year, pd.DataFrame):
            self._clear_legend(plot)

        if isinstance(performance.returns, pd.DataFrame) and num_series > 6:
            sns.set()

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

        try:
            sim_returns = pd.concat(all_simulations, axis=1, sort=False)
        except TypeError:
            # sort was introduced in pandas 0.23
            sim_returns = pd.concat(all_simulations, axis=1)

        if not preaggregate:
            returns = returns.sum(axis=1)

        cum_sim_returns = performance.get_cum_returns(performance.with_baseline(sim_returns))
        cum_returns = performance.get_cum_returns(performance.with_baseline(returns))
        fig = plt.figure("Montecarlo Simulation", figsize=self.figsize)
        fig.suptitle(self.suptitle, **self.suptitle_kwargs)
        axis = fig.add_subplot(211)
        cum_sim_returns.plot(ax=axis, title="Montecarlo Cumulative Returns (n={0})".format(n), legend=False)
        cum_returns.plot(ax=axis, linewidth=4, color="black")
        axis = fig.add_subplot(212)
        performance.get_drawdowns(cum_sim_returns).plot(ax=axis, title="Montecarlo Drawdowns (n={0})".format(n), legend=False)
        performance.get_drawdowns(cum_returns).plot(ax=axis, linewidth=4, color="black")
