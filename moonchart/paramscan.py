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

import pandas as pd
import seaborn as sns
from collections import OrderedDict
import matplotlib
import matplotlib.pyplot as plt
from .perf import DailyPerformance
from .base import BaseTearsheet
from .exceptions import MoonchartError

class ParamscanTearsheet(BaseTearsheet):

    def _from_moonshot(self, results):
        """
        Creates a param scan tear sheet from a moonshot param scan results
        DataFrame.
        """
        if "StrategyOrDate" not in results.columns:
            raise MoonchartError(
                "DataFrame contains no 'StrategyOrDate' column, "
                "are you sure this came from a Moonshot parameter scan?")
        idx_cols=list(results.columns)
        idx_cols.remove("Value")
        results = results.set_index(idx_cols)
        results = results["Value"] # 1-col DataFrame to Series
        idx_cols.remove("Field")
        idx_cols.remove("StrategyOrDate")
        params = idx_cols

        # preserve param order (pandas unstack() will lexsort)
        first_field = results.index.get_level_values("Field")[0]
        first_strategy = results.index.get_level_values("StrategyOrDate")[0]
        desired_cols = results.loc[first_field].loc[first_strategy].index

        results = results.unstack(level=params)
        results =results.reindex(columns=desired_cols)

        return self.create_full_tearsheet(results)

    @classmethod
    def from_moonshot_csv(cls, filepath_or_buffer, figsize=None, pdf_filename=None):
        """
        Create a parameter scan tear sheet from a Moonshot paramscan results CSV.

        Parameters
        ----------
        filepath_or_buffer : str or file-like object
            filepath or file-like object of the CSV

        figsize : tuple (width, height), optional
            (width, height) of matplotlib figure. Default is (16, 12)

        pdf_filename : string, optional
            save tear sheet to this filepath as a PDF instead of displaying

        Returns
        -------
        None

        Examples
        --------
        >>> from moonshot import ParamscanTearsheet
        >>> ParamscanTearsheet.from_moonshot_csv("paramscan_results.csv")
        """
        results = pd.read_csv(filepath_or_buffer)
        t = cls(figsize=figsize, pdf_filename=pdf_filename)
        return t._from_moonshot(results)

    def create_full_tearsheet(self, results, heatmap_2d=True):
        """
        Create a full tear sheet of param scan results.

        Parameters
        ----------
        results : DataFrame
            multi-index (Field, StrategyOrDate) DataFrame of param scan results,
            with param vals as (possibly multi-level) columns

        heatmap_2d : bool
            use heat maps for 2 paramscans; if False, use bar charts

        Returns
        -------
        None
        """
        returns = results.loc["AggReturn"]
        returns.index = pd.to_datetime(returns.index)
        returns.index.name = "Date"

        summary = OrderedDict()
        if results.columns.nlevels == 2:
            param1, param2 = results.columns.names
            summary["Parameter 1"] = param1
            summary["Parameter 2"] = param2
            params_title = " / ".join(results.columns.names)
        else:
            summary["Parameter"] = results.columns.name
            params_title = results.columns.name

        summary["Start Date"] = returns.index.min().date().isoformat()
        summary["End Date"] = returns.index.max().date().isoformat()

        with sns.axes_style("white"):

            fig = plt.figure("Parameter Scan Summary", figsize=(6,6))

            axis = fig.add_subplot(111)
            axis.axis("off")

            table = axis.table(
                cellText=[[v] for v in summary.values()],
                rowLabels=list(summary.keys()),
                loc="center")

            table.scale(1, 2)
            table.set_fontsize("large")

        # Plot 1d bar charts or 2d heat maps
        if results.columns.nlevels == 2 and heatmap_2d:
            self._create_2d_heatmaps(results)
        else:
            self._create_1d_bar_charts(results)

        # Plot performance plots
        performance = DailyPerformance(returns)

        # cut height in half since only one chart per figure
        width, height = self.figsize
        figsize = width, height/2

        self._create_returns_plots(performance, subplot=111, extra_label=" (Aggregate)",
                                   figsize=figsize, legend_title=params_title)

        self._save_or_show()

    def _create_1d_bar_charts(self, results):
        """
        Creates bar charts for 1d param scans.
        """
        fields = (
            ("Cagr", "CAGR", self._y_format_as_percentage),
            ("Sharpe", "Sharpe", self._y_format_at_least_two_decimal_places),
            ("MaxDrawdown", "Max Drawdown", self._y_format_as_percentage),
            ("AbsExposure", "Absolute Exposure", self._y_format_as_percentage),
            ("NormalizedCagr", "Normalized CAGR (CAGR/Absolute Exposure)", self._y_format_as_percentage),
            ("TotalHoldings", "Avg Daily Holdings", None),
        )

        color_palette = sns.color_palette()
        num_series = len(results.columns)
        if num_series > 6:
            color_palette = sns.color_palette("hls", num_series)

        with sns.color_palette(color_palette):
            rows, cols = self._get_plot_dimensions(len(fields))
            # dynamically adjust window height based on number of plots
            width = max((self.figsize[0], cols*5+2))
            height = max((self.figsize[1], rows*2+3))
            fig = plt.figure("Parameter Scan Results", figsize=(width, height))

            for i, (field, title, y_formatter) in enumerate(fields):
                if field not in results.index.get_level_values("Field"):
                    continue
                field_results = results.loc[field]
                field_results.index.name = "Strategy"
                field_results = field_results.T
                field_results.index = field_results.index.astype(str).str.wrap(10)

                axis = fig.add_subplot(rows, cols, i + 1)
                if y_formatter is not None:
                    y_formatter(axis)
                plot = field_results.plot(ax=axis, kind="bar", title=title)
                if isinstance(field_results, pd.DataFrame):
                    self._clear_legend(plot, legend_title="Strategy")

                # Remove legend on all but the upper right subplot, to
                # clean up appearance
                is_upper_right =  i+1 == cols
                if not is_upper_right:
                    plot.legend_.remove()

                # Hide x-axis label except on last row to save space
                is_last_row = (i+1) > (rows-1) * cols
                if not is_last_row:
                    x_axis = axis.axes.get_xaxis()
                    x_label = x_axis.get_label()
                    x_label.set_visible(False)

            fig.tight_layout()
            fig.subplots_adjust(top=0.9)

    def _create_2d_heatmaps(self, results):
        """
        Creates heat maps for 2d param scans. There is one figure per field,
        with subplots for each strategy.
        """
        fields = (
            ("Cagr", "CAGR"),
            ("Sharpe", "Sharpe"),
            ("MaxDrawdown", "Max DD"),
            ("AbsExposure", "Abs Exposure"),
            ("NormalizedCagr", "Normalized CAGR"),
            ("TotalHoldings", "Avg Daily Holdings"),
        )
        fields = OrderedDict(fields)

        fig = None

        for i, (field, label) in enumerate(fields.items()):
            field_results = results.loc[field]
            field_results.index.name = "Strategy"
            field_results = field_results.T
            strategies = field_results.columns
            num_strategies = len(strategies)
            num_fields = len(fields)
            if not fig:
                rows, cols = self._get_plot_dimensions(num_strategies*num_fields)
                # dynamically adjust window height based on number of plots
                width = max((self.figsize[0], cols*5+2))
                height = max((self.figsize[1], rows*2+3))
                fig = plt.figure("Parameter Scan Heat Maps", figsize=(width, height))

            for ii, strategy in enumerate(strategies):
                strategy_results = field_results[strategy]

                title = "{0} ({1})".format(label, strategy)
                axis = fig.add_subplot(rows, cols, i*num_strategies + ii+ 1, title=title)
                strategy_results = strategy_results.unstack()
                strategy_results.index = strategy_results.index.astype(str).str.wrap(10)
                sns.heatmap(strategy_results, annot=True,
                    annot_kws={"size": 9},
                    center=0.0,
                    cbar=False,
                    ax=axis,
                    cmap=matplotlib.cm.RdYlGn)

        fig.tight_layout()