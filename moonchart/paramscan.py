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
from .perf import Performance
from .base import BaseTearsheet

class ParamscanTearsheet(BaseTearsheet):

    def from_moonshot(self, results, **kwargs):
        """
        Creates a param scan tear sheet from a moonshot param scan results
        DataFrame.

        Parameters
        ----------
        results : DataFrame
            DataFrame of paramscan results with columns
            (Field, StrategyOrDate, param1[, param2], Value)

        Returns
        -------
        None
        """
        idx_cols=list(results.columns)
        idx_cols.remove("Value")
        results = results.set_index(idx_cols)
        results = results["Value"] # 1-col DataFrame to Series
        idx_cols.remove("Field")
        idx_cols.remove("StrategyOrDate")
        params = idx_cols
        results = results.unstack(level=params)

        return self.create_full_tearsheet(results, **kwargs)

    def from_moonshot_csv(self, filepath_or_buffer, **kwargs):
        """
        Creates a full tear sheet from a moonshot paramscan results CSV.

        Parameters
        ----------
        filepath_or_buffer : str or file-like object
            filepath or file-like object of the CSV

        Returns
        -------
        None
        """
        results = pd.read_csv(filepath_or_buffer)
        return self.from_moonshot(results, **kwargs)

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
        # Set suptitle
        returns = results.loc["AggReturn"]
        returns.index = pd.to_datetime(returns.index)
        returns.index.name = "Date"
        min_date = returns.index.min().date().isoformat()
        max_date = returns.index.max().date().isoformat()

        if results.columns.nlevels == 2:
            params_title = " / ".join(results.columns.names)
        else:
            params_title = results.columns.name

        self.suptitle = "{0} ({1} - {2})".format(
            params_title, min_date, max_date)

        # Plot 1d bar charts or 2d heat maps
        if results.columns.nlevels == 2 and heatmap_2d:
            self._create_2d_heatmaps(results)
        else:
            self._create_1d_bar_charts(results)

        # Plot performance plots
        performance = Performance(returns)
        performance.fill_performance_cache()
        self._create_performance_plots(performance, subplot=111, extra_label=" (Aggregate)")

        self._save_or_show()

    def _create_1d_bar_charts(self, results):
        """
        Creates bar charts for 1d param scans.
        """
        fields = (
            ("Cagr", "CAGR"),
            ("Sharpe", "Sharpe"),
            ("MaxDrawdown", "Max Drawdown"),
            ("AbsExposure", "Absolute Exposure"),
            ("NormalizedCagr", "Normalized CAGR (CAGR/Exposure)")
        )
        fields = OrderedDict(fields)

        rows, cols = self._get_plot_dimensions(len(fields))
        # dynamically adjust window height based on number of plots
        width = max((self.window_size[0], cols*5+2))
        height = max((self.window_size[1], rows*2+3))
        fig = plt.figure("Parameter Scan Results", figsize=(width, height))
        fig.suptitle(self.suptitle, **self.suptitle_kwargs)

        for i, field in enumerate(list(fields.keys())):
            field_results = results.loc[field]
            field_results.index.name = "Strategy"
            field_results = field_results.T

            axis = fig.add_subplot(rows, cols, i + 1)
            plot = field_results.plot(ax=axis, kind="bar", title=fields[field],
                                      fontsize="small")
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
            ("NormalizedCagr", "Normalized CAGR")
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
                width = max((self.window_size[0], cols*5+2))
                height = max((self.window_size[1], rows*2+3))
                fig = plt.figure("Parameter Scan Heat Maps", figsize=(width, height), tight_layout=self._tight_layout_clear_suptitle)
                fig.suptitle(self.suptitle, **self.suptitle_kwargs)

            for ii, strategy in enumerate(strategies):
                strategy_results = field_results[strategy]

                title = "{0} ({1})".format(label, strategy)
                axis = fig.add_subplot(rows, cols, i*num_strategies + ii+ 1, title=title)
                sns.heatmap(strategy_results.unstack(), annot=True,
                    annot_kws={"size": 9},
                    center=0.0,
                    cbar=False,
                    ax=axis,
                    cmap=matplotlib.cm.RdYlGn)
