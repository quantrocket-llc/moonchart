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

import math
import matplotlib.pyplot as plt
from matplotlib import cycler
from matplotlib.backends.backend_pdf import PdfPages

class BaseTearsheet(object):
    """
    Base class for tear sheets.
    """
    DEFAULT_TITLE = "Performance tear sheet"

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

        self.suptitle = self.DEFAULT_TITLE
        self.suptitle_kwargs = {
            "bbox": dict(facecolor="#e1e1e6", edgecolor='#aaaaaa', alpha=0.5)}

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

    def _get_plot_dimensions(self, plot_count):
        """
        Returns a tuple of rows, cols needed to accomodate the plot_count.
        """
        rows = math.ceil(math.sqrt(plot_count))
        cols = math.ceil(plot_count/rows)
        return rows, cols

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
