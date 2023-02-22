"""
Performance tear sheets for QuantRocket.

Classes
-------
Tearsheet
    Create a tear sheet of performance stats and graphs for backtest
    results or live PNL.

ParamscanTearsheet
    Create a tear sheet from a parameter scan results CSV from Moonshot or
    Zipline.

ShortfallTearsheet
    Create a tear sheet of performance stats and plots highlighting the
    shortfall between simulated or benchmark results and actual results.

DailyPerformance
    Class representing daily performance and derived statistics.

AggregateDailyPerformance
    Class representing aggregate daily performance.

Modules
-------
utils
    Utility functions for performance analysis.
"""
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

from .utils import set_default_palette
set_default_palette()

from .tearsheet import Tearsheet
from .perf import DailyPerformance, AggregateDailyPerformance
from .paramscan import ParamscanTearsheet
from .shortfall import ShortfallTearsheet
from . import utils

__all__ = [
    'Tearsheet',
    'DailyPerformance',
    'AggregateDailyPerformance',
    'ParamscanTearsheet',
    'ShortfallTearsheet',
    'utils',
]