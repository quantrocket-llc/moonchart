from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

from .utils import set_default_palette
set_default_palette()

from .tearsheet import Tearsheet
from .perf import DailyPerformance, AggregateDailyPerformance
from .paramscan import ParamscanTearsheet
from .shortfall import ShortfallTearsheet
