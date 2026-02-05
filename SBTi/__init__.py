"""
This package helps companies and financial institutions to assess the temperature alignment of current
targets, commitments, and investment and lending portfolios, and to use this information to develop
targets for official validation by the SBTi.
"""
__version__ = "1.2.3"

from . import data
from . import utils
from . import temperature_score
from .interfaces import FINZAlignmentCategory
from .portfolio_coverage_tvp import PortfolioCoverageTVP
