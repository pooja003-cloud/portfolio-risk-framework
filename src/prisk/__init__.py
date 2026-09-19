"""
prisk — Multi-Asset Portfolio Risk and Stress-Testing Framework.

A research framework for constructing hypothetical multi-asset portfolios and
measuring their market risk under normal and stressed conditions.

DISCLAIMER
----------
Everything in this package is an independent research exercise built on
publicly available price data. The portfolios are hypothetical. Nothing here
represents a real client account, real assets under management, investment
advice, or a forecast.
"""

__version__ = "1.0.0"
__author__ = "pooja003-cloud"

from prisk.config import Config, load_config

__all__ = ["Config", "load_config", "__version__"]
