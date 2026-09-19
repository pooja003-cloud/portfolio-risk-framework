"""
Configuration loading.

Every modelling assumption lives in ``config/config.yaml``. This module turns
that file into a small typed accessor object so that the rest of the package
never hard-codes a parameter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml


def project_root() -> Path:
    """Return the repository root (the directory containing ``config/``)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "config" / "config.yaml").exists():
            return parent
    # Fall back to two levels above src/prisk/
    return here.parents[2]


DEFAULT_CONFIG_PATH = project_root() / "config" / "config.yaml"


@dataclass
class Config:
    """Typed view over the YAML configuration."""

    raw: Dict[str, Any]
    root: Path = field(default_factory=project_root)

    # -- section accessors ---------------------------------------------------
    @property
    def data(self) -> Dict[str, Any]:
        return self.raw["data"]

    @property
    def portfolios(self) -> Dict[str, Any]:
        return self.raw["portfolios"]

    @property
    def risk(self) -> Dict[str, Any]:
        return self.raw["risk"]

    @property
    def stress(self) -> Dict[str, Any]:
        return self.raw["stress"]

    @property
    def output(self) -> Dict[str, Any]:
        return self.raw["output"]

    # -- universe helpers ----------------------------------------------------
    @property
    def universe(self) -> pd.DataFrame:
        """The investable universe as a DataFrame indexed by ticker."""
        return pd.DataFrame(self.raw["universe"]).set_index("ticker")

    @property
    def tickers(self) -> List[str]:
        """Every ticker that must be downloaded."""
        return [row["ticker"] for row in self.raw["universe"]]

    @property
    def holdings(self) -> List[str]:
        """Tickers eligible to be held in the constructed portfolios."""
        return [
            row["ticker"]
            for row in self.raw["universe"]
            if str(row["role"]).startswith("holding")
        ]

    @property
    def market_ticker(self) -> str:
        """The market proxy used for beta (and as the trading calendar)."""
        return self._role_lookup("holding_and_market", default="SPY")

    @property
    def riskfree_ticker(self) -> str:
        """The cash proxy whose daily return stands in for the risk-free rate."""
        return self._role_lookup("holding_and_riskfree", default="BIL")

    def _role_lookup(self, role: str, default: str) -> str:
        for row in self.raw["universe"]:
            if row["role"] == role:
                return row["ticker"]
        return default

    def sleeve_map(self) -> Dict[str, str]:
        """ticker -> sleeve (equity, fixed_income, commodity, ...)."""
        return {row["ticker"]: row["sleeve"] for row in self.raw["universe"]}

    def sector_map(self) -> Dict[str, str]:
        """ticker -> reporting sector."""
        return {row["ticker"]: row["sector"] for row in self.raw["universe"]}

    def name_map(self) -> Dict[str, str]:
        """ticker -> long name."""
        return {row["ticker"]: row["name"] for row in self.raw["universe"]}

    # -- path helpers --------------------------------------------------------
    def path(self, key_path: str) -> Path:
        """
        Resolve a configured relative directory to an absolute path, creating
        it if necessary.

        Parameters
        ----------
        key_path : str
            Dotted key such as ``"data.cache_dir"`` or ``"output.figures_dir"``.
        """
        node: Any = self.raw
        for key in key_path.split("."):
            node = node[key]
        resolved = self.root / str(node)
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved

    @property
    def trading_days(self) -> int:
        return int(self.risk["trading_days"])


def load_config(path: str | Path | None = None) -> Config:
    """Load the master configuration file."""
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return Config(raw=raw, root=cfg_path.resolve().parents[1])
