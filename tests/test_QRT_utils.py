import pytest
import pandas as pd

from qrt.qrt_utils import most_recent_positions, portfolio_beta, forced_hedge, risk, load_returns_from, eur_usd
from qrt.constants import RUA

pos = most_recent_positions(RUA)

def test_most_recent_positions_and_portfolio_beta():
    try:
        pos = most_recent_positions(RUA)
        assert isinstance(portfolio_beta(pos, RUA), float)
    except FileNotFoundError:
        pytest.skip("No target files present")
    assert not pos.empty

def test_forced_hedge():
    assert isinstance(forced_hedge(pd.Series({RUA.benchmark: 1}), RUA), float)

def test_risk():
    assert isinstance(risk(pd.Series({RUA.benchmark: 1})), float)

def test_load_returns():
    assert not load_returns_from(['AAPL.OQ']).empty

def test_eur_usd():
    assert 0.5 < eur_usd() < 3.0