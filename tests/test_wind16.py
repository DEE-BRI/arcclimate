import os
import sys
import pytest

sys.path.insert(0, os.path.realpath(
    os.path.join(os.path.basename(__file__), '..', 'src')))
from arcclimate.wind import get_wind16

def test_wind16():
    spd, dir = get_wind16([1.0], [1.0])
    assert spd[0] == pytest.approx(1.4141456, 0.0001)
    assert dir[0] == 180.0 + 45.0
