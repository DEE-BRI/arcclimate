import os
import sys
import pytest
import pandas as pd

sys.path.insert(0, os.path.realpath(
    os.path.join(os.path.basename(__file__), '..', 'src')))
from arcclimate.EA import get_smoothing_months, patch_representataive_years

def test_get_smoothing_months():
    # すべて2000年: 閏年のため3月はスムージング対象
    assert list(get_smoothing_months([2000]*12)) == [(3,2000,2000)]
    # すべて2001年: 閏年ではないためスムージング対象なし
    assert list(get_smoothing_months([2001]*12)) == []
    # 毎月代表年が違う場合
    assert list(get_smoothing_months(list(range(2000,2012)))) \
        == [
            (1,2011,2000),
            (2,2000,2001),
            (3,2001,2002),
            (4,2002,2003),
            (5,2003,2004),
            (6,2004,2005),
            (7,2005,2006),
            (8,2006,2007),
            (9,2007,2008),
            (10,2008,2009),
            (11,2009,2010),
            # (12,2010,2011),
            ]
