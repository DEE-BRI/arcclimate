import os
import sys
import pytest

sys.path.insert(0, os.path.realpath(
    os.path.join(os.path.basename(__file__), '..', 'src')))

from arcclimate.weight import vincenty_inverse


def test_vincenty_inverse():
    """2点間の距離の計算のテスト(GRS80)

    Notes:
        期待値は国土地理院の計算プログラムから取得しました。
        https://vldb.gsi.go.jp/sokuchi/surveycalc/surveycalc/bl2stf.html
    """
    lat1 = 36.10377477777778
    lon1 = 140.08785502777778
    lat2 = 35.65502847222223
    lon2 = 139.74475044444443

    L = vincenty_inverse(lat1, lon1, lat2, lon2)
    assert L == pytest.approx(58643.804, 0.01)


def test_vincenty_inverse_same_position():
    """2点間の距離の計算のテスト(同じ拠点)
    """
    lat1 = 36.10377477777778
    lon1 = 140.08785502777778
    lat2 = 36.10377477777778
    lon2 = 140.08785502777778

    L = vincenty_inverse(lat1, lon1, lat2, lon2)
    assert L == 0.0
