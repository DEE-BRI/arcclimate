import os
import sys
import pytest

sys.path.insert(0, os.path.realpath(
    os.path.join(os.path.basename(__file__), '..', 'src')))

from arcclimate.meshcode import get_meshcode, get_mesh_latlon


def test_get_meshcode():
    # 緯度経度を設定する
    lat = 36
    lon = 138

    # メッシュコードを取得する
    meshcode = get_meshcode(lat, lon)

    # 正しいメッシュコードが取得できることを確認する
    assert "54380000" == meshcode


def test_get_mesh_latlon():
    # 緯度経度を設定する
    latitude = 36
    longitude = 138

    # メッシュコードを取得する
    lat, lon = get_mesh_latlon("54380000")

    # 正しいメッシュコードが取得できることを確認する
    assert lat == pytest.approx(36, 0.001)
    assert lon == pytest.approx(138, 0.001)
