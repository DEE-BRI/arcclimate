"""
メッシュコード処理モジュール
"""

from math import floor
from typing import Tuple


def get_meshcode(lat: float, lon: float) -> str:
    """経度・緯度からメッシュコード(1 次、2 次、3 次)を取得

    Args:
      lat(float): 緯度(10 進数)
      lon(float): 経度(10 進数)

    Returns:
      str: 1 次メッシュコード(4 桁), 2 次メッシュコード(2 桁), 3 次メッシュコード(2 桁) 計8桁

    """
    lt = lat * 3.0 / 2.0
    lg = lon
    y1 = floor(lt)
    x1 = floor(lg)
    lt = (lt - y1) * 8.0
    lg = (lg - x1) * 8.0
    y2 = floor(lt)
    x2 = floor(lg)
    lt = (lt - y2) * 10.0
    lg = (lg - x2) * 10.0
    y3 = floor(lt)
    x3 = floor(lg)

    code1 = 0
    code1 += int(y1) % 100 * 100
    code1 += int(x1) % 100 * 1

    code2 = 0
    code2 += int(y2) * 10
    code2 += int(x2) * 1

    code3 = 0
    code3 += int(y3) * 10
    code3 += int(x3) * 1

    return str(code1 * 10000 + code2 * 100 + code3)


def get_mesh_latlon(meshcode: str) -> Tuple[float, float]:
    """メッシュコードから経度緯度への変換

    Args:
      meshcode(str): メッシュコード

    Returns:
      Tuple[float, float]: 緯度(10 進数), 経度(10 進数)
    """
    # メッシュコードから緯度経度を計算(中心ではなく南西方向の座標が得られる)
    y1 = int(meshcode[:2])
    x1 = int(meshcode[2:4])
    y2 = int(meshcode[4])
    x2 = int(meshcode[5])
    y3 = int(meshcode[6])
    x3 = int(meshcode[7])

    # 南西方向の座標からメッシュ中心の緯度を算出
    lat = ((y1 * 80 + y2 * 10 + y3) * 30 / 3600) + 15 / 3600

    # 南西方向の座標からメッシュ中心の経度を算出
    lon = (((x1 * 80 + x2 * 10 + x3) * 45 / 3600) + 100) + 22.5 / 3600

    return lat, lon
