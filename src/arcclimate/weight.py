"""
距離の重みづけ平均の係数を算出するモジュール
"""

import numpy as np
from math import *
from typing import Tuple, Optional


def get_msm_weights(
    lat: float, lon: float
) -> Tuple[float, float, float, float]:
    """MSM4地点の重みづけ計算

    Args:
      lat(float): 推計対象地点の緯度（10進法）
      lon(float): 推計対象地点の経度（10進法）

    Returns:
      Tuple[float, float, float, float]: MSM4地点の重みをタプル(SW,SE,NW,NE)で返す
    """

    # 補間計算 リストはいずれもSW南西,SE南東,NW北西,NE北東の順
    # 入力した緯度経度から周囲のMSMまでの距離を算出して、距離の重みづけ係数をリストで返す
    distances = _get_latlon_msm_distances(lat, lon)

    # MSM4地点のと目標座標の距離から4地点のウェイトを計算
    weights = _get_weights_from_distances(distances)

    return weights


def _get_latlon_msm_distances(
    lat: float, lon: float
) -> Tuple[float, float, float, float]:
    """MSM4地点と推計対象地点との距離の計算

    Args:
      lat(float): 推計対象地点の緯度（10進法）
      lon(float): 推計対象地点の経度（10進法）

    Returns:
      Tuple[float, float, float, float]: MSM4地点と推計対象地点の距離のリスト(SW,SE,NW,NE)
    """
    lat_unit = 0.05  # MSMの緯度間隔
    lon_unit = 0.0625  # MSMの経度間隔

    lat0 = lat
    lon0 = lon

    # メッシュ周囲のMSM位置（緯度経度）の取得
    lat_S = np.floor(lat / lat_unit) * lat_unit  # 南は切り下げ
    lon_W = np.floor(lon / lon_unit) * lon_unit  # 西は切り下げ

    # 緯度経度差から距離の重みづけ平均の係数を算出

    # 南西（左下）との距離
    MSM_SW = vincenty_inverse(lat0, lon0, lat_S, lon_W)

    # 南東（右下）との距離
    MSM_SE = vincenty_inverse(lat0, lon0, lat_S, lon_W + lon_unit)

    # 北西（左上）との距離
    MSM_NW = vincenty_inverse(lat0, lon0, lat_S + lat_unit, lon_W)

    # 北東（右上）との距離
    MSM_NE = vincenty_inverse(
        lat0, lon0, lat_S + lat_unit, lon_W + lon_unit)

    return MSM_SW, MSM_SE, MSM_NW, MSM_NE


def vincenty_inverse(
        lat1: float, lon1: float, lat2: float, lon2: float
) -> Optional[float]:
    """緯度経度差から距離を求めるvincenty法(逆解法)

    Args:
      lat1(float): 地点1の緯度（10進法）
      lon1(float): 地点1の経度（10進法）
      lat2(float): 地点2の緯度（10進法）
      lon2(float): 地点2の経度（10進法）

    Returns:
      float: 2点間の楕円体上の距離(計算に失敗した場合はNone) [単位:m]

    Notes:
      https://ja.wikipedia.org/wiki/Vincenty法
      https://vldb.gsi.go.jp/sokuchi/surveycalc/surveycalc/bl2stf.html
    """
    # 反復計算の上限回数
    ITERATION_LIMIT = 10000

    # 差異が無ければ0.0を返す
    if isclose(lat1, lat2) and isclose(lon1, lon2):
        return 0.0

    # 長軸半径と扁平率から短軸半径を算出する
    # 楕円体はGRS80の値
    a = 6378137.0  # 長軸半径(GRS80)
    ƒ = 1 / 298.257222101  # 扁平率(GRS80)
    b = (1 - ƒ) * a

    p1 = radians(lat1)  # φ1
    p2 = radians(lat2)  # φ2
    r1 = radians(lon1)  # λ1
    r2 = radians(lon2)  # λ2

    # 更成緯度(補助球上の緯度)
    U1 = atan((1 - ƒ) * tan(p1))
    U2 = atan((1 - ƒ) * tan(p2))

    sinU1 = sin(U1)
    sinU2 = sin(U2)
    cosU1 = cos(U1)
    cosU2 = cos(U2)

    # 2点間の経度差
    L = r2 - r1

    # λをLで初期化
    ramda = L

    # λを収束計算。反復回数の上限を10000回に設定
    for i in range(ITERATION_LIMIT):
        sinR = sin(ramda)
        cosR = cos(ramda)
        sinS = sqrt((cosU2 * sinR) ** 2 +
                    (cosU1 * sinU2 - sinU1 * cosU2 * cosR) ** 2)
        cosS = sinU1 * sinU2 + cosU1 * cosU2 * cosR
        sigma = atan2(sinS, cosS)
        sinA = cosU1 * cosU2 * sinR / sinS
        cos2A = 1 - sinA ** 2
        cos2Sm = cosS - 2 * sinU1 * sinU2 / cos2A
        C = ƒ / 16 * cos2A * (4 + ƒ * (4 - 3 * cos2A))
        ramada_p = ramda
        ramda = L + (1 - C) * ƒ * sinA * \
            (sigma + C * sinS *
             (cos2Sm + C * cosS * (-1 + 2 * cos2Sm ** 2)))

        # 偏差が.000000000001以下ならbreak
        if abs(ramda - ramada_p) <= 1e-12:
            break
    else:
        # 計算が収束しなかった場合はNoneを返す
        return None

    # λが所望の精度まで収束したら以下の計算を行う
    u2 = cos2A * (a ** 2 - b ** 2) / (b ** 2)
    A = 1 + u2 / 16384 * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))
    B = u2 / 1024 * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))
    dS = B * sinS * (cos2Sm + B / 4 * (
        cosS * (-1 + 2 * cos2Sm ** 2)
        - B / 6 * cos2Sm * (-3 + 4 * sinS ** 2) * (-3 + 4 * cos2Sm ** 2)))

    # 2点間の楕円体上の距離
    s = b * A * (sigma - dS)

    return s


def _get_weights_from_distances(
    distances: Tuple[float, float, float, float]
) -> Tuple[float, float, float, float]:
    """MSM4地点の重みづけ計算

    Args:
      distances: MSM4地点の距離のリスト

    Returns:
      Tuple[float, float, float, float]: MSM4地点の重みをタプル(SW,SE,NW,NE)で返す

    Notes:
      4地点の重みの合計は1
    """

    weights = [0.0, 0.0, 0.0, 0.0]

    if 0.0 in distances:
        weights[distances.index(0.0)] = 1

    else:
        weights = [1 / i / sum([1 / i for i in distances]) for i in distances]

    return tuple(weights)
