"""
標高処理モジュール
"""

import logging
import requests
import pandas as pd
from typing import Tuple
from meshcode import get_meshcode, get_mesh_latlon


def get_latlon_elevation(
    lat: float,
    lon: float,
    mode_elevation: str = 'api',
    mesh_elevation_master: pd.DataFrame = None
) -> float:
    """標高の取得

    Args:
      mode_elevation: 'mesh':標高補正に3次メッシュ（1㎞メッシュ）の平均標高データを使用する, 
                      'api':国土地理院のAPIを使用する
                      (Default value = 'api')
      mesh_elevation_master: 3次メッシュの標高データ (required if mode_elevation == 'mesh')
                             (Default value = None)
      lat: 推計対象地点の緯度（10進法）
      lon: 推計対象地点の経度（10進法）

    Returns:
      float: 標高
    """
    if mode_elevation == 'mesh':
        # 標高補正に3次メッシュ（1㎞メッシュ）の平均標高データを使用する場合
        # TODO : おそらく↓の lat, lon を上書きする処理は不要。
        elevation = _get_mesh_elevation(lat, lon, mesh_elevation_master)

        logging.info('入力された緯度・経度が含まれる3次メッシュの平均標高 {}m で計算します'.format(elevation))

    elif mode_elevation == 'api':
        # 国土地理院のAPIを使用して入力した緯度f経度位置の標高を返す
        try:
            logging.info('入力された緯度・経度位置の標高データを国土地理院のAPIから取得します')
            elevation = _get_elevation_from_cyberjapandata2(lat, lon)
            logging.info('成功  標高 {}m で計算します'.format(elevation))

        except:
            # 国土地理院のAPIから標高データを取得できなかった場合の判断
            # 標高補正に3次メッシュ（1㎞メッシュ）の平均標高データにフォールバック
            elevation = _get_mesh_elevation(lat, lon, mesh_elevation_master)
            logging.info('国土地理院のAPIから標高データを取得できなかったため、\n'
                         '入力された緯度・経度が含まれる3次メッシュの平均標高 {}m で計算します'.format(elevation))
    else:
        raise ValueError(mode_elevation)

    return elevation


def _get_mesh_elevation(
    lat: float,
    lon: float,
    mesh_elevation_master: pd.DataFrame
) -> float:
    """標高補正に3次メッシュ（1㎞メッシュ）の平均標高データを取得

    Args:
      lat(float): 推計対象地点の緯度（10進法）
      lon(float): 推計対象地点の経度（10進法）
      mesh_elevation_master(pd.DataFrame): 3次メッシュの標高データ

    Returns:
      float: 平均標高[m]
    """
    meshcode = get_meshcode(lat, lon)
    elevation = mesh_elevation_master.loc[int(meshcode), 'elevation']
    return elevation


def _get_elevation_from_cyberjapandata2(lat: float, lon: float) -> float:
    """緯度・経度位置の標高データを国土地理院のAPIから取得

    Args:
      lat(float): 推計対象地点の緯度（10進法）
      lon(float): 推計対象地点の経度（10進法）

    Returns:
      float: 緯度・経度位置の標高データ[m]
    """
    # 国土地理院のAPI
    cyberjapandata2_endpoint = "http://cyberjapandata2.gsi.go.jp/general/dem/scripts/getelevation.php"
    url = '{}?lon=%s&lat=%s&outtype=%s'.format(cyberjapandata2_endpoint)

    url = url % (lon, lat, 'JSON')

    resp = requests.get(url, timeout=10)
    data = resp.json()

    return data['elevation']
