"""
メソ数値予想モデルGPV(MSM)に関するモジュール
"""

import logging
from typing import Tuple
import numpy as np
import os
import pandas as pd
from typing import List, Iterable
from urllib import request


def get_MSM(lat: float, lon: float) -> Tuple[int, int, int, int]:
    """メッシュ周囲のMSM位置（緯度経度）と番号（北始まり0～、西始まり0～）の取得

    Args:
      lat(float): 推計対象地点の緯度（10進法）
      lon(float): 推計対象地点の経度（10進法）

    Returns:
      Tuple[int, int, int, int]: メッシュ周囲のMSM位置（緯度経度）と番号（北始まり0～、西始まり0～）

    """
    lat_unit = 0.05  # MSMの緯度間隔
    lon_unit = 0.0625  # MSMの経度間隔

    # 緯度⇒メッシュ番号
    lat_S = np.floor(lat / lat_unit) * lat_unit  # 南は切り下げ
    MSM_S = int(np.round((47.6 - lat_S) / lat_unit))
    MSM_N = int(MSM_S - 1)

    # 経度⇒メッシュ番号
    lon_W = np.floor(lon / lon_unit) * lon_unit  # 西は切り下げ
    MSM_W = int(np.round((lon_W - 120) / lon_unit))
    MSM_E = int(MSM_W + 1)

    return MSM_S, MSM_N, MSM_W, MSM_E


def load_msm_files(
    lat: float, lon: float, msm_file_dir: str
) -> Tuple[List[str], List[pd.DataFrame]]:
    """MSMファイルを読み込みます。必要に応じてダウンロードを行います。

    Args:
      lat(float): 推計対象地点の緯度（10進法）
      lon(float): 推計対象地点の経度（10進法）
      msm_file_dir(str): MSMファイルの格納ディレクトリ

    Returns:
      msm_list(list[str]): 読み込んだMSMファイルの一覧
      df_msm_list(list[pd.DataFrame]): 読み込んだデータフレームのリスト
    """
    # 計算に必要なMSMを算出して、ダウンロード⇒ファイルpathをリストで返す

    # 保存先ディレクトリの作成
    os.makedirs(msm_file_dir, exist_ok=True)

    # 必要なMSMファイル名の一覧を緯度経度から取得
    msm_list = get_msm_requirements(lat, lon)

    # ダウンロードが必要なMSMの一覧を取得
    msm_list_missed = get_missing_msm(msm_list, msm_file_dir)

    # ダウンロード
    download_msm_files(msm_list_missed, msm_file_dir)

    # MSMファイル読み込み
    df_msm_list = []
    for msm in msm_list:
        # MSMファイルのパス
        msm_path = os.path.join(msm_file_dir, "{}.csv.gz".format(msm))

        # MSMファイル読み込み
        logging.info('MSMファイル読み込み: {}'.format(msm_path))
        df_msm = pd.read_csv(msm_path, index_col='date',
                             parse_dates=True).sort_index()

        # 負の日射量が存在した際に日射量を0とする
        df_msm.loc[df_msm["DSWRF_msm"] < 0.0,"DSWRF_msm"] = 0.0
        df_msm.loc[df_msm["DSWRF_est"] < 0.0,"DSWRF_est"] = 0.0

        df_msm_list.append(df_msm)

    return msm_list, df_msm_list


def download_msm_files(msm_list: Iterable[str], output_dir: str):
    """MSMファイルのダウンロード

    Args:
      msm_list(Iterable[str]): ダウンロードするMSM名 ex)159-338
      output_dir(str): ダウンロード先ディレクトリ名 ex) ./msm/
    """

    # ダウンロード元URL
    dl_url = 'https://s3.us-west-1.wasabisys.com/arcclimate/msm_2011_2020/'

    for msm in msm_list:
        src_url = '{}{}.csv.gz'.format(dl_url, msm)
        save_path = os.path.join(output_dir, '{}.csv.gz'.format(msm))

        logging.info('MSMダウンロード {} => {}'.format(src_url, save_path))
        request.urlretrieve(src_url, save_path)


def get_missing_msm(msm_list: Iterable[str], msm_file_dir: str) -> List[str]:
    """存在しないMSMファイルの一覧を取得

    Args:
      msm_list(Iterable[str]): MSM名一覧
      msm_file_dir(str): MSMファイルの格納ディレクトリ

    Returns:
      不足しているMSM名一覧
    """
    def not_exists(msm): return os.path.isfile(
        os.path.join(msm_file_dir, '{}.csv.gz'.format(msm))) == False
    return list(filter(not_exists, msm_list))


def get_msm_requirements(lat: float, lon: float) -> Tuple[str, str, str, str]:
    """必要なMSMファイル名の一覧を取得

    Args:
      lat(float): 推計対象地点の緯度（10進法）
      lon(float): 推計対象地点の経度（10進法）

    Returns:
      Tuple[str, str, str, str]: 隣接する4地点のMSMファイル名のタプル
    """
    MSM_S, MSM_N, MSM_W, MSM_E = get_MSM(lat, lon)

    # 周囲4地点のメッシュ地点番号
    MSM_SW = "{}-{}".format(MSM_S, MSM_W)
    MSM_SE = "{}-{}".format(MSM_S, MSM_E)
    MSM_NW = "{}-{}".format(MSM_N, MSM_W)
    MSM_NE = "{}-{}".format(MSM_N, MSM_E)

    return MSM_SW, MSM_SE, MSM_NW, MSM_NE


def get_msm_elevations(
    lat: float, lon: float, msm_elevation_master: pd.DataFrame
) -> Tuple[float, float, float, float]:
    """計算に必要なMSMを算出して、MSM位置の標高を探してタプルで返す

    Args:
      lat: 推計対象地点の緯度（10進法）
      lon: 推計対象地点の経度（10進法）
      msm_elevation_master: MSM地点の標高データ [m]

    Returns:
      Tuple[float, float, float, float]: 4地点の標高をタプルで返します(SW, SE, NW, NE)
    """

    MSM_S, MSM_N, MSM_W, MSM_E = get_MSM(lat, lon)

    ele_SW = msm_elevation_master.loc[MSM_S, MSM_W]  # SW
    ele_SE = msm_elevation_master.loc[MSM_S, MSM_E]  # SE
    ele_NW = msm_elevation_master.loc[MSM_N, MSM_W]  # NW
    ele_NE = msm_elevation_master.loc[MSM_N, MSM_E]  # NE

    return ele_SW, ele_SE, ele_NW, ele_NE
