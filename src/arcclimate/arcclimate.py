# coding=utf-8
import argparse
import datetime as dt
import logging
import numpy as np
import io
import os
import sys
import pandas as pd
from typing import Any, Dict, Tuple, Optional

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from elevation import get_latlon_elevation
from weight import get_msm_weights
from MSM import load_msm_files, get_msm_elevations
from mixing_ratio import get_corrected_mixing_ratio
from wind import get_wind16
from temperature import get_corrected_TMP
from pressure import get_corrected_PRES


def interpolate(
    lat: float,
    lon: float,
    start_year: int,
    end_year: int,
    msm_elevation_master: pd.DataFrame,
    mesh_elevation_master: pd.DataFrame,
    msms: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
    mode_elevation='api',
    mode="normal",
    use_est=True,
    vector_wind=False
) -> pd.DataFrame:
    """対象地点の周囲のMSMデータを利用して空間補間計算を行う

    Args:
      lat(float): 推計対象地点の緯度（10進法）
      lon(float): 推計対象地点の経度（10進法）
      start_year(int): 出力する気象データの開始年（標準年データの検討期間も兼ねる）
      end_year(int): 出力する気象データの終了年（標準年データの検討期間も兼ねる）
      msm_elevation_master(pd.DataFrame):  MSM地点の標高データ
      mesh_elevation_master(pd.DataFrame): 3次メッシュの標高データ
      msms(Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]): 4地点のMSMデータ
      mode_elevation(str, Optional): 'mesh':標高補正に3次メッシュ（1㎞メッシュ）の平均標高データを使用する
                           'api':国土地理院のAPIを使用する
      mode(str, Optional): "normal"=補正のみ
                           "EA"=拡張アメダス方式に準じた標準年データを作成する (Default value = 'api')
      use_est(bool, Optional): 標準年データの検討に日射量の推計値を使用する（使用しない場合2018年以降のデータのみで作成） (Default value = True)
      vector_wind(bool, Optional): u,v軸のベクトル風データを出力する (Default value = False)

    Returns:
      pd.DataFrame: MSMデータフレーム
    """

    # 周囲4地点のMSMデータフレームから標高補正したMSMデータフレームを作成
    msm = _get_interpolated_msm(
        lat=lat,
        lon=lon,
        msms=msms,
        msm_elevation_master=msm_elevation_master,
        mesh_elevation_master=mesh_elevation_master,
        mode_elevation=mode_elevation
    )

    # ベクトル風速から16方位の風向風速を計算
    _convert_wind16(msm)

    # 大気放射量の単位をMJ/m2に換算
    _convert_Ld_w_to_mj(msm)

    if mode == "normal":
        # 保存用に年月日をフィルタ
        start_index = dt.datetime(start_year, 1, 1)
        end_index = dt.datetime(end_year+1, 1, 1)
        df_save = msm[(start_index <= msm.index)
                      & (msm.index < end_index)]

    elif mode == "EA":
        # 標準年の計算
        from EA import calc_EA
        df_save, select_year = calc_EA(msm,
                                       start_year=start_year,
                                       end_year=end_year,
                                       use_est=use_est)

        # ベクトル風速から16方位の風向風速を計算
        _convert_wind16(df_save)

        # 大気放射量の単位をMJ/m2に換算
        _convert_Ld_w_to_mj(df_save)

    else:
        raise ValueError(mode)

    # u,v軸のベクトル風データのフィルタ
    if not vector_wind:
        df_save.drop(['VGRD', 'UGRD'], axis=1, inplace=True)

    return df_save


def _get_interpolated_msm(
    lat: float,
    lon: float,
    msms: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
    msm_elevation_master: pd.DataFrame,
    mesh_elevation_master: pd.DataFrame,
    mode_elevation: Optional[str] = 'api'
) -> pd.DataFrame:
    """標高補正

    Args:
      lat(float): 推計対象地点の緯度（10進法）
      lon(float): 推計対象地点の経度（10進法）
      msms(Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]): 4地点のMSMデータフレーム
      msm_elevation_master(pd.DataFrame): MSM地点の標高データマスタ
      mesh_elevation_master(pd.DataFrame): 3次メッシュの標高データ
      mode_elevation(str, Optional): 'mesh':標高補正に3次メッシュ（1㎞メッシュ）の平均標高データを使用する
                                     'api':国土地理院のAPIを使用する (Default)

    Returns:
      pd.DataFrame: 標高補正されたMSMデータフレーム
    """
    logging.info('補間計算を実行します')

    # 緯度経度から標高を取得
    ele_target = get_latlon_elevation(
        lat=lat,
        lon=lon,
        mode_elevation=mode_elevation,
        mesh_elevation_master=mesh_elevation_master
    )

    # 補間計算 リストはいずれもSW南西,SE南東,NW北西,NE北東の順
    # 入力した緯度経度から周囲のMSMまでの距離を算出して、距離の重みづけ係数をリストで返す
    weights = get_msm_weights(lat, lon)

    # 計算に必要なMSMを算出して、MSM位置の標高を探してリストで返す
    elevations = get_msm_elevations(lat, lon, msm_elevation_master)

    # 周囲のMSMの気象データを読み込んで標高補正後に按分する
    msm_target = _get_prportional_divided_msm_df(
        msms=msms,
        weights=weights,
        elevations=elevations,
        ele_target=ele_target
    )

    return msm_target


def _get_prportional_divided_msm_df(
    msms: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
    weights: Tuple[float, float, float, float],
    elevations: Tuple[float, float, float, float],
    ele_target: float
) -> pd.DataFrame:
    """周囲のMSMの気象データを読み込んで標高補正し加算

    Args:
      msms(Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]): 4地点のMSMデータフレーム(タプル)
      weights(Tuple[float, float, float, float]): 4地点の重み(タプル)
      elevations(Tuple[float, float, float, float]): 4地点のMSM平均標高[m](タプル)
      ele_target: 目標地点の標高 [m]

    Returns:
      pd.DataFrame: 標高補正により重みづけ補正されたMSMデータフレーム
    """

    # 標高補正 (SW,SE,NW,NE)
    msm_SW = _get_corrected_msm(msms[0], elevations[0], ele_target)
    msm_SE = _get_corrected_msm(msms[1], elevations[1], ele_target)
    msm_NW = _get_corrected_msm(msms[2], elevations[2], ele_target)
    msm_NE = _get_corrected_msm(msms[3], elevations[3], ele_target)

    # 重みづけによる按分
    msm_target = \
        weights[0] * msm_SW \
        + weights[1] * msm_SE \
        + weights[2] * msm_NW \
        + weights[3] * msm_NE

    return msm_target


def _get_corrected_msm(msm: pd.DataFrame, elevation: float, ele_target: float):
    """MSMデータフレーム内の気温、気圧、重量絶対湿度を標高補正

    Args:
      df_msm(pd.DataFrame): MSMデータフレーム
      ele(float): 平均標高 [m]
      elevation(float): 目標地点の標高 [m]

    Returns:
      pd.DataFrame: 補正後のMSMデータフレーム
    """
    TMP = msm['TMP'].values
    PRES = msm['PRES'].values
    MR = msm['MR'].values

    # 標高差
    ele_gap = ele_target - elevation

    # 気温補正
    TMP_corr = get_corrected_TMP(TMP, ele_gap)

    # 気圧補正
    PRES_corr = get_corrected_PRES(PRES, ele_gap, TMP_corr)

    # 重量絶対湿度補正
    MR_corr = get_corrected_mixing_ratio(
        MR=MR,
        TMP=TMP_corr,
        PRES=PRES_corr
    )

    # 補正値をデータフレームに戻す
    msm = msm.copy()
    msm['TMP'] = TMP_corr
    msm['PRES'] = PRES_corr
    msm['MR'] = MR_corr

    # なぜ 気圧消すのか？
    msm.drop(['PRES'], axis=1, inplace=True)

    return msm


def _convert_wind16(msm: pd.DataFrame):
    """ベクトル風速から16方位の風向風速を計算

    Args:
      df(pd.DataFrame): MSMデータフレーム
    """

    # 風向風速の計算
    w_spd16, w_dir16 = get_wind16(msm['UGRD'], msm['VGRD'])

    # 風速(16方位)
    msm['w_spd'] = w_spd16

    # 風向(16方位)
    msm['w_dir'] = w_dir16


def _convert_Ld_w_to_mj(msm: pd.DataFrame):
    """大気放射量の単位をW/m2からMJ/m2に換算

    Args:
      df(pd.DataFrame): MSMデータフレーム
    """
    msm['Ld'] = msm['Ld'] * (3.6/1000)


def init(
    lat: float, lon: float, path_MSM_ele: str, path_mesh_ele: str, msm_file_dir: str
) -> Dict[str, Any]:
    """初期化処理

    Args:
      lat(float): 推計対象地点の緯度（10進法）
      lon(float): 推計対象地点の経度（10進法）
      path_MSM_ele(str): MSM地点の標高データのファイルパス
      path_mesh_ele(str): 3次メッシュの標高データのファイルパス
      msm_file_dir(str): MSMファイルの格納ディレクトリ

    Returns:
      以下の要素を含む辞書
      - msm_list(list[str]): 読み込んだMSMファイルの一覧
      - df_msm_ele(pd.DataFrame): MSM地点の標高データ
      - df_mesh_ele(pd.DataFrame): 3次メッシュの標高データ
      - df_msm_list(list[pd.DataFrame]): 読み込んだデータフレームのリスト
    """

    # ログレベル指定
    logging.basicConfig(level=logging.ERROR)

    # ロガーの作成
    logger = logging.getLogger(__name__)

    # MSM地点の標高データの読込
    logging.info('MSM地点の標高データ読込: {}'.format(path_MSM_ele))
    df_msm_ele = pd.read_csv(path_MSM_ele, header=None)

    # 3次メッシュの標高データの読込
    logging.info('3次メッシュの標高データ読込: {}'.format(path_mesh_ele))
    df_mesh_ele = pd.read_csv(path_mesh_ele).set_index('meshcode')

    # MSMファイルの読込
    MSM_list, df_msm_list = load_msm_files(lat, lon, msm_file_dir)

    return {
        'msm_list': MSM_list,
        'df_msm_ele': df_msm_ele,
        'df_mesh_ele': df_mesh_ele,
        'df_msm_list': df_msm_list
    }


def to_epw(df: pd.DataFrame, out, lat, lon):

    #TODO: 動作検証
    #ref: https://bigladdersoftware.com/epx/docs/8-3/auxiliary-programs/energyplus-weather-file-epw-data-dictionary.html#energyplus-weather-file-epw-data-dictionary

    # LOCATION
    # 国名,緯度,経度,タイムゾーンのみ出力
    out.write("LOCATION,-,-,JPN,-,-,{},{},9.0,\n".format(lat, lon))

    # DESIGN CONDITION
    # 設計条件なし
    out.write("DESIGN CONDITIONS,0\n")

    # TYPICAL/EXTREME PERIODS
    # 期間指定なし
    out.write("TYPICAL/EXTREME PERIODS,0\n")

    # GROUND TEMPERATURES
    # 地中温度無し
    out.write("GROUND TEMPERATURES,0\n")

    # HOLIDAYS/DAYLIGHT SAVINGS
    # 休日/サマータイム---TODO:要確認
    out.write("HOLIDAYS/DAYLIGHT SAVINGS\n")

    # COMMENT 1
    out.write("COMMENTS 1\n")

    # COMMENT 2
    out.write("COMMENTS 2\n")

    # DATA HEADER
    out.write("DATA PERIODS,0\n")

    for index, row in df.iterrows():
        # N1: 年
        # N2: 月
        # N3: 日
        # N4: 時
        # N5: 分 = 0
        # N6: Dry Bulb Temperature
        # N7-N19: missing
        # N20: w_dir
        # N21: w_spd
        # N22-N32: missing
        # N33: APCP01
        # N34: missing
        out.write("{},{},{},{},0,-,{},,,,,,,,,,,,,,{},{},,,,,,,,,,,,{},\n".format(index.year, index.month, index.day, index.hour, row['TMP'], row['w_dir'], row['w_spd'], row['APCP01']))


def main():
    # コマンドライン引数の処理
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "lat",
        type=float,
        help="推計対象地点の緯度（10進法）"
    )
    parser.add_argument(
        "lon",
        type=float,
        help="推計対象地点の経度（10進法）"
    )
    parser.add_argument(
        "-o",
        dest="out",
        default=None,
        help="保存ファイルパス"
    )
    parser.add_argument(
        "--start_year",
        type=int,
        default=2011,
        help="出力する気象データの開始年（標準年データの検討期間も兼ねる）"
    )
    parser.add_argument(
        "--end_year",
        type=int,
        default=2020,
        help="出力する気象データの終了年（標準年データの検討期間も兼ねる）"
    )
    parser.add_argument(
        "--mode",
        choices=["normal", "EA"],
        default="normal",
        help="計算モードの指定 標準=normal(デフォルト), 標準年=EA"
    )
    parser.add_argument(
        "-f",
        choices=["CSV", "EPW"],
        default="CSV",
        help="出力形式 CSV or EPW"
    )
    parser.add_argument(
        "--mode_elevation",
        choices=["mesh", "api"],
        default="api",
        help="標高判定方法 API=api(デフォルト), メッシュデータ=mesh"
    )
    parser.add_argument(
        "--disable_est",
        action='store_true',
        help="標準年データの検討に日射量の推計値を使用しない（使用しない場合2018年以降のデータのみで作成）"
    )
    parser.add_argument(
        "--msm_file_dir",
        default='.{0}.msm_cache{0}'.format(os.sep),
        help="MSMファイルの格納ディレクトリ"
    )
    parser.add_argument(
        "--log",
        choices=['DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL'],
        default='ERROR',
        help="ログレベルの設定"
    )
    args = parser.parse_args()

    # ログレベル設定
    log_level = getattr(logging, args.log.upper(), None)
    logging.basicConfig(level=log_level)

    # MSMフォルダの作成
    os.makedirs(args.msm_file_dir, exist_ok=True)

    # 初期化
    conf = init(
        lat=args.lat,
        lon=args.lon,
        path_MSM_ele=os.path.abspath(os.path.join(
            os.path.dirname(__file__), "data", "MSM_elevation.csv")),
        path_mesh_ele=os.path.abspath(os.path.join(
            os.path.dirname(__file__), "data", "mesh_3d_elevation.csv")),
        msm_file_dir=args.msm_file_dir
    )

    # EA方式かつ日射量の推計値を使用しない場合に開始年が2018年以上となっているか確認
    if args.mode == "EA":
        if args.disable_est:
            if args.start_year < 2018:
                logging.info('--disable_estを設定した場合は開始年を2018年以降にする必要があります')
                print("""Error: If "disable_est" is set, the start year must be 2018 or later"""
                      , file=sys.stderr)
                sys.exit(1)
                
            else:
                args.use_est = False
        
        else:
            args.use_est = True

    # 補間処理
    df_save = interpolate(
        lat=args.lat,
        lon=args.lon,
        start_year=args.start_year,
        end_year=args.end_year,
        msm_elevation_master=conf['df_msm_ele'],
        mesh_elevation_master=conf['df_mesh_ele'],
        msms=conf['df_msm_list'],
        mode=args.mode,
        mode_elevation=args.mode_elevation,
        use_est=not args.disable_est,
        vector_wind=True
    )

    # 保存
    out = io.StringIO()
    if args.f == "CSV":
        df_save.to_csv(out)
    elif args.f == "EPW":
        # TODO: 標高を引き渡す
        to_epw(df_save, out, args.lat, args.lon)

    if args.out is None:
        print(out.getvalue())
    else:
        logging.info('CSV保存: {}'.format(args.out))
        with open(args.out, mode='w') as f:
            print(out.getvalue(), file=f)

    logging.info('計算が終了しました')


if __name__ == '__main__':
    main()
