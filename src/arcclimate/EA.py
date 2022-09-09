"""
標準年の計算モジュール
"""

# 拡張アメダス(MetDS(株)気象データシステム社)の
# 標準年データの2010年版の作成方法を参考とした
# ※2020年版は作成方法が変更されている
#     
# 参考文献
# 二宮 秀與 他
# 外皮・躯体と設備・機器の総合エネルギーシミュレーションツール
# 「BEST」の開発(その172)30 年拡張アメダス気象データ
# 空気調和・衛生工学会大会 学術講演論文集 2016.5 (0), 13-16, 2016

import calendar
import datetime as dt
import numpy as np
import pandas as pd
from typing import List, Tuple, Generator
from datetime import timedelta


def calc_EA(df: pd.DataFrame, start_year: int, end_year: int, use_est: bool) \
        -> Tuple[pd.DataFrame, List[int]]:
    """標準年の計算

    Args:
      df(pd.DataFrame): MSMデータフレーム
      start_year(int): 標準年データの検討開始年
      end_year(int): 標準年データの検討終了年
      use_est(bool): 標準年データの検討に日射量の推計値を使用する(使用しない場合2018年以降のデータのみで作成)

    Returns:
      Tuple[pd.DataFrame, List[int]]: 標準年MSMデータフレームおよび選択された年のリストのタプル
    """
    if use_est:
        # * 標準年データの検討に日射量の推計値を使用する
        #   -> `DSWRF_msm`列を削除し、`DSWRF_est`列を`DSWRF`列へ変更(推計値データを採用)
        df.drop("DSWRF_msm", axis=1, inplace=True)
        df.rename(columns={"DSWRF_est": "DSWRF"}, inplace=True)

        start_index = dt.datetime(int(start_year), 1, 1)
        end_index = dt.datetime(int(end_year), 12, 31)
        df_targ = df[(start_index <= df.index) &
                     (df.index <= end_index)].copy()

        # TODO: drop, rename処理はcopyの後の方がよさそう

    else:
        # * 2018年以降のデータのみで作成
        # * `DSWRF_est`列を削除し、`DSWRF_msm`列を`DSWRF`列へ変更(MSMデータを採用)
        df.drop("DSWRF_est", axis=1, inplace=True)
        df.rename(columns={"DSWRF_msm": "DSWRF"}, inplace=True)

        start_year = np.where(start_year <= 2018, 2018, start_year)

        start_index = dt.datetime(start_year, 1, 1)
        end_index = dt.datetime(int(end_year), 12, 31)
        df_targ = df[(start_index <= df.index) &
                     (df.index <= end_index)].copy()

        # TODO: drop, rename処理はcopyの後の方がよさそう
        # TODO: copy処理は if/elseの両方で同じに見える

    # ベクトル風速`w_spd`, 16方位の風向風速`w_dir`の列を削除
    df.drop(["w_spd", "w_dir"], axis=1, inplace=True)

    # groupのターゲットを追加
    # カラム [TMP, MR, DSWF, Ld, VGRD, UGRD, PRES, APCP01, MR_sat, w_spd, w_dir]
    #   ↓
    # カラム [TMP, MR, DSWF, Ld, VGRD, UGRD, PRES, APCP01, MR_sat, w_spd, w_dir, y, m, d]
    #                            TMP        MR  DSWRF        Ld      VGRD      UGRD          PRES    APCP01    MR_sat     w_spd  w_dir     y   m   d
    # date
    # 2011-01-01 00:00:00  -6.776232  1.969510    0.0  0.772116 -3.051297  0.837700  98383.696285  0.021976  2.344268  3.139605  337.5  2011  01  01
    # 2011-01-01 01:00:00  -5.862659  2.085005    0.0  0.793957 -3.203325  0.251010  98437.952674  0.001623  2.511178  3.203325  360.0  2011  01  01
    df_targ["y"] = df_targ.index.strftime('%Y')
    df_targ["m"] = df_targ.index.strftime('%m')
    df_targ["d"] = df_targ.index.strftime('%d')

    df_temp = grouping(df_targ)
    df_temp = get_mean_std(df_temp)

    # df_temp
    #         y   m   TMP  DSWRF    MR  APCP01  w_spd
    # 0    2011  01  True   True  True    True   True
    # 1    2012  01  True   True  True    True   True
    # 2    2013  01  True   True  True    True   True

    # FS計算
    #         y   m    TMP  DSWRF     MR  APCP01  w_spd    TMP_FS
    # 0    2011  01  False  False  False   False  False  0.076795
    # 1    2012  01  False   True  False    True  False  0.149116
    FS = get_FS(df_targ, df_temp)

    # 月別に代表的な年を取得
    rep_years = _get_representative_years(df_temp, FS)

    # 月別に代表的な年から接合した1年間のデータを作成
    df_EA = _make_EA(df, rep_years)

    # 接合部の円滑化
    for target, before_year, after_year in _get_smoothing_months(rep_years):
        _smooth_month_gaps(target, before_year, after_year, df, df_EA)

    if use_est:
        df_EA.rename(columns={"DSWRF": "DSWRF_est"}, inplace=True)

    else:
        df_EA.rename(columns={"DSWRF": "DSWRF_msm"}, inplace=True)

    return df_EA, rep_years


def grouping(msm: pd.DataFrame) -> pd.DataFrame:
    """月偏差値,月平均,年月平均のMSMデータフレームの作成

    Args:
      df: MSMデータフレーム

    Returns:
      pd.DataFrame: 月偏差値,月平均,年月平均のMSMデータフレーム
    """
    # 月ごとの標準偏差に変換したデータフレーム作成
    g_m_std = msm.groupby(["m"]).std().reset_index()

    # 月平均に変換したデータフレーム作成
    g_m_mean = msm.groupby(["m"]).mean().reset_index()

    # 年月平均に変換したデータフレーム作成
    g_ym_mean = msm.groupby(["y", "m"]).mean().reset_index()

    # データフレームの合成
    # カラム [y, m,
    #         TMP_mean_ym, MR_mean_ym, DSWF_mean_ym, Ld_mean_ym,
    #         VGRD_mean_ym, UGRD_mean_ym,
    #         PRES_mean_ym, APCP01_mean_ym,
    #         MR_sat_mean_ym, w_spd_mean_ym, w_dir_mean_ym,
    #         TMP_mean_m, MR_mean_m, DSWF_mean_m, Ld_mean_m,
    #         VGRD_mean_m, UGRD_mean_m,
    #         PRES_mean_m, APCP01_mean_m,
    #         MR_sat_mean_m, w_spd_mean_m, w_dir_mean_m,
    #         TMP_std_m, MR_std_m, DSWF_std_m, Ld_std_m,
    #         VGRD_std_m, UGRD_std_m,
    #         PRES_std_m, APCP01_std_m,
    #         MR_sat_std_m, w_spd_std_m, w_dir_std_m]
    df_temp = pd.merge(g_ym_mean, g_m_mean, on='m', suffixes=['', '_mean_m'])
    df_temp = pd.merge(df_temp, g_m_std, on='m',
                       suffixes=['_mean_ym', '_std_m'])

    return df_temp


def get_mean_std(df_temp: pd.DataFrame) -> pd.DataFrame:
    """

    Args:
      df_temp(pd.DataFrame): 

    Returns:
      pd.DataFrame: 
    """
    set_1 = ["TMP", "DSWRF", "MR"]
    for weather in set_1:
        # 月平均と年月平均の差分(絶対値)計算 => "XXX_mean"
        df_temp[weather + "_dev"] = abs(df_temp[weather + "_mean_m"] - df_temp[weather + "_mean_ym"])

        # 月平均と年月平均の差分 "XXX_mean" が月標準偏差σ以下か？ => "XXX"
        # ()
        df_temp[weather] = (df_temp[weather + "_dev"] <= df_temp[weather + "_std_m"])

    set_2 = ["APCP01", "w_spd"]
    for weather in set_2:
        # 月平均と年月平均の差分(絶対値)計算 => "XXX_mean"
        df_temp[weather + "_dev"] = abs(df_temp[weather + "_mean_m"] - df_temp[weather + "_mean_ym"])

        # 月平均と年月平均の差分 "XXX_mean" が月標準偏差σ×1.5以下か？ => "XXX"
        df_temp[weather] = (df_temp[weather + "_dev"] <= (1.5 * df_temp[weather + "_std_m"]))

    # 各項目が想定信頼区間に入っているかを真偽値で格納したデータフレーム
    #         y   m   TMP_dev   TMP   DSWRF  MR   APCP01  w_spd
    # 0    2011  01     0.01   True   True  True   True   True
    # 1    2012  01     0.01   True   True  True   True   True
    # 2    2013  01     0.01   True   True  True   True   True
    return df_temp.loc[:, ["y", "m", "TMP_dev", "TMP", "DSWRF", "MR", "APCP01", "w_spd"]]


def get_FS(df, df_temp):
    """FS(Finkelstein Schafer statistics)計算

    Args:
      df:
      df_temp: 

    Returns:

    """


    # FSの容器の作成
    #         y   m
    # 0    2011  01
    # 1    2012  01
    FS = df_temp.loc[:, ["y", "m"]]

    weather_list = ["TMP", "DSWRF", "MR", "APCP01", "w_spd"]

    # 日平均計算
    g_ymd_mean = df.groupby(["y", "m", "d"], as_index=False).mean()

    # 月ごとのグループを作成
    g_ymd_mean_m = g_ymd_mean.groupby(["m"], as_index=False)

    # 月ごとの累積度数分布(CDF)の計算
    for name, group in g_ymd_mean_m:
        for weather in weather_list:
            g = group.sort_values(weather).reset_index()
            N = len(g)
            g.loc[:, "cdf_ALL"] = [(i + 1) / N for i in range(N)]
            g = g.sort_values('index').set_index('index')
            g_ymd_mean.loc[list(g.index), weather +
                           "_cdf_ALL"] = g["cdf_ALL"].values

    # 年月ごとの累積度数分布(CDF)の計算
    g_ymd_mean_ym = g_ymd_mean.groupby(["y", "m"], as_index=False)
    for name, group in g_ymd_mean_ym:
        for weather in weather_list:
            g = group.sort_values(weather).reset_index()
            N = len(g)
            g.loc[:, "cdf_year"] = [(i + 1) / N for i in range(N)]
            g = g.sort_values('index').set_index('index')
            g_ymd_mean.loc[list(g.index), weather +
                           "_cdf_year"] = g["cdf_year"].values

    for weather in weather_list:
        g_ymd_mean.loc[:, weather + "_FS"] = np.abs(g_ymd_mean[weather + "_cdf_ALL"].values - g_ymd_mean[weather + "_cdf_year"].values)
        df_FS = g_ymd_mean.loc[:, ["y", "m", weather + "_FS"]].groupby(["y", "m"], as_index=False).mean()
        FS = pd.merge(FS, df_FS, on=['y', 'm'])

    FS_std = FS.groupby(["m"]).std().reset_index()

    FS = pd.merge(FS, FS_std, on='m', suffixes=['', '_std'])

    set_1 = ["TMP", "DSWRF", "MR"]
    for weather in set_1:
        FS[weather] = (FS[weather + "_FS"] <= FS[weather + "_FS_std"])

    set_2 = ["APCP01", "w_spd"]
    for weather in set_2:
        FS[weather] = (FS[weather + "_FS"] <= (1.5 * FS[weather + "_FS_std"]))

    return FS.loc[:, ["y", "m", "TMP", "DSWRF", "MR", "APCP01", "w_spd", 'TMP_FS']]


# **** 代表年の決定と接合処理 ****


def _get_representative_years(df_temp: pd.DataFrame, FS: pd.DataFrame) -> List[int]:
    """

    Args:
      df_temp(pd.DataFrame): 
      FS: 

    Returns:
      List[int]: 月別の代表的な年
    """
    df_threshold = pd.merge(df_temp, FS, on=['y', 'm'], suffixes=['_mean', '_fs']).sort_values(["m", "y"]).reset_index(
        drop=True)

    # 選定は、気温(偏差)=>水平面全天日射量(偏差)=>絶対湿度(偏差)=>降水量(偏差)=>風速(偏差)=>
    # 気温(FS)=>水平面全天日射量(FS)=>絶対湿度(FS)=>降水量(FS)=>風速(FS)の順に判定を行う
    # 最終的に複数が候補となった場合は気温(偏差)が最も0に近い年を選定する。

    select_list = ["TMP_mean",
                   "DSWRF_mean",
                   "MR_mean",
                   "APCP01_mean",
                   "w_spd_mean",
                   "TMP_fs",
                   "DSWRF_fs",
                   "MR_fs",
                   "APCP01_fs",
                   "w_spd_fs"]

    select_year = []

    g_m = df_threshold.groupby(["m"])

    for name, group in g_m:
        center_y = group['y'].astype(int).mean()

        # 判定指標でループ(候補が単一の年になるまで繰り返す)
        for select in select_list:
            group_temp = group[group[select] == True]

            if group_temp['y'].count() == 0:
                # group_temp(selectがTrueの年)が0個
                # =>group(前selectがTrueの年)の中から気温(偏差)が最も小さい年を選定

                # TMP_devが最小の年を抜粋
                group_temp = group[group['TMP_dev']
                            == group['TMP_dev'].min()].copy()

                if group_temp['y'].count() != 1:
                    # TMP_devの最小が複数残った場合 => 次の判定指標へ

                    group = group_temp

                else:
                    # TMP_devの最小が1つの場合 => 代表年として選定

                    select_year += list(group_temp['y'].values)
                    break

            elif group_temp['y'].count() == 1:
                # group_temp(selectがTrueの年)が1個 => 代表年として選定

                select_year += list(group_temp['y'].values)
                break

            elif select == "w_spd_fs":
                # 判定指標がw_spd_fs(最後の判定指標)の時 => 気温(偏差)で判定

                # TMP_devが最小の年を抜粋
                group_temp = group[group['TMP_dev']
                            == group['TMP_dev'].min()].copy()

                if group_temp['y'].count() != 1:
                    # TMP_devの最小が複数残った場合 => 対象期間の中心(平均)に近い年を選定

                    group_temp['y_abs'] = abs(group_temp.loc[:,'y'].astype(int)-center_y)                    
                    group_temp = group_temp[group_temp['y_abs'] == group_temp['y_abs'].min()]

                    if group_temp['y'].count() != 1:
                        # 対象期間の中心(平均)に近い年が複数残った場合 => 若い年を選定

                        select_year += list(group_temp['y'].min())

                    else:
                        # 対象期間の中心(平均)に近い年が1つ => 代表年として選定

                        select_year += list(group_temp['y'].values)
                        break

                else:
                    # TMP_devの最小が1つの場合 => 代表年として選定

                    select_year += list(group_temp['y'][group_temp['TMP_dev']
                                        == group_temp['TMP_dev'].min()].values)
                    break

            else:
                group = group_temp

    return select_year


def _make_EA(df: pd.DataFrame, rep_years: List[int]) -> pd.DataFrame:
    """標準年の作成
    月別に代表的な年から接合した1年間のデータを作成します。

    Args:
      df(pd.DataFrame):
      rep_years(List[int]): 月別の代表的な年

    Returns:
      pd.DataFrame: 標準年のMSMデータフレーム
    """
    df_EA = pd.DataFrame()

    # 月日数
    mdays = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    # 月別に代表的な年のデータを抜き出す
    for month, year in enumerate(rep_years, 1):

        # 当該代表年月の開始日とその次月開始日
        start_date = dt.datetime(int(year), month, 1)
        end_date = start_date + timedelta(days=mdays[month-1])

        # 抜き出した代表データ
        df_temp = df[(start_date <= df.index) & (
            df.index < end_date)].sort_index()

        # 接合
        df_EA = pd.concat([df_EA, df_temp])

    df_EA.loc[:, "date"] = pd.date_range(
        start="1970-01-01", end="1970-12-31 23:00", freq="H")

    return df_EA.set_index("date")


# **** 円滑化処理 ****


def _get_smoothing_months(
    rep_years: List[int]
) -> Generator[int, int, int]:
    """円滑化が必要な月の取得

    Args:
      rep_years(List[int]): 月別の代表的な年

    Returns:
      Generator[int, int, int]: 円滑化が必要な月,前月の代表年,対象月の代表年のタプル
    """
    for i in range(11):

        # 1月から計算
        target = i + 1

        # 前月の代表年
        before_year = int(rep_years[i - 1])

        # 対象月の代表年
        after_year = int(rep_years[i])

        # 前月と対象月では対象月が異なる または 前月が2月かつその代表年が閏年の場合
        if before_year != after_year or (target == 3 and calendar.isleap(int(before_year))):
            yield target, before_year, after_year


def _smooth_month_gaps(after_month: int, before_year: int, after_year: int, df_temp: pd.DataFrame, df_EA: pd.DataFrame):
    """月別に代表的な年からの接合部を滑らかに加工する

    Args:
      after_month(int): 対象月
      after_year(int): 対象月の代表年
      before_year(int): 前月の代表年
      df_temp(pd.DataFrame): 
      df_EA(pd.DataFrame): 
    """
    # [1, 0.92, 0.83, ...., 0.0], [0.0, 0.08, 0.17, 0.25, ..., 1.0]
    before_coef = np.linspace(1, 0, 13, endpoint=True)
    after_coef = np.linspace(0, 1, 13, endpoint=True)

    # 対象月の1970年における対象月の1日
    center = dt.datetime(year=1970, month=int(after_month), day=1, hour=0)

    # 前月の代表年における対象月の1日
    before = dt.datetime(year=int(before_year), month=int(after_month), day=1, hour=0)

    # 対象月の代表年における対象月の1日
    after = dt.datetime(year=int(after_year), month=int(after_month), day=1, hour=0)

    # 12月と1月の結合(年をまたぐ)
    if after_month == 1:

        # 前月の代表年における12月31日18時
        before_start = dt.datetime(year=int(before_year), month=12, day=31, hour=18)

        # 前月の代表年の翌年の1月1日6時
        before_end = dt.datetime(year=int(before_year + 1), month=1, day=1, hour=6)

        # 前月の代表年の12月31日18時から翌年1月1日6時までのMSMデータフレーム
        df_before = df_temp.loc[before_start:before_end, :].copy()

        # 対象月の代表年の前年の12月31日18時
        after_start = dt.datetime(year=int(after_year - 1), month=12, day=31, hour=18)

        # 対象月の代表年の1月1日6時
        after_end = dt.datetime(year=int(after_year), month=1, day=1, hour=6)

        # 対象月の代表年の前年12月31日18時から翌年1月1日6時までのMSMデータフレーム
        df_after = df_temp.loc[after_start:after_end, :].copy()

        # 1970年12月31日18時-23時
        timestamp_12 = dt.datetime(year=1970, month=12, day=31, hour=18)
        timestamp_12 = list(pd.date_range(
            timestamp_12, periods=6, freq="H").values)

        # 1970年1月1日0時-6時
        timestamp_1 = dt.datetime(year=1970, month=1, day=1, hour=0)
        timestamp_1 = list(pd.date_range(
            timestamp_1, periods=7, freq="H").values)

        # 1970年12月31日18時-23時 および 1月1日0時-6時
        timestamp = timestamp_12 + timestamp_1

    # 2月と3月の結合(うるう年の回避)
    elif after_month == 3:

        # 結合する2つの月の若い月(前月)の代表年における2月28日18時(はじまり)
        before_start = dt.datetime(year=int(before_year), month=2, day=28, hour=18)

        # 前月の代表年における3月1日6時(おわり)
        before_end = before + dt.timedelta(hours=6)

        # 前月の代表年における2月28日18時から3月1日6時までのMSMデータフレーム
        df_before = df_temp.loc[before_start:before_end, :].copy()

        # 結合する2つの月の遅い月(対象月)の代表年における2月28日18時(はじまり)
        after_start = dt.datetime(year=int(after_year), month=2, day=28, hour=18)

        # 対象月の代表年における3月1日6時(おわり)
        after_end = after + dt.timedelta(hours=6)

        # 対象月の代表年における2月28日18時から3月1日6時までのMSMデータフレーム
        df_after = df_temp.loc[after_start:after_end, :].copy()

        # MSMデータフレームから2月29日を除外
        df_before = df_before[df_before.index.day != 29]
        df_after = df_after[df_after.index.day != 29]

        # 対象月の1970年における対象月の1日の前日18時から翌日6時まで
        timestamp = pd.date_range(
            center + dt.timedelta(hours=-6), periods=13, freq="H")

    else:
        # 前月の代表年における対象月の1日の前月末日18時
        before_start = before + dt.timedelta(hours=-6)

        # 前月の代表年における対象月の1日6時
        before_end = before + dt.timedelta(hours=6)

        # 前月の代表年における対象月の1日の前月末日18時から1日6時までのMSMデータフレーム
        df_before = df_temp.loc[before_start:before_end, :].copy()

        # 対象月の代表年における対象月の1日の前月末日18時
        after_start = after + dt.timedelta(hours=-6)

        # 対象月の代表年における対象月の1日6時
        after_end = after + dt.timedelta(hours=6)

        # 対象月の代表年における対象月の1日の前月末日18時から1日6時までのMSMデータフレーム
        df_after = df_temp.loc[after_start:after_end, :].copy()

        # 対象月の1970年における対象月の1日の前日18時から翌日6時まで
        timestamp = pd.date_range(
            center + dt.timedelta(hours=-6), periods=13, freq="H")

    # 前月の代表年における月末から翌月にかけての13時間 -> 係数を1,0.92,... と掛ける。
    # 対象月の代表年における前月末18時からの13時間 -> 係数を0,0.08,,... と掛ける。
    # 以上を合算する。
    df_new = \
        df_before.mul(before_coef, axis=0).reset_index(drop=True) + \
        df_after.mul(after_coef, axis=0).reset_index(drop=True)

    # タイムスタンプを上書き
    df_new.loc[:, "date"] = timestamp

    df_EA.loc[timestamp, :] = df_new.set_index("date")
