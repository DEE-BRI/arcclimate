import os
import sys
import math
import numpy as np
import pytest

sys.path.insert(0, os.path.realpath(
    os.path.join(os.path.basename(__file__), '..', 'src')))

from arcclimate.mixing_ratio import get_VH, get_aT, get_eSAT, get_mixing_ratio, get_corrected_mixing_ratio

def test_get_corrected_MR():
    # 温度 [C]
    TMP = 20.0

    # 気圧 [hPa]
    PRES = 1013.25

    # 重量絶対湿度 [g/kg(DA)]
    MR_sat = get_mixing_ratio(PRES, TMP)
    assert 1437.53 == pytest.approx(MR_sat, 0.01)

    # 重量絶対湿度の標高補正1
    # 1437.53 < 8300 なので、 補正結果は 1437.53
    MR_corr = get_corrected_mixing_ratio(
        MR=np.array([8300.0]),
        TMP=np.array([TMP]),
        PRES=np.array([PRES])
    )
    assert 1437.53 == pytest.approx(MR_corr, 0.01)

    # 重量絶対湿度の標高補正2
    # 1437 .53 > 300.0 なので、 補正結果は 300.0
    MR_corr = get_corrected_mixing_ratio(
        MR=np.array([300.0]),
        TMP=np.array([TMP]),
        PRES=np.array([PRES])
    )
    assert 300.0 == pytest.approx(MR_corr, 0.01)


def test_get_mixing_ratio():

    # 気圧 [hPa]
    PRES = 1013.25

    # 絶対温度 [K]
    T = 293.15

    # 乾燥空気の気体定数 [J/kgK]
    Rd = 287

    # 飽和水蒸気圧 [hPa]
    eSAT = get_eSAT(T)

    # 飽和水蒸気量 [g/m3]
    aT = get_aT(eSAT, T)

    # 重量絶対湿度 [g/kg(DA)]
    MR = aT / (PRES / (Rd * T))

    assert MR == pytest.approx(get_mixing_ratio(PRES, 20.0), 0.00001)


def test_get_eSAT():
    # 絶対温度 [K]
    T = 293.15

    ln_P = -5800.2206 / T \
        + 1.3914993\
        - 0.048640239 * T\
        + 0.000041764768 * T ** 2\
        - 0.000000014452093 * T ** 3\
        + 6.5459673 * math.log(T)

    # 飽和水蒸気圧 [Pa]
    P = math.exp(ln_P)

    assert P / 100 == pytest.approx(get_eSAT(np.array([T]))[0], 0.00001)


def test_get_aT():
    # 飽和水蒸気圧 [hPa]
    eSAT = 40.1

    # 絶対温度 [K]
    T = 293.15

    # 飽和水蒸気量 [g/m3]
    aT = 217 * 40.1 / 293.15

    assert aT == get_aT(eSAT, T)


def test_get_VH():
    # 飽和水蒸気量 [g/m3]
    aT = 100

    # 相対湿度 [%]
    RH = 60

    # 容積絶対湿度 [g/m3]
    VH = 100 * 0.6

    assert VH == get_VH(aT, RH)
