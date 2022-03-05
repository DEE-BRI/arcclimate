"""
重量絶対湿度の計算モジュール
"""

import numpy as np


def get_corrected_mixing_ratio(
    MR: np.ndarray, TMP: np.ndarray, PRES: np.ndarray
) -> np.ndarray:
    """重量絶対湿度の標高補正

    Args:
      MR(np.ndarray): 補正前重量絶対湿度 (Mixing Ratio) [g/kg(DA)]
      TMP(np.ndarray): 気温 [C]
      PRES(np.ndarray): 気圧 [hPa]

    Returns:
      np.ndarray: 重量絶対湿度の標高補正後のMR [g/kg(DA)]
    """
    # (最低)重量絶対湿度 [g/kg(DA)]
    MR_min = get_mixing_ratio(PRES, TMP)

    # 重量絶対湿度の補正
    MR_corr = np.maximum(MR, MR_min)

    return MR_corr


def get_mixing_ratio(PRES: np.ndarray, TMP: np.ndarray) -> np.ndarray:
    """重量絶対湿度を求める

    Args:
        PRES (np.ndarray): 気圧 [hPa]
        TMP (np.ndarray): 気温 [C]

    Returns:
        np.ndarray: 重量絶対湿度 [g/kg(DA)]
    """

    # 絶対温度 [K]
    T = TMP + 273.15

    # 飽和水蒸気圧 [hPa]
    eSAT = get_eSAT(T)

    # 飽和水蒸気量 [g/m3]
    aT = get_aT(eSAT, T)

    # 重量絶対湿度 [g/kg(DA)]
    MR = aT / ((PRES / 100) / (2.87 * T))

    return MR


def get_eSAT(T: np.ndarray) -> np.ndarray:
    """Wexler-Hylandの式 飽和水蒸気圧 eSAT

    Args:
      T(np.ndarray): 絶対温度 [K]

    Returns:
      np.ndarray: 飽和水蒸気圧 [hPa]
    """
    return np.exp(-5800.2206 / T
                  + 1.3914993 - 0.048640239 * T
                  + 0.41764768 * 10 ** (-4) * T ** 2
                  - 0.14452093 * 10 ** (-7) * T ** 3
                  + 6.5459673 * np.log(T)) / 100


def get_aT(eSAT: np.ndarray, T: np.ndarray) -> np.ndarray:
    """飽和水蒸気量 a(T) Saturated water vapor amount

    Args:
      eSAT(np.ndarray): 飽和水蒸気圧 [hPa]
      T(np.ndarray): 絶対温度 [K]

    Returns:
      np.ndarray: 飽和水蒸気量 [g/m3]
    """
    return (217 * eSAT) / T


def get_VH(aT: np.ndarray, RH: np.ndarray) -> np.ndarray:
    """容積絶対湿度 volumetric humidity

    Args:
      aT(np.ndarray): 飽和水蒸気量 [g/m3]
      RH(np.ndarray): 相対湿度 [%]

    Returns:
      np.ndarray: 容積絶対湿度 [g/m3]
    """
    return aT * (RH / 100)
