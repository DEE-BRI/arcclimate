"""
気圧に関するモジュール
"""

import numpy as np


def get_corrected_PRES(PRES: np.ndarray, ele_gap: float, TMP: np.ndarray) -> np.ndarray:
    """気圧の標高補正

    Args:
        PRES (np.ndarray): 補正前の気圧 [hPa]
        ele_gap (float): 標高差 [m]
        TMP (np.ndarray): 気温 [℃]

    Returns:
        np.ndarray: 標高補正後の気圧 [hPa]
    
    Notes:
        気温減率の平均値を0.0065℃/mとする。
    """
    return PRES * np.power(1 - ((ele_gap * 0.0065) / (TMP + 273.15)), 5.257)
