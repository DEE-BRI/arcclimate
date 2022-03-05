"""
気温の関するモジュール
"""

import numpy as np


def get_corrected_TMP(TMP: np.ndarray, ele_gap: float) -> np.ndarray:
    """気温の標高補正

    Args:
        TMP (np.ndarray): 気温 [℃]
        ele_gap (np.ndarray): 標高差 [m]

    Returns:
        np.ndarray: 標高補正後の気温 [C]

    Notes:
        気温減率の平均値を0.0065℃/mとする。
    """
    return TMP + ele_gap * -0.0065
