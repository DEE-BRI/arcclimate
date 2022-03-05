"""
風速風向計算モジュール
"""

import numpy as np


def get_wind16(UGRD: np.ndarray, VGRD: np.ndarray):
    """ベクトル風速から16方位の風向風速を計算

    Args:
      UGRD(np.ndarray): 東西のベクトル成分
      VGRD(np.ndarray): 南北のベクトル成分

    Returns:
      Tuple[np.ndarray, np.ndarray]: 16方位の風速と風向
    """

    # 風速 
    # 三平方の定理により、東西、南北のベクトル成分から風速を計算
    w_spd = np.sqrt(np.power(UGRD, 2) + np.power(VGRD, 2))

    # 風向
    # 東西、南北のベクトル成分から風向を計算
    w_dir = np.degrees(np.arctan2(UGRD, VGRD) + np.pi)

    # 16方位への丸め処理
    w_dir16 = np.round(w_dir / 22.5, decimals=0) * 22.5
    w_dir16_gap = np.abs(w_dir16 - w_dir)
    w_spd16 = np.cos(np.radians(w_dir16_gap)) * w_spd

    return w_spd16, w_dir16
