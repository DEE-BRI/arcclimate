"""
相対湿度、水蒸気分圧および露点温度の計算モジュール
"""

import numpy as np

def func_RH_eSAT(MR: np.ndarray, TMP: np.ndarray, PRES: np.ndarray
) -> np.ndarray:
    """相対湿度と水蒸気分圧を求める
    Args:
      MR(np.ndarray): 補正前重量絶対湿度 (Mixing Ratio) [g/kg(DA)]
      TMP(np.ndarray): 気温 [C]
      PRES(np.ndarray): 気圧 [Pa]

    Returns:
      np.ndarray: 相対湿度[%]
      np.ndarray: 水蒸気分圧 [hpa]
    """
    P = PRES/100 # hpa
    T = TMP + 273.15  # 絶対温度
    VH = MR*(P/(T*2.87)) 
    
    eSAT = np.exp( -5800.2206 / T + 1.3914993 - 0.048640239 * T + 0.41764768 * 10 ** (-4) * T**2\
               -0.14452093 * 10 ** (-7) * T**3 + 6.5459673 * np.log( T )) / 100 # hPa
    aT = ( 217 * eSAT) / T
    RH = VH / aT * 100
    Pw = RH / 100 * eSAT  # hPa
    
    return RH,Pw
    
def func_DT_0(Pw):
    """水蒸気分圧から気温（露点温度）を求める近似式
    パソコンによる空気調和計算法 著:宇田川光弘,オーム社, 1986.12 より
    0.039 <= Pw(hpa) < 6.112（-50～0℃の時）
    Args:
      Pw:水蒸気分圧(hpa)

    Returns:
      np.ndarray: 露点温度[℃]
    """
    Y = np.log(Pw*100) # Pa
    return -60.662 + 7.4624*Y + 0.20594*Y**2 + 0.016321*Y**3

def func_DT_50(Pw):
    """水蒸気分圧から気温（露点温度）を求める近似式
    パソコンによる空気調和計算法 著:宇田川光弘,オーム社, 1986.12 より
    6.112 <= Pw(hpa) <= 123.50（0～50℃の時）
    Args:
      Pw:水蒸気分圧(hpa)

    Returns:
      np.ndarray: 露点温度[℃]
    """
    Y = np.log(Pw*100) # Pa
    return -77.199 + 13.198*Y -0.63772*Y**2 + 0.071098*Y**3
