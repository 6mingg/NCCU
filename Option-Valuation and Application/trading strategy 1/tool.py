# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 14:10:57 2023

@author: 6min9
"""

import sys
import math
import scipy
import numpy as np
import pandas as pd
import scipy.signal as sg
import matplotlib.pyplot as plt

def sma(price, n): # ex : sma20 為計算過去20天的簡單移動平均
  return price.rolling(n).mean()

def wma(price, n): # ex : wma20 為計算過去20天的加權移動平均
  return price.ewm(com=n).mean()

# Highpass filter by John F. Ehlers, converted by DdlV
def highpass(Data, n=48): #高頻成分代表價格在短期內快速變化的部分。這些變化通常是短期波動、噪音或交易活動引起的價格變動。高頻成分可能包含著各種交易者的短期交易行為，例如日內交易者或頻繁交易者對價格的快速反應。高頻成分在時間尺度上變化迅速，可能持續幾分鐘或幾小時。
  a	= (0.707*2*math.pi) / n

  alpha1 = (math.cos(a)+math.sin(a)-1)/math.cos(a);
  b	= 1-alpha1/2
  c	= 1-alpha1

  ret = [0] * len(Data)
  for i in range(2, len(Data)):
    ret[i] = b*b*(Data.iloc[i]-2*Data[i-1]+Data.iloc[i-2])+2*c*ret[i-1]-c*c*ret[i-2]

  return pd.Series(ret, index=Data.index)

# lowpass filter
def lowpass(Data,n): #低頻成分代表價格在長期內緩慢變化的趨勢部分。這些趨勢可能是長期市場趨勢或價格的長期結構。低頻成分可能反映整體市場的走勢，例如長期上漲趨勢或下跌趨勢，以及宏觀經濟因素對價格的影響。低頻成分的變化通常在時間尺度上較為緩慢，可能持續數天、數週、甚至數月。
  a = 2.0/(1+n)

  lp = [Data[0], Data[1]] + [0] * (len(Data) - 2)
  for i in range(2, len(Data)):
    lp[i] = (a-0.25*a*a)*Data[i]+ 0.5*a*a*Data[i-1]\
      - (a-0.75*a*a)*Data[i-2]\
      + 2*(1.-a)*lp[i-1]\
      - (1.-a)*(1.-a)*lp[i-2]

  return pd.Series(lp, index=Data.index)

def hullma(price, n): # 比起sma wma，更貼近價格曲線
  wma1 = wma(price, n//2)
  wma2 = wma(price, n)
  return wma(wma1 * 2 - wma2, int(math.sqrt(n)))

def zlma(price, n): # 比起sma wma，更貼近價格曲線
  """
  John Ehlers' Zero lag (exponential) moving average
  https://en.wikipedia.org/wiki/Zero_lag_exponential_moving_average
  """
  lag = (n - 1) // 2
  series = 2 * price - price.shift(lag)
  return wma(series, n)

def alma(price, n):
  # triangular window with 60 samples.
  h = sg.gaussian(n, n*0.2)

  # We convolve the signal with this window.
  fil = sg.convolve(price, h / h.sum())
  filtered = pd.Series(fil[:len(price)], index=price.index)
  return filtered

def detrend(price, n):
  return price - highpass(price, n)

def linear_reg(price, n):
  import talib
  return talib.LINEARREG(price, timeperiod=n)

def bollinger_bands(price, window, num_std):

    middle_band = price.rolling(window=window).mean()  # 计算中轨，即移动平均线
    std = price.rolling(window=window).std()  # 计算收盘价的标准差
    
    upper_band = middle_band + num_std * std  # 计算上轨
    lower_band = middle_band - num_std * std  # 计算下轨
    
    bollinger_bands_df = pd.DataFrame({
        'Middle Band': middle_band,
        'Upper Band': upper_band,
        'Lower Band': lower_band
    })
    
    return bollinger_bands_df

def percent(df, n):
    q = np.percentile(df, n)
    return q

def kdj(df, n=9, m1=3, m2=3): #隨機指標
    """
    Calculate KDJ and return the result as a DataFrame.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing price data, including 'High', 'Low', and 'Close' columns.
        n (int): Time window for calculating the highest high and lowest low. Default is 9.
        m1 (int): Moving average window for calculating K line. Default is 3.
        m2 (int): Moving average window for calculating D line. Default is 3.
        
    Returns:
        pd.DataFrame: DataFrame containing 'K', 'D', and 'J' columns.
    """
    df_re = pd.DataFrame()
    
    # Calculate RSV
    df_re['lowest_low'] = df['Low'].rolling(n).min()
    df_re['highest_high'] = df['High'].rolling(n).max()
    df_re['RSV'] = (df['Close'] - df_re['lowest_low']) / (df_re['highest_high'] - df_re['lowest_low']) * 100

    # Calculate K line using m1-day moving average of RSV
    df_re['K'] = df_re['RSV'].rolling(m1).mean()
    
    # Calculate D line using m2-day moving average of K line
    df_re['D'] = df_re['K'].rolling(m2).mean()
    
    # Calculate J line (J = 3 * K - 2 * D)
    df_re['J'] = 3 * df_re['K'] - 2 * df_re['D']
    # Drop intermediate columns
    df_re.drop(['lowest_low', 'highest_high', 'RSV'], axis=1, inplace=True)
    
    return df_re

def macd(df, short_window=12, long_window=26, signal_window=9): #移動平均匯聚與發散
    short_ema = df.ewm(span=short_window, min_periods=1, adjust=False).mean()
    long_ema = df.ewm(span=long_window, min_periods=1, adjust=False).mean()

    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_window, min_periods=1, adjust=False).mean()
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram
    