# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 23:12:11 2023

@author: USER
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

def date(datetime_str):
    date_object = datetime.strptime(datetime_str, "%Y/%m/%d")
    return date_object

def calculate_atm_price(close_price): #計算價平
    if (close_price % 50) <= 25:
        return 50 * (close_price // 50)
    else:
        return 50 * ((close_price // 50)+1)
    
def bollinger_bands(price, window, num_std): #布林通道 input(price data, 天數, 標準差)

    middle_band = price.rolling(window=window).mean()  
    std = price.rolling(window=window).std()  
    
    upper_band = middle_band + num_std * std  
    lower_band = middle_band - num_std * std  
    
    bollinger_bands_df = pd.DataFrame({
        'Middle Band': middle_band,
        'Upper Band': upper_band,
        'Lower Band': lower_band
    })
    
    return bollinger_bands_df

def sma(price, n): # ex : sma20 為計算過去20天的簡單移動平均
  return price.rolling(n).mean()

def compare_date(data1, data2):
    # 找出兩個 DataFrame 中都存在的日期值
    common_dates = set(data1.index).intersection(set(data2.index))
    # 在兩個 DataFrame 中篩選出包含共同日期的行
    filtered_df1 = data1[data1.index.isin(common_dates)]
    filtered_df2 = data2[data2.index.isin(common_dates)]
    return filtered_df1, filtered_df2

bool_window = 20
close = 1 # 設離 atm 幾個 tick
far = 2

data_file = './data/monthly data'
raw_price_data = pd.read_csv('./data/tx.csv', index_col=False).dropna()
raw_price_data['Date'] = raw_price_data['Date'].apply(date)
settle_data = pd.read_csv('./data/settle_dates.csv', index_col=False).dropna()
settle_data['最後結算日'] = settle_data['最後結算日'].apply(date)
settle_data = settle_data.set_index('最後結算日', drop = True)
raw_price_data = raw_price_data[raw_price_data['Date']<=settle_data.index[-1]]
price_data = pd.DataFrame({'date':raw_price_data['Date'], 
                           'atm_price':raw_price_data['Close'].apply(calculate_atm_price), 
                           'ori_price':raw_price_data['Close']})
price_data = price_data.set_index('date', drop = True)

# 技術指標
boolin = bollinger_bands(price_data['ori_price'], bool_window, 2)
sma50 = sma(price_data['ori_price'], 50)
tech = pd.DataFrame({'date':raw_price_data['Date'], 
                     'middle_bound':boolin['Middle Band'].values, 
                     'upper_bound':boolin['Upper Band'].values, 
                     'lower_bound':boolin['Lower Band'].values, 
                     'sma50':sma50.values}).reset_index(drop = True)
tech = tech.set_index('date', drop = True)

# When SMA(50) is above (below) upper (lower) bollinger band 
# a call (put) spread (selling call (put) out of the money and 
# the buying a further out of the money call (put)) slightly out of the money with an expiration date of 4 days.

tech['flag1'] = tech['sma50']>tech['upper_bound'] # selling call at out of money, buying a further out of the money call
tech['flag2'] = tech['sma50']<tech['lower_bound'] # s+elling puy at out of money, buying a further out of the money put

trading_information = pd.DataFrame(columns = ['date', 'strategy', 'closer strike(sell)', 'closer price(sell)', 'further strike(buy)', 'further price(buy)', 'profit'])

temp = pd.DataFrame(columns = ['date', 'contract', 'contract2'])
market_information = pd.DataFrame(columns = ['date', 'short call at k1', 'short call price', 'long call at k2', 'long call price', 'long put at k3', 'long put price', 'short put at k4', 'short put price'])

for file in os.listdir(data_file)[:]:
    temp_data = pd.read_csv(os.path.join(data_file, file), encoding='big5' ,index_col=False)
    df = temp_data[(temp_data['契約']=='TXO') & (temp_data['交易時段']=='一般')]
    df['交易日期'] = df['交易日期'].apply(date)
    df = df.set_index('交易日期', drop = True)
    df, price_data_temp= compare_date(df, price_data)
    df['到期月份(週別)'] = df['到期月份(週別)'].astype(str).str.replace(' ', '')
    for day in df.index.unique().astype('datetime64[ns]')[:]:
        contract_month = settle_data.loc[settle_data.index >= day, '契約月份'].iloc[0]
        if day in settle_data.index.values[:-1]:
            contract_month2 = settle_data.loc[settle_data.index >= day, '契約月份'].iloc[1]
            temp_df = pd.concat([df[(df.index == day) & (df['到期月份(週別)']==contract_month)], df[(df.index == day) & (df['到期月份(週別)']==contract_month2)]], axis=0)
        else :
            temp_df = df[(df.index == day) & (df['到期月份(週別)']==contract_month)]
            contract_month2 = 0
        temp.loc[len(temp)] = [day, contract_month, contract_month2]
        atm = price_data_temp.loc[price_data_temp.index == day, 'atm_price'].iloc[0]
        

        # strategy
        strike_price = temp_df['履約價'].unique()
            
        if atm not in strike_price:
            if atm>price_data.loc[day, 'ori_price']:
                atm = atm-50
            elif atm<price_data.loc[day, 'ori_price']:
                atm = atm+50
        
        atm_id = np.where(strike_price == atm)[0][0]
        
        if atm_id<far :
            # for fig 1 => sell call spread => short at k1 call, long at k2 call (spot price < k1 < k2)
            k1 = strike_price[np.where(strike_price == atm)[0][0]]
            c1 = temp_df.loc[(temp_df['履約價'].values == k1)&(temp_df['買賣權'].values == '買權'), '收盤價']
            
            k2 = strike_price[np.where(strike_price == atm)[0][0] + close]
            c2 = temp_df.loc[(temp_df['履約價'].values == k2)&(temp_df['買賣權'].values == '買權'), '收盤價']
            
            # for fig 2 => sell call spread => long at k1 put, short at k2 put (k1 < k2 < spot price)
            k3 = strike_price[np.where(strike_price == atm)[0][0] - close]
            p1 = temp_df.loc[(temp_df['履約價'].values == k3)&(temp_df['買賣權'].values == '賣權'), '收盤價']
            
            k4 = strike_price[np.where(strike_price == atm)[0][0]]
            p2 = temp_df.loc[(temp_df['履約價'].values == k4)&(temp_df['買賣權'].values == '賣權'), '收盤價']
        else:  
            # for fig 1 => sell call spread => short at k1 call, long at k2 call (spot price < k1 < k2)
            k1 = strike_price[np.where(strike_price == atm)[0][0] + close]
            c1 = temp_df.loc[(temp_df['履約價'].values == k1)&(temp_df['買賣權'].values == '買權'), '收盤價']
            
            k2 = strike_price[np.where(strike_price == atm)[0][0] + far]
            c2 = temp_df.loc[(temp_df['履約價'].values == k2)&(temp_df['買賣權'].values == '買權'), '收盤價']
            
            # for fig 2 => sell call spread => long at k1 put, short at k2 put (k1 < k2 < spot price)
            k3 = strike_price[np.where(strike_price == atm)[0][0] - far]
            p1 = temp_df.loc[(temp_df['履約價'].values == k3)&(temp_df['買賣權'].values == '賣權'), '收盤價']
            
            k4 = strike_price[np.where(strike_price == atm)[0][0] - close]
            p2 = temp_df.loc[(temp_df['履約價'].values == k4)&(temp_df['買賣權'].values == '賣權'), '收盤價']
        
        market_information.loc[len(market_information)] = [day, k1, c1, k2, c2, k3, p1, k4, p2] # 結算日中有兩個c1c2p1p2，一個是上個契約的結束，一個是下個契約的開始


market_information = market_information.sort_values(by='date').set_index('date', drop = True)
market_information['flag1'] = tech['flag1']
market_information['flag2'] = tech['flag2']
market_information['spot price'] = price_data['ori_price']

trading_information = pd.DataFrame(columns = ['date', 'strategy', 'k1', 'price at k1(short call / long put)', 'k2', 'price at k2(long call / short put)', 'spot price', 'profit'])
trading_history = pd.DataFrame(columns = list(trading_information.columns))
cost = 3
for i, row in market_information.iterrows():
    flag1 = row['flag1'] #for fig 1 => sell call spread => short at k1 call, long at k2 call (spot price < k1 < k2)
    flag2 = row['flag2'] # for fig 2 => sell put spread => long at k1 put, short at k2 put (k1 < k2 < spot price)
    spot_p = float(row['spot price'])
    if (trading_information.empty is not True) and (i in settle_data.index.values[:-1]) :
        trading_history = pd.concat([trading_history, trading_information], axis = 0)
        if trading_information['strategy'].iloc[0] == '1':
            pnl_sc, pnl_lc = 0, 0
            for j, row2 in trading_information.iterrows():
                pnl_sc = pnl_sc + row2['price at k1(short call / long put)'] - cost if spot_p <= row2['k1'] else pnl_sc + row2['k1'] - spot_p + row2['price at k1(short call / long put)'] - cost
                pnl_lc = pnl_lc - row2['price at k2(long call / short put)'] - cost if spot_p <= row2['k2'] else pnl_lc + spot_p - row2['k2'] - row2['price at k2(long call / short put)'] - cost
            pnl_tt = pnl_sc + pnl_lc
            trading_history.loc[len(trading_history)] = [i, 'close 1', row[0], row[1][0], row[2], row[3][0], spot_p, pnl_tt]
            trading_information.drop(trading_information.index, inplace=True)
        elif trading_information['strategy'].iloc[0] == '2':
            pnl_lp, pnl_sp = 0, 0
            for j, row2 in trading_information.iterrows():
                pnl_lp = pnl_lp - row2['price at k1(short call / long put)'] - cost if spot_p >= row2['k1'] else pnl_lp + row2['k1'] - spot_p - row2['price at k1(short call / long put)'] - cost
                pnl_sp = pnl_sp + row2['price at k2(long call / short put)'] - cost if spot_p >= row2['k2'] else pnl_sp + spot_p - row2['k2'] + row2['price at k2(long call / short put)'] - cost
                pnl_tt = pnl_lp + pnl_sp
            trading_history.loc[len(trading_history)] = [i, 'close 2', row[4], row[5][0], row[6], row[7][0], spot_p, pnl_tt]
            trading_information.drop(trading_information.index, inplace=True)
        
    if flag1:
        if len(row[1])==2:
            trading_information.loc[len(trading_information)] = [i, '1', float(row[0]), float(row[1][1]), float(row[2]), float(row[3][1]), spot_p, -6]
        else:
            trading_information.loc[len(trading_information)] = [i, '1', float(row[0]), float(row[1][0]), float(row[2]), float(row[3][0]), spot_p, -6]
    elif flag2:
        if len(row[1])==2:
            trading_information.loc[len(trading_information)] = [i, '2', float(row[4]), float(row[5][1]), float(row[6]), float(row[7][1]), spot_p, -6]
        else:
            trading_information.loc[len(trading_information)] = [i, '2', float(row[4]), float(row[5][0]), float(row[6]), float(row[7][0]), spot_p, -6]
        
trading_history = trading_history.sort_values(by='date').set_index('date', drop = True)
trading_history['rt'] = trading_history['profit'].cumsum()

flag1_rt = trading_history[trading_history['strategy'] == 'close 1']['profit'].sum()
flag2_rt = trading_history[trading_history['strategy'] == 'close 2']['profit'].sum()
final_rt = flag1_rt + flag2_rt

# 股票價格，履約價k，選擇權進場日的價格，when結算日進場，就用下一個，抱到結算日，現在要解決在market information 中有empty的row

# 求 put call ratio(不確定用不用的到)
# =============================================================================
# infor = pd.DataFrame(columns = ['日期', 'call_volume', 'put_volume', 'PutCallRatio'])
# for file in os.listdir(data_file):
#     temp_data = pd.read_csv(os.path.join(data_file, file), encoding='big5' ,index_col=False)
#     df = temp_data[(temp_data['契約']=='TXO') & (temp_data['交易時段']=='一般')]
#     trading_day = df['交易日期'].unique()
#     for day in trading_day:
#         buy_v = df[(df['買賣權'] == '買權') & (df['交易日期'] == day)]['成交量'].sum()
#         sell_v = df[(df['買賣權'] == '賣權') & (df['交易日期'] == day)]['成交量'].sum()
#         infor.loc[len(infor)] = [day, buy_v, sell_v, sell_v/buy_v]
# =============================================================================







