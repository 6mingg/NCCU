import pandas as pd
import os
import numpy as np

# import data
file = '../data'
rate = pd.read_csv(os.path.join(file, 'rating.csv'))[['splticrm', 'datadate', 'spcsrc', 'tic']]
ratio = pd.read_csv(os.path.join(file, 'ratios.csv'))
ratio.drop(['cusip', 'adate', 'qdate', 'gvkey', 'permno'], axis=1, inplace=True)
col_we_need = ['rating', 'tic', 'date'] + list(pd.read_csv(os.path.join(file, 'corporate_rating.csv')).columns[6:])

# adjust the rating dataframe
rate = rate.dropna(subset = ['splticrm']) #把rating資料是空的給drop掉
rate['splticrm'] = rate['splticrm'].str.strip(' ')
rate['tic'] = rate['tic'].str.strip(' ')
com = rate['tic'].value_counts()
com = com[com>85] #至少要有85筆rating資料
rate = rate[rate['tic'].isin(com.index)]
com2 = rate.groupby('tic')['splticrm'].nunique()
com2 = com2[com2>1] # 至少層級都有變過一次
rate = rate[rate['tic'].isin(com2.index)]

# calculate the ratios
ratio = ratio[ratio['TICKER'].str.strip(' ').isin(rate['tic'])]

# 計算新的 ratio
ratio2 = pd.DataFrame()
ratio2['tic'] = ratio['TICKER']
ratio2['date'] = ratio['public_date']
ratio2['currentRatio'] = ratio['curr_ratio']
ratio2['quickRatio'] = ratio['quick_ratio']
ratio2['cashRatio'] = ratio['cash_ratio']
ratio2['daysOfSalesOutstanding'] = 365 / ratio['rect_turn']
ratio2['netProfitMargin'] = ratio['npm']
ratio2['pretaxProfitMargin'] = ratio['pretret_earnat']
ratio2['grossProfitMargin'] = ratio['gpm']
ratio2['operatingProfitMargin'] = ratio['opmbd']
ratio2['returnOnAssets'] = ratio['roa']
ratio2['returnOnCapitalEmployed'] = ratio['roce']
ratio2['returnOnEquity'] = ratio['roe']
ratio2['assetTurnover'] = ratio['at_turn']
ratio2['fixedAssetTurnover'] = ratio['sale_equity'] / ratio['equity_invcap']
ratio2['debtEquityRatio'] = ratio['totdebt_invcap'] / ratio['equity_invcap']
ratio2['debtRatio'] = ratio['totdebt_invcap'] / (ratio['equity_invcap'] + ratio['totdebt_invcap'])
ratio2['effectiveTaxRate'] = ratio['efftax']
ratio2['freeCashFlowOperatingCashFlowRatio'] = ratio['fcf_ocf'] / ratio['oancf']
ratio2['freeCashFlowPerShare'] = ratio['fcf_ocf'] / ratio['csho']
ratio2['cashPerShare'] = ratio['cash_conversion'] / ratio['csho']
ratio2['companyEquityMultiplier'] = (ratio['dltt_be'] + ratio['debt_invcap']) / ratio['equity_invcap']
ratio2['ebitPerRevenue'] = (ratio['opmbd'] + ratio['intcov']) / ratio['revt']
ratio2['enterpriseValueMultiple'] = (ratio['marketcap'] + ratio['lt_debt'] - ratio['cash_lt']) / ratio['ebitda']
ratio2['operatingCashFlowPerShare'] = ratio['oancf'] / ratio['csho']
ratio2['operatingCashFlowSalesRatio'] = ratio['oancf'] / ratio['sale']
ratio2['payablesTurnover'] = ratio['pay_turn']
remain_col = ['CAPEI', 'bm', 'evm', 'pe_op_basic', 'pe_op_dil',
       'ptpm', 'cfm', 'aftret_eq',
       'aftret_invcapx', 'aftret_equity', 'pretret_noa',
       'GProf', 'equity_invcap', 'debt_invcap', 'totdebt_invcap',
       'capital_ratio', 'int_debt', 'int_totdebt', 'cash_lt', 'invt_act',
       'rect_act', 'debt_at', 'debt_ebitda', 'short_debt', 'curr_debt',
       'lt_debt', 'profit_lct', 'ocf_lct', 'cash_debt', 'fcf_ocf', 'lt_ppent',
       'dltt_be', 'debt_assets', 'debt_capital', 'de_ratio', 'intcov',
       'intcov_ratio', 
       'cash_conversion', 'inv_turn',
       'sale_invcap', 'sale_equity', 'rd_sale', 'adv_sale',
       'staff_sale', 'accrual', 'divyield']
for c in remain_col:
    ratio2[c] = ratio[c]

rate = rate.rename(columns = {'splticrm':'rating', 'datadate':'date', 'spcsrc':'sp500 rating'})


# merge
df = pd.merge(rate, ratio2, on=['date', 'tic'], how='inner').reset_index(drop = True)
nan_count_per_column = df.isna().sum()
df2 =  df.dropna()
df2 = df2.rename(columns = {'rating':'Rating'})
df2.to_csv(os.path.join(file, 'df.csv'), index = False)
