import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
#from matplotlib.finance import candlestick
from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY, date2num
import seaborn as sns
import time
import random
from sklearn import linear_model
from statsmodels.tsa.stattools import adfuller as adf
from pandas_datareader import data as web
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts # original: from pandas.stats.api import ols (outdated)
import scipy.stats as stats


stock1="FB"
stock2="MSFT"

start = datetime.datetime(2013,8, 27) #YYYY, M, D
end = datetime.datetime(2019, 8, 27)

def indexer(list1):
    indexed_list = []
    for i in range(len(list1)):
        indexed_list.append( (list1[i] / (list1)[0]) * 100 )
    return indexed_list

stock1_data = web.DataReader(stock1, "yahoo", start, end)
stock1_data['ln_close'] = np.log(indexer(stock1_data['Close']))
stock1_data[stock1]=stock1_data['Close']

stock2_data = web.DataReader(stock2, "yahoo", start, end)
stock2_data['ln_close'] = np.log(indexer(stock2_data['Close']))
stock2_data[stock2]=stock2_data['Close']



df = pd.DataFrame(index=stock1_data.index)
df[stock1] = stock1_data["Close"]
df[stock2] = stock2_data["Close"]
# Calculate optimal hedge ratio "beta"
res1 = sm.OLS(df[stock2], df[stock1])
res2 = sm.OLS(df[stock1], df[stock2])
#beta_hr = res.beta.x
#beta_hr = res2.fit().params[0]
beta_hr = res1.fit().params[0]


def run_ADF_regression(stock1_data,stock2_data):
    x_points = np.array(stock1_data[stock1])
    y_points = np.array(stock2_data[stock2])
    reg = linear_model.LinearRegression()
    reg.fit(x_points.reshape(-1, 1), y_points.reshape(-1, 1))

    return reg

def get_residuals(stock1_data,stock2_data):
    stock_a = stock1_data[stock1]
    stock_b = stock2_data[stock2]

    #stock_a.reset_index(inplace = True)
    #stock_b.reset_index(inplace = True)
    
    reg = run_ADF_regression(stock1_data,stock2_data)
    residuals = reg.intercept_ + reg.coef_[0] * stock2_data[stock2] - stock1_data[stock1]

    res_df = residuals.to_frame()

    res_df.columns = ['residuals']

    return res_df

error=[x for x in get_residuals(stock1_data,stock2_data)['residuals']]

err=pd.DataFrame(columns=['residuals'])
err['residuals']=[x for x in get_residuals(stock1_data,stock2_data)['residuals']]
err['MA20'] = err['residuals'].rolling(1000).mean()
err['20dSTD'] = err['residuals'].rolling(1000).std() 
err['Upper'] = err['MA20'] + (err['20dSTD'])
err['Lower'] = err['MA20'] - (err['20dSTD'])

err['residuals'].plot()
err['MA20'].plot()
err['Upper'].plot()
err['Lower'].plot()

plt.show()

'''
indeces=[]
for i in range(len(err)):
    if err['residuals'][i]>=err['Upper'][i] or err['residuals'][i]<=err['Lower'][i]:
        indeces.append(i)
        plt.plot(i,err['residuals'][i],'ro')
plt.show()

'''
        
































'''

def ADF_test(residuals, output_log = False, title = "ADF Test Results"):
    t0 = residuals
    t1 = residuals.shift()

    shifted = t1 - t0
    shifted.dropna(inplace = True)

    plt.plot(shifted, c='green')
    plt.show()

    adf_value = adf(shifted, regression = 'nc')

    test_statistic = adf_value[0]
    pvalue = adf_value[1]
    usedlags = adf_value[2]
    nobs = adf_value[3]


    if output_log:
            #output on figure eventually, that looks really professional
            print (title)
            print ("Test Statistic: %.4f\nP-Value: %.4f\nLags Used: %d\nObservations: %d" % (test_statistic, pvalue, usedlags, nobs))

            for crit in adf_value[4]:
                    print (crit, adf_value[4][crit])
                    print ("Critical Value (%s): %.3f" % (crit, adf_value[crit]))

    return adf_value


print(ADF_test(get_residuals(stock1_data,stock2_data)))
'''


