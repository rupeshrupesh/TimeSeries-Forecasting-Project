# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 12:59:08 2019

@author: Rupesh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')


df = pd.read_csv('TimeSeries-1.csv', parse_dates=['period'], index_col='period', date_parser=dateparse)

df.index

df.head()

dataset = df.drop(['Unnamed: 0', 'ear', 'Month'], axis=1)

dataset.index

dataset.head()

dataset[:'2018-10-01']

# Making a Series of brand A from the dataframe
ts = dataset['A']

# Plotting the line plot for all six years
plt.plot(ts)

from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries):
    #Determing rolling statistics
    rolmean = ts.rolling(window=12).mean()
    rolstd = ts.rolling(window=12).std()
     #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
     #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], 
                         index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

# Performing dickey-fuller test on the series
test_stationarity(ts[:'2019-10-01'])
  # Looking at the p-value we can make a inference that the Series is not Stationary

"""
ts_log = np.log(ts)
plt.plot(ts_log)

ts['2019-08-01':'2019-10-01'].mean()
ts['2018-10-01']
ts['2013-01-01':'2018-12-01'].sum()
ts[('2013-01-01','2014-01-01','2015-01-01','2016-01-01','2017-01-01','2018-01-01'),:]

moving_avg = ts_log.rolling(window=12).mean()
plt.plot(ts_log)
plt.plot(moving_avg, color='red')
plt.show()

ts_log_moving_avg_diff = ts_log - moving_avg
ts_log_moving_avg_diff.head(12)


ts_log_moving_avg_diff.dropna(inplace=True)
test_stationarity(ts_log_moving_avg_diff)

ts_log_diff = ts_log - ts_log.shift()
plt.plot(ts_log_diff)

ts_log_diff.dropna(inplace=True)
test_stationarity(ts_log_diff)

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts_log.dropna())

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(ts_log, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()

ts_log_decompose = residual
ts_log_decompose.dropna(inplace=True)
test_stationarity(ts_log_decompose)
"""

# Looking at the ACF and PACF plots.
#ACF and PACF plots:
from statsmodels.tsa.stattools import acf, pacf

lag_acf = acf(ts.dropna(), nlags=20)
lag_pacf = pacf(ts.dropna(), nlags=20, method='ols')

#Plot ACF: 
plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()

"""
from statsmodels.tsa.arima_model import ARIMA

# code for AR model
model = ARIMA(ts.dropna(), order=(2, 1, 0))  
results_AR = model.fit(disp=-1)  
#plt.plot(ts_log_diff.dropna())
plt.plot(results_AR.fittedvalues, color='red')
#plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-ts_log_diff)**2))

# code for MA model
model = ARIMA(ts.dropna(), order=(0, 1, 2))  
results_MA = model.fit(disp=-1)  
#plt.plot(ts_log_diff)
plt.plot(results_MA.fittedvalues, color='red')
#plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-ts_log_diff)**2))

#code for ARIMA model order - (3,1,3)
model = ARIMA(ts.dropna(), order=(3, 1, 3))  
results_ARIMA = model.fit(disp=-1)  
#plt.plot(ts_log_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')
#plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_log_diff)**2))
print(results_ARIMA.fittedvalues)

# code for ARIMA model order - (2,1,1)
model = ARIMA(ts_log.dropna(), order=(2, 1, 1))  
results_ARIMA = model.fit(disp=-1)  
plt.plot(ts_log_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')
#plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_log_diff)**2))
print(results_ARIMA.fittedvalues)
print(results_ARIMA.summary())

"""

"""
actual = ts['2019-08-01':'2019-10-01']
model_112 = results_ARIMA.fittedvalues['2019-08-01':'2019-10-01']

(abs(actual - model_112)/actual).mean()

model_211 = results_ARIMA.fittedvalues['2019-08-01':'2019-10-01'].sum()

abs(actual-model_112)/actual

abs(actual-model_211)/actual


model = ARIMA(ts_log.dropna(), order=(2, 1, 1))  
results_ARIMA = model.fit(disp=-1) 
print(results_ARIMA.summary())

""""

#code for SARIMAX model with seasonal order (1,0,0,12)
import statsmodels.api as sm

sm.tsa.statespace.SARIMAX

my_order = (2,1,0)
seasonal_order = (1,0,0)
model = sm.tsa.statespace.SARIMAX(ts.dropna(), order=my_order,seasonal_order=(1,0,0,12))
model_fit = model.fit(disp=-1)
yhat = model_fit.forecast(1)

print(model_fit.summary())

actual = ts['2019-08-01':'2019-10-01']
model_sarima = model_fit.fittedvalues['2019-08-01':'2019-10-01']
error_pct = (abs(actual - model_sarima)/actual).mean()

plt.plot(ts.dropna())
plt.plot(model_fit.fittedvalues,color='r')
plt.show()
print(error_pct)



