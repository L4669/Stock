#! /usr/bin/python3.6

"""
DoC - December, 6th Dec, 2020
Description - Code to assist in pair trading of stocks
Inputs by User - symbols for stock to be analyzed
Outputs - Statistics related to linear regression and ADF test.
Authors - L & JJ

1. No data is stored locally. historical data of last 24 months 
   downloaded and saved in RAM. Stock split adjusted data 
   Yahoo Finance library is used for this purpose.
2. Interval of 24 months is hard coded in the code
3. Regression is calculated using two methods for cross verification
4. Based on p-values from ADF test, stationarity of time series data is interpreted.
5. Internet connection is required to download the data from nse website
"""


import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import yfinance as yf
import datetime
import dateutil.relativedelta
import matplotlib.pyplot as plt
import sys


def download_history(db_name):
    # download of full database
    print('Beginning data download for ', db_name)
    
    stock = yf.Ticker(db_name + ".NS")

    hist = stock.history(period="2y")

    print("Data download finished.")
    print("\n")
    
    # extracting close price data alone
    close_price = hist['Close'].to_numpy()
    # extracting time list
    time_frame = hist.axes[0].tolist()

    return close_price, time_frame


def regression(db1, db2):
    x = db1
    y = db2

    # using scikit learn module
    x_arr = np.array(x).reshape((-1, 1))
    y_arr = np.array(y)

    nan_list_x = np.argwhere(np.isnan(x_arr))
    inf_list_x = np.argwhere(np.isinf(x_arr))

    remove_list = []
    for i in range(len(nan_list_x)):
        remove_list.append(nan_list_x[i][0])
    for i in range(len(inf_list_x)):
        remove_list.append(inf_list_x[i][0])
    
    x_arr_ref1 = np.delete(x_arr, remove_list)
    y_arr_ref1 = np.delete(y_arr, remove_list)
    
    nan_list_y = np.argwhere(np.isnan(y_arr))
    inf_list_y = np.argwhere(np.isinf(y_arr))

    remove_list = []
    for i in range(len(nan_list_y)):
        remove_list.append(nan_list_y[i][0])
    for i in range(len(inf_list_y)):
        remove_list.append(inf_list_y[i][0])
    
    x_arr_ref2 = np.delete(x_arr_ref1, remove_list)
    y_arr_ref2 = np.delete(y_arr_ref1, remove_list)

    x_arr_ref3 = np.array(x_arr_ref2).reshape((-1, 1))
    y_arr_ref3 = np.array(y_arr_ref2)
    model = LinearRegression().fit(x_arr_ref3, y_arr_ref3)
    c = model.intercept_
    m = model.coef_
    
    y_pred = model.intercept_ + model.coef_ * x
    residuals = y - y_pred
    stdev_resd = np.std(residuals)
        
    # using statmodels module
    x_mod = sm.add_constant(x)
    sm_model = sm.OLS(y, x_mod)
    sm_res = sm_model.fit()

    return x, y, y_pred, residuals, stdev_resd, c, m, sm_res


if __name__ == "__main__":
    
    # symbol names to be input by user
    db_1 = input("Enter Symbol for 1st company: ")
    db_2 = input("Enter Symbol for 2nd company: ")
    print("\n")

    # download historical data
    db1_data, time_frame1 = download_history(db_1)
    db2_data, time_frame2 = download_history(db_2)
    datalen = len(db1_data)
    if len(db1_data) != len(db2_data):
        sys.exit("ERROR: Length of data is different from both stock")

    # linear regression
    x1, y1, y1_pred, residuals1, stdev_resd1, c1, m1, sm_res1 =\
            regression(db1_data, db2_data)
    intercept_stderr_1 = sm_res1.bse[0]
    err_ratio_1 = intercept_stderr_1/stdev_resd1

    x2, y2, y2_pred, residuals2, stdev_resd2, c2, m2, sm_res2 =\
            regression(db2_data, db1_data)
    intercept_stderr_2 = sm_res2.bse[0]
    err_ratio_2 = intercept_stderr_2/stdev_resd2
    
    print("With X: ", db_1, " and Y: ", db_2, ", Error Ratio = ", err_ratio_1)
    print("With X: ", db_2, " and Y: ", db_1, ", Error Ratio = ", err_ratio_2)
    print("\n")

    if err_ratio_2 < err_ratio_1:
        m = m2
        c = c2
        residuals = residuals2
        stdev_resd = stdev_resd2
        sm_res = sm_res2
        print("Selecting X: ", db_2, " and Y: ", db_1, " for further processing") 
    else:
        m = m1
        c = c1
        residuals = residuals1
        stdev_resd = stdev_resd1
        sm_res = sm_res1
        print("Selecting X: ", db_1, " and Y: ", db_2, " for further processing") 
    
    print("Intercept: ", c)
    print("Slope: ", m)
    print("Std. Error for current residual: ", \
            residuals[datalen-1]/stdev_resd)
    print("Standard Deviation of residuals: ", stdev_resd)
    print("\n")


    print("Sklearn module Result Summary")
    print("==============================================")
    print("Slope: ", m[0])
    print("Intercept: ", c)
    print("Std. Dev. of Residuals: ", stdev_resd)
    print("==============================================")
    print("\n")
    
    print(sm_res.summary())
    #print(dir(sm_res))

    # Augmented Dickey–Fuller test on residuals
    result = adfuller(residuals, regression='nc')
    print("\n")
    print("===========ADF Summary==============")
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
            print('\t%s: %.3f' % (key, value))
    print("===================================") 

    # graph for trend visualization
    plt.plot(time_frame1, residuals)
    plt.hlines(stdev_resd, time_frame1[0], \
        time_frame1[datalen-1], 'r', '--', label='1 SD')
    plt.hlines(-stdev_resd, time_frame1[0], \
        time_frame1[datalen-1], 'r', '--')
    plt.hlines(2*stdev_resd, time_frame1[0], \
        time_frame1[datalen-1], 'y', '--', label='2 SD')
    plt.hlines(-2*stdev_resd, time_frame1[0], \
        time_frame1[datalen-1], 'y', '--')
    plt.hlines(3*stdev_resd, time_frame1[0], \
        time_frame1[datalen-1], 'g', '--', label='3 SD')
    plt.hlines(-3*stdev_resd, time_frame1[0], \
        time_frame1[datalen-1], 'g', '--')
    plt.hlines(np.mean(residuals), time_frame1[0], \
        time_frame1[datalen-1], 'k', '--', label='Mean')
    plt.legend()
    plt.show()
