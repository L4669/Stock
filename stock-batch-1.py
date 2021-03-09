#! /usr/bin/python3.8

"""
DoC - December, 20th Dec, 2020
Description - Code to assist in pair trading of stocks, analyze all
combinations of symbols as per list provided by user
Inputs by User - symbol file in format : <company name>, <symbol>
Outputs - Statistics related to linear regression and ADF test.
Authors - L & JJ

1. uses the base code from stock-4 program
2. utilizes yfinance api
3. hard coded time interval 2 years
4. yfinance limit per hour - 2000 requests, 
   intentional delay to be used to prevent IP block
5. Code captures unequal length data, NaN values and Inf values
   related error
6. Data output in csv file
"""


import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import yfinance as yf
import datetime
import dateutil.relativedelta
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import norm 
import time


def download_history(db_name):
    # download of full database    
    stock = yf.Ticker(db_name + ".NS")

    hist = stock.history(period="2y")
    
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


def M1_signal(x, y, datalen):
    cp_ratio = []

    for i in range(datalen):
        cp_ratio.append(y[i]/x[i])

    mu = np.mean(cp_ratio)
    std = np.std(cp_ratio)
    cdf = norm(mu, std).cdf(cp_ratio[datalen-1])*100

    signal = ""
    if (cdf >= 0.3) and (cdf <= 2.5):
        signal = "LONG"
    elif (cdf >= 97.5) and (cdf <= 99.7):
        signal = "SHORT"
    else:
        signal = "NOSIG"
    return signal


def M2_signal(std_err):
    signal = ""
    if (std_err <= -2.5):
        signal = "LONG"
    elif (std_err >= 2.5):
        signal = "SHORT"
    else:
        signal = "NOSIG"

    return signal 


def correlation(x, y):
    corr_coeff = np.cov(x, y, bias=True)[0][1]/np.std(x)/np.std(y)
    return corr_coeff


if __name__ == "__main__":
    
    # reading symbols from list file
    fname = "/home/euler/Documents/Projects/Stock/list.csv"
    fd = open(fname, "r")
    lines = fd.readlines()
    fd.close()

    symbols_arr = []
    for line in lines[1:]:
        var = line.split(",")
        symbols_arr.append(var[1].strip())

    symbol_comb = list(combinations(symbols_arr, 2))
    
    # open file for results
    date_today = str(datetime.date.today())
    fname = "/home/euler/Documents/Projects/Stock/batch_result_" +\
        date_today+".csv"
    fd = open(fname, "w+")
    
    # writing headers
    fd.write("Pairs,")
    fd.write("M1-signal,")
    fd.write("Intercept,")
    fd.write("Slope,")
    fd.write("p-value,")
    fd.write("std_err,")
    fd.write("Correlation,")
    fd.write("M2-signal")
    fd.write("\n")

    counter = 0
    for comb in symbol_comb[:100]:
        result_bucket = ""
        x = []
        y = []
        m1_sig = ""
        m2_sig = ""
        
        db_1 = comb[0]
        db_2 = comb[1]

        # download historical data
        db1_data, time_frame1 = download_history(db_1)
        db2_data, time_frame2 = download_history(db_2)
        datalen = len(db1_data)
        if len(db1_data) != len(db2_data):
            print("ERROR: Unequal Length ", db_1, " & ", db_2)
            fd.write(db_1+"_"+db_2+","+"ERROR")
            fd.write("\n")
            continue

        # linear regression
        x1, y1, y1_pred, residuals1, stdev_resd1, c1, m1, \
            sm_res1 = regression(db1_data, db2_data)
        intercept_stderr_1 = sm_res1.bse[0]
        err_ratio_1 = intercept_stderr_1/stdev_resd1

        x2, y2, y2_pred, residuals2, stdev_resd2, c2, m2, \
            sm_res2 = regression(db2_data, db1_data)
        intercept_stderr_2 = sm_res2.bse[0]
        err_ratio_2 = intercept_stderr_2/stdev_resd2

        if err_ratio_2 < err_ratio_1:
            m = m2
            c = c2
            residuals = residuals2
            stdev_resd = stdev_resd2
            sm_res = sm_res2
            x = db2_data
            y = db1_data
            result_bucket += db_2+"_"+db_1+","
        else:
            m = m1
            c = c1
            residuals = residuals1
            stdev_resd = stdev_resd1
            sm_res = sm_res1
            x = db1_data
            y = db2_data
            result_bucket += db_1+"_"+db_2+","
        
        std_err = residuals[datalen-1]/stdev_resd
        corr_per = correlation(x, y)*100
        m1_sig = M1_signal(x, y, datalen)
        m2_sig = M2_signal(std_err)

        # Augmented Dickey–Fuller test on residuals
        adf_result = adfuller(residuals, regression='nc')
        p_value = adf_result[1]*100

        result_bucket += m1_sig + "," + str(c) + "," + \
            str(m[0]) + "," + str(p_value) + "," + str(std_err) +\
                "," + str(corr_per) + "," + m2_sig + "\n"
        print("Writing data for ", db_1, " & ", db_2, " ...")
        fd.write(result_bucket)
        time.sleep(1)

    fd.close()