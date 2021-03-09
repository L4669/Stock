#! /usr/bin/python3.8

"""
DoC - December, 28th Dec, 2020
Description - for back test
Inputs by User - symbols of the pair
Outputs - <Trade Start Date>, <Trade Exit Date>, <Signal Type>, <Profit/Loss>,
<Qty. X>, <Qty. Y>
Authors - L & JJ

1. Data output in csv file
"""

import math
import numpy as np
from sklearn.linear_model import LinearRegression
import yfinance as yf
import datetime
import dateutil.relativedelta
from itertools import combinations
from scipy.stats import norm 
import time
import sys


def download_history(db_name):
    # download of full database    
    stock = yf.Ticker(db_name + ".NS")

    hist = stock.history(period="2y")
    
    # extracting close price data alone
    close_price = hist['Close'].to_numpy()
    # extracting time list
    time_frame = hist.axes[0].tolist()

    return close_price, time_frame

def M1_cdf(x, y, datalen):
    cp_ratio = []
    cdf = []

    for i in range(datalen):
        cp_ratio.append(float(y[i])/float(x[i]))

    mu = np.mean(cp_ratio)
    std = np.std(cp_ratio)

    for ratio in cp_ratio:
        cdf.append(norm(mu, std).cdf(ratio)*100)
    
    return cdf

def M1_signal(cdf):
    if (cdf >= 0.3) and (cdf <= 2.5):
        signal = "LONG"
    elif (cdf >= 97.5) and (cdf <= 99.7):
        signal = "SHORT"
    else:
        signal = "NOSIG"
    return signal

def M2_std_err(x, y):
    
    # using scikit learn module
    x_arr = np.array(x).reshape((-1, 1))
    y_arr = np.array(y)

    # =======================Error Handling==========================
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
    # ======================Error Handling Ends======================

    x_arr_ref3 = np.array(x_arr_ref2).reshape((-1, 1))
    y_arr_ref3 = np.array(y_arr_ref2)
    
    model = LinearRegression().fit(x_arr_ref3, y_arr_ref3)
    c = model.intercept_
    m = model.coef_
    
    y_pred = model.intercept_ + model.coef_ * x
    residuals = y - y_pred
    stdev_resd = np.std(residuals)

    std_err = []
    for resd in residuals:
        std_err.append(resd/stdev_resd)

    return std_err

def M2_signal(std_err):
    if (std_err <= -2.5 and std_err >= -3.0):
        signal = "LONG"
    elif (std_err >= 2.5 and std_err <= 3.0):
        signal = "SHORT"
    else:
        signal = "NOSIG"

    return signal 

def diophantine(a, b):
    # a*x - b*y = 0
    d = math.gcd(int(a), int(b))
    x = b/d
    y = a*x/b
    return int(x), int(y)

if __name__ == "__main__":
    
    # symbol names to be input by user
    Y = input("Enter Symbol for 1st company (Y): ")
    X = input("Enter Symbol for 2nd company (X): ")
    print("\n")

    
    # open file for results
    date_today = str(datetime.date.today())
    fname = "/home/euler/Documents/Projects/Stock/backtest_result_" +\
        date_today + Y + "_" + X + ".csv"
    fd = open(fname, "w+")
    
    # writing headers
    fd.write("Signal Date,")
    fd.write("Exit Date,")
    fd.write("Signal Type,")
    fd.write("Profit/Loss,")
    fd.write("Qty. X,")
    fd.write("Qty. Y,")
    fd.write("\n")

    # download historical data
    Y_data, time_frame1 = download_history(Y)
    X_data, time_frame2 = download_history(X)
    
    if len(Y_data) != len(X_data):
        print("ERROR: Unequal Length ", Y, " & ", X)
        fd.write(Y+"_"+X+","+"ERROR")
        fd.write("\n")
        sys.exit()

    datalen = len(Y_data)

    #===============================M1 Signal==============================
    cdf_arr = M1_cdf(X_data, Y_data, datalen)

    trade_inprogress = 0
    # start and end prices for X & Y
    x1 = 0
    y1 = 0
    x2 = 0
    y2 = 0

    signal = ""
    trade_start_idx = 0
    
    for i in range(datalen):
        result_bucket = ""        
        cdf = cdf_arr[i]
        if trade_inprogress == 0:
            signal = M1_signal(cdf)
            y1 = Y_data[i]
            x1 = X_data[i]
            X_qty, Y_qty = diophantine(x1, y1)
            trade_start_idx = i
        
        if signal == "LONG":
            trade_inprogress = 1
            if (cdf < 0.3) or (cdf > 2.5):
                trade_inprogress = 0
                y2 = Y_data[i]
                x2 = X_data[i]
                measure = (y2-y1)*Y_qty + (x1-x2)*X_qty
                trade_inprogress = 0
                result_bucket += str(time_frame1[trade_start_idx]) + "," + \
                    str(time_frame1[i]) + "," + "M1_" + signal + "," + \
                        str(measure) + "," + str(X_qty) + "," + str(Y_qty) + "\n"
                fd.write(result_bucket)
        elif signal == "SHORT":
            trade_inprogress = 1
            if (cdf < 97.5) or (cdf > 99.7):
                trade_inprogress = 0
                y2 = Y_data[i]
                x2 = X_data[i]
                measure = (y1-y2)*Y_qty + (x2-x1)*X_qty
                trade_inprogress = 0
                result_bucket += str(time_frame1[trade_start_idx]) + "," + \
                    str(time_frame1[i]) + "," + "M1_" + signal + "," + \
                        str(measure) + "," + str(X_qty) + "," + str(Y_qty) + "\n"
                fd.write(result_bucket)
        else:
            pass
    #======================================================================

    #===============================M2 Signal==============================
    std_err_arr = M2_std_err(X_data, Y_data)

    trade_inprogress = 0
    # start and end prices for X & Y
    x1 = 0
    y1 = 0
    x2 = 0
    y2 = 0

    signal = ""
    trade_start_idx = 0

    for i in range(datalen):
        result_bucket = ""
        std_err = std_err_arr[i]
        if trade_inprogress == 0:
            signal = M2_signal(std_err)
            y1 = Y_data[i]
            x1 = X_data[i]
            X_qty, Y_qty = diophantine(x1, y1)
            trade_start_idx = i
        
        if signal == "LONG":
            trade_inprogress = 1
            if (std_err < -3.0) or (std_err > -2.0):
                trade_inprogress = 0
                y2 = Y_data[i]
                x2 = X_data[i]
                measure = (y2-y1)*Y_qty + (x1-x2)*X_qty
                trade_inprogress = 0
                result_bucket += str(time_frame1[trade_start_idx]) + "," + \
                    str(time_frame1[i]) + "," + "M2_" + signal + "," + \
                        str(measure) + "," + str(X_qty) + "," + str(Y_qty) + "\n"
                fd.write(result_bucket)
        elif signal == "SHORT":
            trade_inprogress = 1
            if (std_err < 2) or (std_err > 3):
                trade_inprogress = 0
                y2 = Y_data[i]
                x2 = X_data[i]
                measure = (y1-y2)*Y_qty + (x2-x1)*X_qty
                trade_inprogress = 0
                result_bucket += str(time_frame1[trade_start_idx]) + "," + \
                    str(time_frame1[i]) + "," + "M2_" + signal + "," + \
                        str(measure) + "," + str(X_qty) + "," + str(Y_qty) + "\n"
                fd.write(result_bucket)
        else:
            pass
    #======================================================================

    fd.close()