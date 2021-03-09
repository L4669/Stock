#! /usr/bin/python3.8

"""
DoC - December, 28th Dec, 2020
Description - for back test
Inputs by User - symbol file in format : <company name-Y_company_name-X>
Outputs - <pairs>, <%efficiency M1>, <%efficiency M2> 
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
    if (std_err <= -2.5):
        signal = "LONG"
    elif (std_err >= 2.5):
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
    
    # reading symbol pairs from list file
    fname = "/home/euler/Documents/Projects/Stock/batch_result_filtered.csv"
    fd = open(fname, "r")
    lines = fd.readlines()
    fd.close()

    symbols_arr = []
    for line in lines[1:]:
        line = line.strip()
        var = line.split(",")
        var1 = var[0].split("_")
        symbols_arr.append([var1[0], var1[1]])

    
    # open file for results
    date_today = str(datetime.date.today())
    fname = "/home/euler/Documents/Projects/Stock/backtest_result_" +\
        date_today+".csv"
    fd = open(fname, "w+")
    
    # writing headers
    fd.write("Pairs,")
    fd.write("M1-Efficiency(%),")
    fd.write("M2-Efficiency(%),")
    fd.write("\n")
    

    for comb in symbols_arr[:25]:
        result_bucket = ""

        Y = comb[0] # Y as per nomenclature we follow
        X = comb[1] # X as per nomenclature we follow
        print(Y, X)
        result_bucket += Y + "_" + X + ","

        # download historical data
        Y_data, time_frame1 = download_history(Y)
        X_data, time_frame2 = download_history(X)
        
        if len(Y_data) != len(X_data):
            print("ERROR: Unequal Length ", Y, " & ", X)
            fd.write(Y+"_"+X+","+"ERROR")
            fd.write("\n")
            continue

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
        profit_trades = 0
        loss_trades = 0
        neutral_trades = 0 
        for i in range(datalen):
            cdf = cdf_arr[i]
            if trade_inprogress == 0:
                signal = M1_signal(cdf)
                y1 = Y_data[i]
                x1 = X_data[i]
                X_qty, Y_qty = diophantine(x1, y1)
                #print(X_qty*x1 - Y_qty*y1)
            
            if signal == "LONG":
                trade_inprogress = 1
                if (cdf < 0.3) or (cdf > 2.5):
                    trade_inprogress = 0
                    y2 = Y_data[i]
                    x2 = X_data[i]
                    measure = (y2-y1)*Y_qty + (x1-x2)*X_qty
                    if measure > 0:
                        profit_trades += 1
                    elif measure < 0:
                        loss_trades += 1
                    else:
                        neutral_trades += 1
                    trade_inprogress = 0
            elif signal == "SHORT":
                trade_inprogress = 1
                if (cdf < 97.5) or (cdf > 99.7):
                    trade_inprogress = 0
                    y2 = Y_data[i]
                    x2 = X_data[i]
                    measure = (y1-y2)*Y_qty + (x2-x1)*X_qty
                    if measure > 0:
                        profit_trades += 1
                    elif measure < 0:
                        loss_trades += 1
                    else:
                        neutral_trades += 1
                    trade_inprogress = 0
            else:
                pass
        
        try:
            M1_eff_per = 100*profit_trades/(profit_trades+\
                loss_trades+neutral_trades)
        except:
            M1_eff_per = "NO TRADE"

        result_bucket += str(M1_eff_per) + ","
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
        profit_trades = 0
        loss_trades = 0
        neutral_trades = 0 
        for i in range(datalen):
            std_err = std_err_arr[i]
            if trade_inprogress == 0:
                signal = M2_signal(std_err)
                y1 = Y_data[i]
                x1 = X_data[i]
                X_qty, Y_qty = diophantine(x1, y1)
                #print(X_qty*x1 - Y_qty*y1)
            
            if signal == "LONG":
                trade_inprogress = 1
                if (std_err < -3.0) or (std_err > -2.0):
                    trade_inprogress = 0
                    y2 = Y_data[i]
                    x2 = X_data[i]
                    measure = (y2-y1)*Y_qty + (x1-x2)*X_qty
                    if measure > 0:
                        profit_trades += 1
                    elif measure < 0:
                        loss_trades += 1
                    else:
                        neutral_trades += 1
                    trade_inprogress = 0
            elif signal == "SHORT":
                trade_inprogress = 1
                if (std_err < 2) or (std_err > 3):
                    trade_inprogress = 0
                    y2 = Y_data[i]
                    x2 = X_data[i]
                    measure = (y1-y2)*Y_qty + (x2-x1)*X_qty
                    if measure > 0:
                        profit_trades += 1
                    elif measure < 0:
                        loss_trades += 1
                    else:
                        neutral_trades += 1
                    trade_inprogress = 0
            else:
                pass

        try:
            M2_eff_per = 100*profit_trades/(profit_trades+\
                loss_trades+neutral_trades)
        except:
            M2_eff_per = "NO TRADE"

        result_bucket += str(M2_eff_per)
        #======================================================================

        fd.write(result_bucket)
        fd.write("\n")

        time.sleep(0.5)

    fd.close()

        
        



                




