import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
#from ProcessingData.correlation import data_group
plt.rcParams.update({'figure.figsize':(8,8), 'figure.dpi':120})

def estimate_p_d_q(data,col):
    from statsmodels.tsa.stattools import adfuller


    X = data.iloc[:,col]
    #print(X.head())
    re = adfuller(X.values)
    print(f'p_value: {re[1]}')

    fig, axis = plt.subplots(3,3)
    
    axis[0,0].plot(X.values); axis[0,0].set_title('original series')
    plot_acf(X.values,ax=axis[0,1])
    plot_pacf(X.values,ax=axis[0,2])
    
    axis[1,0].plot(X.diff().values); axis[1,0].set_title('diff1 series')
    plot_acf(X.diff().dropna().values,ax=axis[1,1])
    plot_pacf(X.diff().dropna().values,ax=axis[1,2])
    
    axis[2,0].plot(X.diff().diff().values); axis[2,0].set_title('diff2 series')
    plot_acf(X.diff().diff().dropna().values,ax=axis[2,1])
    plot_pacf(X.diff().diff().dropna().values,ax=axis[2,2])

    plt.savefig('./Log/Arima/autocorr_h.png') 
    plt.show()

#NOTE: The result show that the series of Q become over stationary if applied 1 diff(), d = 0
        # and p_value << 0.05 => stationary, partial corr show that lag 1 and 2 are above significant limit, take p = 1 or 2
        #many lags above significant limit at autocorr, take q= 3 for the two > 0.6

        #H is also nearly statinary, d =0 , pcorr show that should take  p = 1, q = 5

def train(data,col,target_timestep=1):
    data = data.iloc[:,col]
    #diff_dat = data.diff().dropna(axis=0)
    
    X = data.values
    print(X.shape)
    print(X[:5])

    train, test = X[:672], X[672:]
    # model = ARIMA(train,order=(1,0,5))
    # model_fit = model.fit(disp=0)
    # print(model_fit.summary())

    # model_fit.plot_predict(dynamic=False)
    # plt.show()
    predict = []
    his = [x for x in train] #copy train set
    i = 0
    while i <= len(test) - target_timestep:
        model = ARIMA(his,order=(1,0,1)) #roll forecasting
        model_fit = model.fit(disp=0)

        out, se, conf = model_fit.forecast(target_timestep) #predict 15 timesteps

        predict += list(out)
        his += list(out) #after predict 15 day, append the real value of that day to the training set
        i += target_timestep
        #print(f'Expect val: {test[i]}, Predict val: {yhat}')

    mae = mean_absolute_error(test,predict)
    mse = mean_squared_error(test,predict)
    r2 = r2_score(test,predict)

    print(f'R2: {r2} MAE: {mae}, MSE: {mse}')

    plt.plot(test,label='truth')
    plt.plot(predict,label='predict')
    plt.legend(loc='best')
    plt.savefig('./Log/Arima/predict_h.png')
    plt.show()

#NOTE: for predicting 15 timestep of Q, MAE: 32.1879945, MSE: 4622.17393
    # for predicting 15 timesteps of H, MAE: 0.1775146074982923, MSE: 0.06569121362830027
    # for 5 timesteps of H - R2: 0.7547666837422088 MSE 0.13861548894711695, MSE: 0.04969416864527996
    # 5 ts Q - R2:0.41013856602 MAE: 27.212760111253523, MSE: 4285.637923301549
if __name__=='__main__':
    # import sys
    # import os
    # sys.path.append('../')
    # print(sys.path)
    data = pd.read_csv('./RawData/urqua_river.csv',header=None,sep='\t')
    #data = data.drop('time',axis=1)
    data = data.dropna(axis=0)
    #estimate_p_d_q(data,2)
    train(data,2)

    
        