import math
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, mean_squared_log_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_model import ARMA, ARIMA
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm
import pandas as pd
import warnings
import itertools
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')


#from ProcessingData.correlation import data_group
plt.rcParams.update({'figure.figsize': (20, 12), 'figure.dpi': 120})


def test_stationarity(timeseries):
    # Determing rolling statistics
    rolmean = timeseries.rolling(window=30).mean()
    rolstd = timeseries.rolling(window=30).std()

    # Plot rolling statistics:
    fig, axis = plt.subplots(1, 3, figsize=(20, 6))

    orig = axis[0].plot(timeseries, color='blue', label='Original')
    mean = axis[0].plot(rolmean, color='red', label='Rolling Mean')
    std = axis[0].plot(rolstd, color='black', label='Rolling Std')

    plot_acf(timeseries.dropna().values, ax=axis[1])
    plot_pacf(timeseries.dropna().values, ax=axis[2])
    # plt.legend(loc='best')

    return axis


def estimate_p_d_q(X):

    print(X.head())
    re = adfuller(X.values)
    print(f'p_value: {re[1]}')

    ax1 = test_stationarity(X)
    ax1[0].set_title('Original series')

    ax2 = test_stationarity(X.diff())
    ax2[0].set_title('1-Differenced series')

    ax3 = test_stationarity(X.diff().diff())
    ax3[0].set_title('2-Differenced series')

    plt.savefig('./acf_pacf.png')
    # plt.show()


def model_ARIMA(y, param=(2, 0, 2), timestep=5):
    warnings.filterwarnings("ignore")

    train = y[:int(0.85*(len(y)))]
    test = y[int(0.85*(len(y))):]

    #his = [x for x in train]
    his = train  # copy train set
    predict = []

    for i in range(0, len(test), timestep):

        print(f'{i*100 / len(test):.3}%')

        step = min(len(test)-i, timestep)

        model = ARIMA(his, order=param).fit(disp=False)

        yhat, se, conf = model.forecast(step)
        predict += list(yhat)

        # after predict 5 days, append the real value of that day to the training set
        his = his.append(test[i:i+step])

    mae = mean_absolute_error(test, predict)
    mse = mean_squared_error(test, predict)
    r2 = r2_score(test, predict)

    print(f'R2: {r2} MAE: {mae}, MSE: {mse}')

    plt.plot(test.values, label='truth')
    plt.plot(predict, label='predict')
    plt.legend(loc='best')

    # plt.savefig('./predict.png')
    plt.show()
    return r2, predict


def test_aic_pdq(y, p=range(1, 8), d=range(0, 13), q=range(0, 13)):

    warnings.filterwarnings("ignore")

    print("Running")
    train = y[:int(0.85*(len(y)))]
    valid = y[int(0.85 * (len(y))):]

    pdq = list(itertools.product(p, d, q))

    print(len(pdq))

    import math
    min_aic = math.inf
    max_r2 = -math.inf

    min_aic_param = max_r2_param = None

    for param in pdq:
        try:
            mod = ARIMA(train, order=param)

            results = mod.fit(disp=False)
            aic = results.aic

            if aic < min_aic:
                min_aic = aic
                min_aic_model = results
                min_aic_param = param

            ytrue = valid.head(5).values
            yhat, se, conf = results.forecast(5)

            r2 = r2_score(ytrue, yhat)

            if r2 > max_r2:
                max_r2 = r2
                max_r2_model = results
                max_r2_param = param

            print(f'ARIMA{param}, AIC:{aic}, R2:{r2}')

        except:
            continue

    return min_aic_param, max_r2_param


if __name__ == '__main__':

    def dateparse(dates): return datetime.strptime(dates, '%d/%m/%Y')
    # data = pd.read_csv('SonTay.csv,
    #                  parse_dates=['date'], index_col='date', date_parser=dateparse)
    data = pd.read_csv('Kontum-daily.csv')

    y = data['discharge']
    # estimate_p_d_q(y)

    #min_aic_param, max_r2_param = test_aic_pdq(y)

    model_ARIMA(y, (1, 0, 1))
