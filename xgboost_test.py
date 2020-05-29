import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost.sklearn import XGBRegressor
import matplotlib.pyplot as plt


def data_generate(raw_dat, pre_range=5):
    raw_dat.iloc[:, 3] = raw_dat.iloc[:, 3] / 35.315

    for i in range(pre_range):
        raw_dat[f'pre_{i+1}'] = raw_dat.iloc[:, 3].shift(periods=i + 1, axis=0)

    raw_dat = raw_dat.dropna(axis=0, how='any')
    raw_dat.index = pd.to_datetime(raw_dat['datetime'])
    #raw_dat['dayofweek'] = raw_dat.index.dayofweek
    raw_dat['month'] = raw_dat.index.month
    raw_dat['year'] = raw_dat.index.year
    raw_dat['dayofyear'] = raw_dat.index.dayofyear
    raw_dat['dayofmonth'] = raw_dat.index.day
    raw_dat['weekofyear'] = raw_dat.index.weekofyear
    #raw_dat['hour'] = raw_dat.index.hour
    raw_dat = raw_dat.drop(['status', 'datetime'], axis=1)
    print(raw_dat.head(10))
    raw_dat.to_csv('./ProcessedData/urqua_daily.csv', index=True)

    # cols = [
    #     'wind', 'min_temp', 'max_temp', 'solar_r', 'humidity', 'discharge',
    #     'rain', 'water_level'
    # ]  #col names
    # for i in cols:
    #     for j in range(pre_range):
    #         raw_dat[f'{i}_{j}'] = raw_dat[i].shift(periods=j + 1, axis=0)

    # raw_dat = raw_dat.dropna(axis=0)

    # print(raw_dat.iloc[:10, 8:])
    # raw_dat.to_csv('./ProcessedData/Kontum-daily.csv')

    return raw_dat


def feature_importances_xgboost(dataset,
                                cols_feature,
                                train_per=0.2,
                                valid_per=0.2):
    #dataset = dataset.to_numpy()
    X = dataset[:, 3:]
    Y = dataset[:, 2]
    # split data into train and test sets
    train_size = int(len(dataset) * train_per)
    valid_size = int(len(dataset) * valid_per)
    X_train = X[:train_size]
    y_train = Y[:train_size]

    print(X_train.shape)
    print(X_train[:5])

    X_valid = X[train_size:train_size + valid_size]
    y_valid = Y[train_size:train_size + valid_size]

    X_test = X[train_size + valid_size:]
    y_test = Y[train_size + valid_size:]

    model = XGBRegressor(objective='reg:tweedie',
                         booster='gbtree',
                         max_depth=10,
                         n_estimators=1000,
                         colsample_bytree=0.8,
                         subsample=0.8,
                         eta=0.3,
                         seed=2)

    model.fit(X_train,
              y_train,
              eval_metric="mae",
              eval_set=[(X_valid, y_valid)],
              verbose=False,
              early_stopping_rounds=15)
    # plot feature importance
    # from xgboost import plot_importance
    # plot_importance(model)
    for col, score in zip(cols_feature, model.feature_importances_):
        print(col, score)
    feature_importances = list(zip(cols_feature, model.feature_importances_))
    feature_importances = sorted(feature_importances, key=lambda x: x[1])

    plt.bar(range(3, dataset.shape[1]),
            model.feature_importances_,
            orientation='vertical')
    plt.show()
    # make predictions for test data and evaluate
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    plt.plot(y_test, label='truth')
    plt.plot(predictions, label='predict')
    plt.legend()
    plt.show()

    print(f'MAE: {mae} MSE: {mse} R2: {r2}')


if __name__ == '__main__':
    # data = pd.read_csv('./RawData/urqua_river.csv',header=None,sep='\t',names=['year','month','discharge'])
    # data = data.dropna(axis=0)
    # #dat = data_generate(data)
    # dat = pd.read_csv('./ProcessedData/urqua_river.csv',header=0).dropna(axis=0)

    data = pd.read_csv('./ProcessedData/urqua_daily.csv',
                       header=0,
                       index_col=0)
    #import numpy as np
    #data['discharge'] = data['discharge'].to_numeric()
    # data['discharge'] = pd.to_numeric(data['discharge'], errors='coerce')
    # dat = data_generate(data, pre_range=5)
    # print(data.dtypes)
    # print(data.head())

    dat = data.to_numpy()

    feature_importances_xgboost(dat,
                                cols_feature=[
                                    'pre_1', 'pre_2', 'pre_3', 'pre_4',
                                    'pre_5', 'month', 'year', 'dayofyear',
                                    'dayofmonth', 'weekofyear'
                                ],
                                train_per=0.6,
                                valid_per=0.3)

    #NOTE: XGBoost bring result MAE: 18.04024713150927 MSE: 4213.707219770521 R2: 0.6476693828870468
    #NOTE:  For daily dataset urqua MAE: 42.545505091653354 MSE: 21377.726272764718 R2: 0.7224047581105089
