import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression


def univariate_selection(data):
    X = data.iloc[:, [0, 1, 2, 3, 4, 6]]
    y_q = data.iloc[:, 5]
    y_h = data.iloc[:, 7]

    best_feature = SelectKBest(score_func=f_regression, k=6)

    fit_q = best_feature.fit(X, y_q)
    score_q = pd.DataFrame(fit_q.scores_)
    p_val_q = fit_q.pvalues_

    fit_h = best_feature.fit(X, y_h)
    score_h = fit_h.scores_
    p_val_h = fit_h.pvalues_

    dt_cols = pd.DataFrame(X.columns)
    feature_scores = pd.concat([dt_cols, score_q], axis=1)
    feature_scores.columns = ['feature', 'score_q']
    feature_scores['p_val_q'] = p_val_q
    feature_scores['score_h'] = score_h
    feature_scores['p_val_h'] = p_val_h

    print(
        feature_scores
    )  #NOTE: the result show that wind, max-min temp has lowest score, then comes solar_r


def feature_importances(data):
    X = data.iloc[:, [0, 1, 2, 3, 4, 6]]
    y_q = data.iloc[:, 5]
    y_h = data.iloc[:, 7]

    from sklearn.ensemble import ExtraTreesRegressor
    import matplotlib.pyplot as plt

    model = ExtraTreesRegressor()

    model.fit(X, y_q)
    f_importance_q = model.feature_importances_

    model.fit(X, y_h)
    f_importance_h = model.feature_importances_

    fig = plt.figure(figsize=(10, 6))
    fig.add_subplot(121)
    plt.barh(X.columns, f_importance_q)
    plt.title('Q')

    fig.add_subplot(122)
    plt.barh(X.columns, f_importance_h)
    plt.title('H')

    plt.show()  #NOTE: this show similar result compare to univariance method


if __name__ == '__main__':
    data = pd.read_csv('./RawData/Kontum-daily.csv', header=0, index_col=0)
    data = data.drop('time', axis=1)
    #univariate_selection(data)
    feature_importances(data)
