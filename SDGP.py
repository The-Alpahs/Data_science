import warnings
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib
from pylab import rcParams
import glob

for dataSetName in glob.iglob('Datasets/*.xlsx'):
    matplotlib.rcParams['axes.labelsize'] = 14
    matplotlib.rcParams['xtick.labelsize'] = 12
    matplotlib.rcParams['ytick.labelsize'] = 12
    matplotlib.rcParams['text.color'] = 'G'
    df = pd.read_excel(dataSetName)

    y = df.set_index(['Date'])
    y.head(5)

    y.plot(figsize=(19, 4))
    plt.title(dataSetName)
    plt.show()

    rcParams['figure.figsize'] = 18, 8
    decomposition = sm.tsa.seasonal_decompose(y, model='additive')
    fig = decomposition.plot()
    plt.show()

    mod = sm.tsa.statespace.SARIMAX(y,
                                    order=(0, 0, 1),
                                    seasonal_order=(1, 1, 1, 12),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
    results = mod.fit()
    print(results.summary().tables[1])

    results.plot_diagnostics(figsize=(18, 8))
    plt.show()

    pred = results.get_prediction(start=pd.to_datetime('2019-02-01'), dynamic=False)
    pred_ci = pred.conf_int()
    ax = y['2017':].plot(label='observed')
    pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 4))
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.2)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    plt.legend()
    plt.show()

    y_forecasted = pred.predicted_mean
    y_truth = y['2019-02-01':]
    mse = ((y_forecasted - y_truth) ** 2).mean()
    print('The Mean Squared Error is {}'.format(round(mse, 2)))
    print('The Root Mean Squared Error is {}'.format(round(np.sqrt(mse), 2)))

    pred_uc = results.get_forecast(steps=12)
    pred_ci = pred_uc.conf_int()
    ax = y.plot(label='observed', figsize=(14, 4))
    pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.25)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    plt.legend()
    plt.show()

    y_forecasted = pred.predicted_mean
    y_forecasted.head(12)

    y_truth.head(12)

    pred_ci.head(24)

    print('\nThese are the forecast values of ' + dataSetName + '\n')
    forecast = pred_uc.predicted_mean
    forecast.head(12)
    print(forecast)