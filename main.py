# [START app]
from flask import Flask, jsonify, request
app = Flask(__name__)

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from scipy.optimize import linprog

from sklearn import linear_model as lm
from sklearn import preprocessing as pre
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

import quandl
quandl.ApiConfig.api_key = "_A6U7ooxne5KHcB6cphy"

# Load info for sp500
sp500_general = pd.read_csv('constituents.csv')
sp500_risks = pd.read_csv('company_risk.csv')
sp500 = sp500_general.merge(sp500_risks, how='inner', on='Symbol')

# Normalize the Risk Column and remove too-risky values
min_max_scaler = pre.MinMaxScaler()
sp500['Risk'] = min_max_scaler.fit_transform(sp500['Risk'].values.reshape(-1, 1)).flatten()

top = sp500[sp500['Risk'] <= 0.9].iloc[0:100]['Symbol'].values.tolist()


data = pd.read_csv('wiki_prices.csv', usecols=['ticker', 'close', 'date'])
data = data[data['ticker'].isin(top) & (data['date'] >= '2017-01-01')]

codebook = dict(enumerate(data['ticker'].astype('category').cat.categories))




data['ticker'] = data['ticker'].astype('category').cat.codes
data['day'] = (data['date'] - data['date'].min()).dt.days
data = data.drop(['date'], axis=1)
data['last_day_close'] = data.groupby(['ticker'])['close'].shift()
data['last_day_diff'] = data.groupby(['ticker'])['last_day_close'].diff()
data = data.dropna()
LAST_DAY = data['day'].max()

def ttsplit(df, train_size):
    X = df.drop(['close'], axis = 1)
    y = df['close']
    return train_test_split(X, y, train_size=train_size, random_state=42)


################################### MODELS #######################################

def build_random_forest(df):
    mean_error = []
    sizes = [1/4]
    for size in sizes:
        xtr, xts, ytr, yts = ttsplit(df, size)

        mdl = RandomForestRegressor(n_estimators=1000, n_jobs=-1, random_state=0)
        mdl.fit(xtr, ytr)

        p = mdl.predict(xts)

        error = mean_squared_error(yts, p)
        print('RMSE Error: %.5f' % (error))
        mean_error.append(error)
    print('Mean Error = %.5f' % np.mean(mean_error))
    return mdl

def build_linear_regressor(df):
    X_train, X_test, y_train, y_test = ttsplit(df, 0.25)

    # Fit and predict
    model = lm.LinearRegression(fit_intercept=True)
    model.fit(X_train, y_train)
    y_predicted = model.predict(X_test)

    print(f'The validation RMSE for this model is '
          f'{round(mean_squared_error(y_test, y_predicted), 2)}.')

    return model

def build_elastic_net_predictor(df):
    scaler = pre.StandardScaler()
    X_train, X_test, y_train, y_test = ttsplit(df, 0.25)

    l1_ratios = np.arange(0, 1.1, .1)
    alphas = np.arange(0.1, 200.1, .1)
    model = lm.ElasticNetCV(l1_ratio=l1_ratios,
                            alphas=alphas,
                            cv=5,
                            fit_intercept=True,
                            max_iter=1000)

    # Fit and predict
    model.fit(scaler.fit_transform(X_train), y_train)
    y_predicted = model.predict(scaler.fit_transform(X_test))

    print(f'The validation RMSE for this model with '
          f'alpha={round(float(model.alpha_), 2)} is '
          f'{round(mean_squared_error(y_test, y_predicted), 2)}.')

    return model

def build_lstm_network(df):
    X_train, X_test, y_train, y_test = ttsplit(df, 0.25)

    X_train = X_train.as_matrix().reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.as_matrix().reshape((X_test.shape[0], 1, X_test.shape[1]))

    model = Sequential()
    model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')

    history = model.fit(X_train,
                        y_train,
                        epochs=1000,
                        batch_size=20,
                        validation_data=(X_test, y_test),
                        verbose=2,
                        shuffle=False)

    # print('Test RMSE: %.3f' % np.sqrt(mean_squared_error(y_test, model.predict(X_test))))
    return model

################################### PREDICTION #############################################

def predict_future_stock_values(mdl, source_df, days_out):
    abs_day = LAST_DAY + days_out
    # Warning: Columns must be ordered properly for predictor to work!
    tickers = source_df['ticker'].unique()
    days = np.arange(LAST_DAY + 1, abs_day + 1, 1)

    x = source_df.copy()
    for d in days:
        # Construct a dataframe for the next day, borrowing appropriate values.
        i = x[x['day'] == x['day'].max()]
        i['day'] += 1
        i['last_day_diff'] = i['close'] - i['last_day_close']
        i['last_day_close'] = i['close']
        i = i.drop(['close'], axis=1)

        i = i[['ticker', 'day', 'last_day_close', 'last_day_diff']]

        # Predict new close values
        y = mdl.predict(i.as_matrix().reshape((i.shape[0], 1, i.shape[1])))
        i['close'] = pd.Series(y.reshape(y.shape[0]), index=i.index)

        x = x.append(i).sort_values(['ticker', 'day'], ascending=[True, True])

    x = x[['ticker', 'close', 'day', 'last_day_close', 'last_day_diff']].reset_index(drop=True)
    return x[x['day'] >= LAST_DAY]

# Term is in number of days from current!
def suggest_strategy(df, principal, target, term, model=None):
    current_day = df['day'].max()
    if not model:
        model = build_lstm_network(df)
    extended_data = predict_future_stock_values(model, df, term)

    period = extended_data[(extended_data['day'] >= current_day) & (extended_data['day'] <= current_day + term)]
    stocks_with_current_price = period.groupby('ticker').agg({'close': 'first'})
    candidate_stocks = stocks_with_current_price[stocks_with_current_price['close'] <= 2 * principal].index.values
    candidate_stock_data = period.loc[period['ticker'].isin(candidate_stocks)]

    c = []
    for cs in candidate_stocks:
        max_close = candidate_stock_data[candidate_stock_data['ticker'] == cs]['close'].max()
        c.append(max_close)

    c = np.array(c)
    A_ub = np.array([np.array(stocks_with_current_price.loc[candidate_stocks]['close']), c])
    b_ub = np.array([principal, target])

    optimize_result = linprog(-c, A_ub, b_ub)
    stock_amounts = {codebook[idx]: amnt for
                         idx, amnt in enumerate(np.round(optimize_result.x, decimals=1)) if amnt > 0}

    return {
        'total': -optimize_result.fun,
        'recommendations': stock_amounts
    }

# Example usage of creating a model and suggesting a strategy.
# lstm = build_lstm_network(data)
# suggest_strategy(data, 200, 222, 100, model=lstm)


@app.route('/advise')
def get_recommendations():
    return jsonify(
        suggest_strategy(
            data, float(request.args.get('principle')),
            float(request.args.get('target')), int(request.args.get('term'))))


@app.route('/')
def hello():
    return 'Hello!'

if __name__ == '__main__':
    app.run()

# [END app]
