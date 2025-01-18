import pandas as pd
from sklearn.linear_model import LinearRegression
from joblib import dump


df_train = pd.read_csv('train/result_train.csv')


X_train = df_train.drop('target', axis=1)
y_train = df_train['target']


model = LinearRegression()

model.fit(X_train, y_train)

dump(model, 'model.joblib')

