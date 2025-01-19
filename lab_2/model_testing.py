from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
from joblib import load


model = load('model.joblib')
df_test = pd.read_csv('test/result_test.csv')

y_test = df_test["target"]

X_test = df_test.drop("target", axis=1)


predict = pd.DataFrame(model.predict(X_test))

print(f"--- Metrics:"
      f"--- MAE - {mean_absolute_error(y_test, predict)}"
      f"--- MSE - {mean_squared_error(y_test, predict)}"
      f"--- RMSE - {mean_squared_error(y_test, predict)**0.5}"
      f"--- r2 - {r2_score(y_test, predict, multioutput='variance_weighted')}")

