# BEST PARAMS {'crit': 'mse', 'n_estimators': 685, 'max_depth': 18, 'max_feat': 'auto'}
import pandas as pd 
import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import ensemble
from xgboost import XGBRegressor
import optuna
import joblib
df = pd.read_csv("../input/cleaned_train.csv")

X = df.drop("count",axis=1).values
y = df["count"].values

scaler = preprocessing.MinMaxScaler()

X = scaler.fit_transform(X)


reg = ensemble.RandomForestRegressor(criterion='mse', n_estimators= 685, max_depth= 18, max_features= 'auto')

reg.fit(X,y)

df2 = pd.read_csv("../input/cleaned_test.csv")
# datetime_col = df["datetime"]


X_test= df2.values
X_test= scaler.transform(X)
joblib.dump(reg,"../model/model.bin")
preds = reg.predict(X_test)

df_pred = pd.DataFrame(preds)



df_pred.to_csv("pred11s.csv",index=False)