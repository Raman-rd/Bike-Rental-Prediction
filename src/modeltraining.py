import pandas as pd 
import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import ensemble
from xgboost import XGBRegressor
import optuna
df = pd.read_csv("../input/cleaned_train.csv")

X = df.drop("count",axis=1).values
y = df["count"].values

scaler = preprocessing.MinMaxScaler()

X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size=0.30,random_state=100)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

kfold = model_selection.KFold(n_splits=10)

reg = ensemble.RandomForestRegressor()
errors=[]
for train_i,test_i in kfold.split(X_train_scaled,y_train):

    train_x , train_y = X_train_scaled[train_i] ,y_train[train_i]

    test_x,test_y = X_train_scaled[test_i] ,y_train[test_i]

    reg.fit(train_x,train_y)

    preds = reg.predict(test_x)

    error= metrics.mean_squared_error(test_y,preds)

    errors.append(error)
print(np.mean(errors))

def objective(trial):

        criterion = trial.suggest_categorical("crit",["mse","mae"])
        n_estimators = trial.suggest_int("n_estimators",50,1000)
        max_depth = trial.suggest_int("max_depth",2,30)
        max_features = trial.suggest_categorical("max_feat",["auto","sqrt","log2"])
        reg2 = ensemble.RandomForestRegressor(criterion=criterion,n_estimators=n_estimators,max_depth=max_depth,max_features=max_features)
        return model_selection.cross_val_score(reg2,X_train_scaled,y_train,scoring="neg_root_mean_squared_error",cv=10,n_jobs=-1).mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective,n_trials=100)

trial = study.best_trial

print("mse {}".format(trial.value))
print("BEST PARAMS {}".format(trial.params))