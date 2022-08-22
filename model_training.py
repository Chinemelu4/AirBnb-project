from audioop import add
import pandas as pd
import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import xgboost as xgb

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("House Price prediction")

def read_dataframe(filename):
    
    df = pd.read_csv(filename)
    cat_col=['host_identity_verified','neighbourhood group',
        'instant_bookable', 'cancellation_policy', 'room type']
    num_col=['Construction year', 'service fee', 'minimum nights',
       'number of reviews']
    target=['price']
    
    df=df[cat_col+num_col+target]
    
    fees_col=['price','service fee']
    for col in fees_col:
        df[col]=df[col].apply (lambda x:int(x.split("$")[1].rstrip().replace(",","")))
        
    for col in df[num_col]:
        df[col]=df[col].astype(int)
    
    return df

def add_features(train_path="./train.csv",
                 val_path="./val.csv"):
    df_train = read_dataframe(train_path)
    df_val = read_dataframe(val_path)

    print(len(df_train))
    print(len(df_val))
    
    cat_col=['host_identity_verified','neighbourhood group',
        'instant_bookable', 'cancellation_policy', 'room type']
    num_col=['Construction year', 'service fee', 'minimum nights',
       'number of reviews']
    
    
    dv = DictVectorizer()
 
    train_dicts = df_train[cat_col+num_col].to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)
    
    
    val_dicts = df_val[cat_col+num_col].to_dict(orient='records')
    X_val = dv.transform(val_dicts)
    
    target='price'
    
    y_train = df_train[target].values
    y_val = df_val[target].values

    return X_train, X_val, y_train, y_val, dv
    

def train_model_search(train, valid, y_val):
    def objective(params):
        with mlflow.start_run():
            mlflow.set_tag("model", "xgboost")
            mlflow.log_params(params)
            booster = xgb.train(
                params=params,
                dtrain=train,
                num_boost_round=100,
                evals=[(valid, 'validation')],
                early_stopping_rounds=50
            )
            y_pred = booster.predict(valid)
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            mlflow.log_metric("rmse", rmse)

        return {'loss': rmse, 'status': STATUS_OK}

    search_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
        'learning_rate': hp.loguniform('learning_rate', -3, 0),
        'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
        'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
        'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
        'objective': 'reg:linear',
        'seed': 42
    }

    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=1,
        trials=Trials()
    )
    return

def train_best_model(train, valid, y_val, dv):
    with mlflow.start_run():
        
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        best_params = {
            'learning_rate': 0.22568238453428718,
            'max_depth': 12,
            'min_child_weight': 0.6935354153509427,
            'objective': 'reg:linear',
            'reg_alpha': 0.030233807650675962,
            'reg_lambda': 0.0035810637126474035,
            'seed': 42
        }

        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=100,
            evals=[(valid, 'validation')],
            early_stopping_rounds=50
        )

        y_pred = booster.predict(valid)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        Accuracy=r2_score(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("Accuracy", Accuracy)

        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")

if __name__ == "__main__":
    X_train, X_val, y_train, y_val, dv = add_features()
    train = xgb.DMatrix(X_train, label=y_train)
    valid = xgb.DMatrix(X_val, label=y_val)
    train_model_search(train, valid, y_val)
    train_best_model(train, valid, y_val, dv)