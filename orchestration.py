import pandas as pd
import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import mlflow

import xgboost as xgb

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

from prefect import flow, task

@task
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

@task
def add_features(df_train, df_val):
    df_train = read_dataframe(train_path)
    df_val = read_dataframe(val_path)
    
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


@task
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

@task
def train_best_model(X_train, X_val, y_train, y_val, dv):
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
        mlflow.log_metric("rmse", rmse)

        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")

@flow
def main_flow(train_path: str = './train.csv', 
                val_path: str = './val.csv'):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Housing Pricing Experiment")
    # Load
    df_train = read_dataframe(train_path)
    df_val = read_dataframe(val_path)

    # Transform
    X_train, X_val, y_train, y_val, dv = add_features(df_train, df_val).result()

    # Training
    train = xgb.DMatrix(X_train, label=y_train)
    valid = xgb.DMatrix(X_val, label=y_val)
    best = train_model_search(train, valid, y_val)
    train_best_model(X_train, X_val, y_train, y_val, dv, wait_for=best)

main_flow()

#from prefect.deployments import Deployment
#from prefect.orion.schemas.schedules import IntervalSchedule
#from prefect.flow_runners import SubprocessFlowRunner
#from datetime import timedelta

#Deployment(
#    flow=main_flow,
#    name="model_training",
#    # schedule=IntervalSchedule(interval=timedelta(weeks=1)),
#    flow_runner=SubprocessFlowRunner(),
#    tags=["ml"],
#)