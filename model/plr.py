"""
Train a mathematical model

run:
parameters:
metrics:
"""

import copy
import click
from datetime import datetime
from pytz import timezone
from configparser import ConfigParser

import mlflow
from mlflow import pyfunc

import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

from lib import enrichment
from lib import MlModel
from model_enhanced import BinaryModel

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

### load configuration
config = ConfigParser()
config.read("config.ini")
tz = timezone(config["general"]["timezone"])
target = config["data"]["target"]

### mlflow initialization
mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])

@click.command(help="")
@click.option("--logic", default=None)
@click.option("--processed_time", default=None)
@click.option("--random_state", default=3)
def train(logic:str, processed_time:str, random_state:int):
    ts_start = int(datetime.now().timestamp())
    client = mlflow.tracking.MlflowClient(tracking_uri=config["mlflow"]["tracking_uri"])

    run_uuid, artifact_uri = enrichment.look_up_run(client, "processing", query="plain", run_time="2019-01-27", tz=tz)
    df = enrichment.get_artifact(client, run_uuid, artifact_uri, file_name="training_set")
    processor = enrichment.get_artifact(client, run_uuid, artifact_uri, file_name="processor")

    df = df.head(10000)

    ### split the data set into a training set and a validation set
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(df.drop(target, axis=1), df[target],
                                                      test_size=0.4, random_state=3)

    ### hyperparameter tuning with grid search (score = auc)
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.linear_model import LogisticRegression

    param = {"plr__C": [0.1, 1, 10],
             "plr__penalty": ["l1", "l2"]#,
             #"plr__class_weight": [{0: 1, 1: w} for w in [1, 10, 50, 100, 200]]
             }
    cv = 3
    scoring = "roc_auc"
    pipeline = Pipeline([("scaler", MinMaxScaler()),
                         ("plr", LogisticRegression(random_state=random_state))])
    model = GridSearchCV(pipeline, param, cv=cv, scoring=scoring, verbose=1, n_jobs=3)
    model.fit(X_train, y_train)

    print("%s-fold CV, scoring = %s" % (cv,scoring))
    print("best_params:", model.best_params_)
    print("best_score:", model.best_score_)
    best_params = {k[5:]:v for k,v in model.best_params_.items()}

    ## find the best threshold by using the validation set
    print("Looking for the best threshold")

    from lib import modeling
    best_results = modeling.cv_results_summary(model).loc[1,:]

    y_score_cw = enrichment.compute_score(model, X_val)
    roc_cw = modeling.ROCCurve(y_val, y_score_cw, pos_label=1)
    best_f1 = roc_cw.get_scores().sort_values(by="f1_score", ascending=False).iloc[0, :]
    best_threshold = best_f1.name
    print("The best threshold:", best_threshold)

    ## retrain with the whole dataset
    print("retrain a model with the best parameters and the whole data set")
    final_model = Pipeline([("scaler", MinMaxScaler()),
                         ("plr", LogisticRegression(**best_params,
                                                    random_state=random_state))])
    final_model.fit(df.drop(target,axis=1), df[target])
    ts_end = int(datetime.now().timestamp())

    ## data
    data = {
        "ml_model": BinaryModel(processor=processor, model=final_model, threshold=best_threshold, pos_label=1)
    }

    ## parameters
    parameters = copy.deepcopy(model.best_params_)
    parameters["algorithm"]      = "LogisticRegression"
    parameters["trained_time"]   = ts_start
    parameters["random_state"]   = random_state
    parameters["cv"]             = cv
    parameters["scoring"]        = scoring
    parameters["processed_time"] = processed_time
    parameters["logic"]          = logic

    ## metric
    metrics = {"cv_score"      : best_results["mean_test_score"],
               "cv_score_std"  : best_results["std_test_score"],
               "training_score": best_results["mean_train_score"],
               "threshold"     : best_threshold,
               "recall"        : best_f1["recall"],
               "precision"     : best_f1["precision"],
               "f1_score"      : best_f1["f1_score"],
               "duration"      : int(ts_end - ts_start)
    }

    enrichment.store_artifact(data, experiment="model", parameters=parameters, metrics=metrics,
                              model=data["ml_model"])


if __name__ == "__main__":
    print("-" * 20, "modelling start")
    train()
    print("-" * 20, "modelling end")