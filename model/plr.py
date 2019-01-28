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

import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

from lib import enrichment
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
pos_label = int(config["data"]["pos_label"])

cv = int(config["training"]["cv"])
scoring = config["training"]["scoring"]

### mlflow initialization
mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])


from typing import Union, Any, Dict
import numpy as np

def train_the_best_model(X_train:Union[pd.DataFrame,np.ndarray], y_train:Union[pd.Series,np.ndarray],
        X_val:Union[pd.DataFrame,np.ndarray]=None, y_val:Union[pd.Series,np.ndarray]=None, cv:int=5,
        scoring:Any="accuracy", random_state:Any=None, **kwargs) -> Dict[str,Any]:
    """
    trains a model and evaluates the result. This function must be implemented before starting a run.
    Do not forget to commit your source code if you modify it.

    :param X_train: Training set
    :param y_train: Target variable of the training set
    :param X_val: Validation set
    :param y_val: Target variable of the validation set
    :param cv: number of folds in cross-validation
    :param scoring: (usually) passed to GridSearchCV
    :param random_state: (usually) given by the command parameter
    :param kwargs: Anything you need except above.
    :return: dict containing the final model, hyperparameters and the values of performance metrics.
    """

    """
    HERE YOU TRAIN YOUR MATHEMATICAL MODEL AND EVALUATE IT.
    
    - You do not need to use all of the given arguments of the function.
    - You should discussed performance metrics for the project in advance.   
    - The retuen values are used to create arguments for enrichment.store_artifact.
    - Do not forget to commit your modification before executing this script.    
    """

    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.linear_model import LogisticRegression

    ### hyperparameter tuning with grid search (score = auc)
    param = {"plr__C": [0.1, 1, 10],
             "plr__penalty": ["l1", "l2"]#,
             #"plr__class_weight": [{0: 1, 1: w} for w in [1, 10, 50, 100, 200]]
             }
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
    best_results = modeling.cv_results_summary(model).loc[1,:] ## best hyperparameters

    y_score_cw = enrichment.compute_score(model, X_val)
    roc_cw = modeling.ROCCurve(y_val, y_score_cw, pos_label=1)
    best_f1 = roc_cw.get_scores().sort_values(by="f1_score", ascending=False).iloc[0, :]
    best_threshold = best_f1.name ## best threshold
    print("The best threshold:", best_threshold)

    ## retrain with the whole dataset
    print("retrain a model with the best parameters and the whole data set")
    final_model = Pipeline([("scaler", MinMaxScaler()),
                            ("plr", LogisticRegression(**best_params,
                                                    random_state=random_state))])
    final_model.fit(kwargs["data"].drop(target,axis=1), kwargs["data"][target])

    ## Fill the following values
    ## The values must be discussed at the beginning of the project.

    final_model = final_model ## Trained Pipeline or scikit-learn model
    threshold = best_threshold ## float
    algorithm = "LogisticRegression" ## Arbitrary string
    best_params_dict = model.best_params_ ## Just give model.best_params_

    metrics = {"cv_score"      : best_results["mean_test_score"],
               "cv_score_std"  : best_results["std_test_score"],
               "training_score": best_results["mean_train_score"],
               "threshold"     : best_threshold,
               "recall"        : best_f1["recall"],
               "precision"     : best_f1["precision"],
               "f1_score"      : best_f1["f1_score"]
    }

    ## Do not modify the following (without understanding the contents).
    result = {
        "final_model": final_model,
        "threshold": threshold,
        "algorithm" : algorithm,
        "best_params_dict": best_params_dict,
        "metrics" : metrics
     }
    assert all(v is not None for k,v in result.items())
    return result

@click.command(help="")
@click.option("--logic", default=None)
@click.option("--processed_time", default=None)
@click.option("--random_state", default=3)
def train(logic:str, processed_time:str, random_state:int):
    ts_start = int(datetime.now().timestamp())
    client = mlflow.tracking.MlflowClient(tracking_uri=config["mlflow"]["tracking_uri"])

    ## Retrieve the processed data and preprocessor
    run_uuid, artifact_uri, processed_time = enrichment.look_up_run(
        client, "processing", query="plain", run_time=processed_time, tz=tz)
    df = enrichment.get_artifact(client, run_uuid, artifact_uri, file_name="training_set")
    processor = enrichment.get_artifact(client, run_uuid, artifact_uri, file_name="processor")

    df = df.head(10000)

    ## split the data set into a training set and a validation set
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(df.drop(target, axis=1), df[target],
                                                      test_size=0.4, random_state=3)

    ## START RUN
    mlflow.set_experiment("model")
    with mlflow.start_run():
        result = train_the_best_model(X_train, y_train, X_val, y_val, cv=cv, scoring=scoring,
                                      random_state=random_state, data=df)

        model = BinaryModel(processor=processor, model=result["final_model"],
                            threshold=result["threshold"], pos_label=pos_label)

        ## parameters
        parameters = {"algorithm": result["algorithm"],
                      "threshold": result["threshold"],
                      "cv": cv,
                      "scoring": scoring,
                      "random_state": random_state,
                      "trained_time": ts_start,
                      "processed_time": processed_time,
                      "logic": logic}
        parameters = {**parameters, **result["best_params_dict"]}

        enrichment.store_artifact(data={},
                                  parameters=parameters,
                                  metrics=result["metrics"],
                                  model=model)


if __name__ == "__main__":
    print("-" * 20, "modelling start")
    train()
    print("-" * 20, "modelling end")