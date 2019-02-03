"""
Store the feature matrices (train, test) and the conversion function

run:
  parameters: table, retrieval_time, logic, processed_time
  metrics: ncol, nrow_train, nrow_test
"""

import sys
print("You are working on", sys.version)
for p in sys.path:
    print(p)

import click
from typing import Callable, Tuple
from datetime import datetime
from pytz import timezone
from configparser import ConfigParser

import mlflow

import pandas as pd
from lib import enrichment

### load configuration
config = ConfigParser()
config.read("config.ini")
tz = timezone(config["general"]["timezone"])
target = config["data"]["target"]

### mlflow initialization
mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])


def generate_processor(X:pd.DataFrame, target:str)-> Tuple[Callable[[pd.DataFrame], pd.DataFrame],str]:
    """
    generate a preprocessor

    :param X: DataFrame (quasi-raw data)
    :param target: name of the target variable
    :return: (function converting a raw data into a feature matrix, name of logic)
    """

    """
    HERE YOU TRAIN SOME PREPROCESSORS WHICH REQUIRE TRAINING
    
    A transformer such as LabelBinarizer must be trained here.
    """

    ## do not forget to give a name to your logic
    logic = "plain"

    from sklearn.preprocessing import LabelBinarizer

    def to_day(t):
        return "1st_day" if t <= 60*60*24 else "2nd_day"

    X["Date"] = X["Time"].apply(to_day)
    lb = LabelBinarizer()
    lb.fit(X["Date"])

    def preprocessor(X: pd.DataFrame) -> pd.DataFrame:
        """
        convert a row data (DataFrame) into a feature matrics

        :param X: DataFrame (raw data)
        :return: DataFrame (feature matrix)
        """

        """
        HERE YOU IMPLEMENT A LOGIC TO CREATE A FEATURE MATRIX

        - An instance of this class should not be plugged into Pipeline of scikit-learn.
          This is because this method includes feature engineering. In other words,
          Any scaling (that depends on a training data) belongs to Pipeline, not here.
        - A transformer requiring a training should be trained outside this definition.
        - A raw data can contain additional columns because of a certain purpose. We have
          to remove here. 
        - A raw data contain or does not contain the target variable. In production there
          is of course no target variable, but there is the target variable when training
          the model.  
        """

        ## The projection such as the following is normally necessary.
        cols = ["Time"] + ["V%d" % j for j in range(1, 29)] + ["Amount"]
        if target in X.columns:
            cols.append(target)
        X = X[cols].copy()

        ## Implement your logic.
        X["Date"] = lb.transform(X["Time"].apply(to_day))
        X.drop("Time", axis=1, inplace=True)

        return X

    return preprocessor, logic

@click.command(help="Convert the tables into a feature matrix and store the convert function.")
@click.option("--table", default=None) ## CSV
@click.option("--retrieval_time", default=None) ## CSV
def processing(table:str, retrieval_time:str):
    ts_start = int(datetime.now().timestamp())
    client = mlflow.tracking.MlflowClient(tracking_uri=config["mlflow"]["tracking_uri"])

    ### Retrieve data from mlflow
    run_uuid, artifact_uri, retrieval_time = enrichment.look_up_run(
        client, experiment="load", query=table, run_time=retrieval_time, tz=tz)
    df_train = enrichment.get_artifact(client, run_uuid, artifact_uri, file_name="training_set")
    df_test = enrichment.get_artifact(client, run_uuid, artifact_uri, file_name="test_set")

    mlflow.set_experiment("model")
    with mlflow.start_run():
        preprocessor, logic = generate_processor(df_train, target)

        ## define the argument of enrichment.store_artifact
        data = { "processor": preprocessor,
                 "training_set": preprocessor(df_train),
                 "test_set": preprocessor(df_test)
        }
        parameters = {"retrieval_time": retrieval_time,
                      "table": table,
                      "logic": logic,
                      "processed_time": ts_start
        }
        metrics = {"ncol": data["training_set"].shape[1],
                   "nrow_train": data["training_set"].shape[0],
                   "nrow_test" : data["test_set"].shape[0]
        }
        enrichment.store_artifact(data=data, parameters=parameters, metrics=metrics)

if __name__ == "__main__":
    print("-" * 20, "processing start")
    processing()
    print("-" * 20, "processing end")
