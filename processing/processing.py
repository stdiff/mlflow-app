"""
Store the feature matrices (train, test) and the conversion function

run:
  parameters: table, retrieval_time, logic, processed_time
  metrics: ncol, nrow_train, nrow_test
"""

import click
from typing import List
from datetime import datetime
from pytz import timezone
from configparser import ConfigParser

import mlflow

import pandas as pd
from sklearn.base import TransformerMixin

from lib import enrichment

### load configuration
config = ConfigParser()
config.read("config.ini")
tz = timezone(config["general"]["timezone"])
target = config["data"]["target"]

### mlflow initialization
mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])


class SecToHour(TransformerMixin):
    def __init__(self, unit:str="h"):
        """
        A harmless class
        """
        self.unit = unit

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X / (60*60)


class Processor(TransformerMixin):
    def __init__(self, features:List[str], target:str):
        """
        This class contains the necessary information about data processing.

        :param features: list of feature variables
        :param target:  name of target variable
        """

        self.features = features
        self.target = target

        ## Give a fitted transformer if you need
        self.sec_to_hour = None

    def fit(self, X:pd.DataFrame, y=None):
        """
        The preprocess of the data processing. (e.g. training LabelBinarizer)

        :param X: training data
        :param y: not used
        :return: the instance itself
        """

        self.sec_to_hour = SecToHour()
        self.sec_to_hour.fit(X)

        return self


    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        """
        The main function of data processing. This converts a given data
        into a feature matrix.

        :param X: input data (with or without target variable)
        :return: feature matrix
        """

        cols = self.features.copy()
        ### Note that the given data has no target variable in the production
        if self.target in X.columns:
            cols.append(self.target)

        data = X[cols].copy()

        """
        HERE YOU IMPLEMENT A LOGIC TO CREATE A FEATURE MATRIX
        
        An instance of this class should not be plugged into Pipeline of scikit-learn.
        This is because this method includes feature engineering. In other words,
        Any scaling (that depends on a training data) belongs to Pipeline, not here.        
        """

        data["Time"] = self.sec_to_hour.transform(data["Time"])

        return data


@click.command(help="Convert the tables into a feature matrix and store the convert function.")
@click.option("--table", default=None) ## CSV
@click.option("--retrieval_time", default=None) ## CSV
@click.option("--logic", default="plain")
def processing(table:str, retrieval_time:str, logic:str):
    print("-" * 20 + " load start")

    client = mlflow.tracking.MlflowClient(tracking_uri=config["mlflow"]["tracking_uri"])

    ## Retrieve data from mlflow
    run_uuid, artifact_uri = enrichment.look_up_run_load(client, tz, retrieval_time, table)
    df_train = enrichment.get_artifact(client, run_uuid, artifact_uri, file_name="training_set")
    df_test = enrichment.get_artifact(client, run_uuid, artifact_uri, file_name="test_set")

    ## define the argument of enrichment.store_artifact
    features = ["Time"] + ["V%d" % j for j in range(1, 29)] + ["Amount"]
    data = {"processor": Processor(features=features, target=target)}
    data["training_set"] = data["processor"].fit_transform(df_train)
    data["test_set"] = data["processor"].transform(df_test)

    parameters = {"retrieval_time": retrieval_time,
                  "table": table,
                  "logic": logic,
                  "processed_time": int(datetime.now().timestamp())
                  }
    metrics= {"ncol": data["training_set"].shape[1],
              "nrow_train": data["training_set"].shape[0],
              "nrow_test": data["test_set"].shape[0]
              }

    enrichment.store_artifact(data, experiment="processing", parameters=parameters, metrics=metrics)


if __name__ == "__main__":
    processing()

