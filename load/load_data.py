"""
This script

- registers the original data set as an artifact, and
- splits the data set into a training and a test set and
  register both to the experiment load

Case 1) --retrieval_time is a unix time

Look for the full dataset with the given retrieval_time.
If a run (for the full dataset) exists, then nothing is done.
Otherwise an error is raised.

Case 2) --retrieval_time is a date in ISO format

Look for the full dataset whose retrieval_time lies in the
specified date. If a run exists, then nothing is done.
Otherwise an error is raised.

Case 3) --retrieval_time is not given.

Retrieve the row data and start runs.
"""

from typing import Tuple,Any,Union,Dict

import click
from configparser import ConfigParser
from datetime import datetime
from pytz import timezone

import mlflow
import pandas as pd

from lib import enrichment

### load configuration
config = ConfigParser()
config.read("config.ini")
tz = timezone(config["general"]["timezone"])

### mlflow initialization
mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])

def retrieve_data(data_path:str, test_size:float, retrieval_time:Union[str,int],
                  random_state:Any) -> Tuple[Dict[str,Any], Dict[str,Any], Dict[str,Any]]:
    """
    Load the whole data and split it into a training set and a test set

    :param data_path: path to the data file (in local)
    :param test_size: proportion of the test set to the whole data set
    :param retrieval_time: unix_time
    :param random_state:
    :return: tuple of dicts: data, parameters and metrics
    """

    """
    YOU HAVE TO IMPLEMENT THIS FUNCTION.
    
    - You need to decide/modify the arguments of this function.
    - You do not need to split the data into a training set and a test set.
    """

    from sklearn.model_selection import train_test_split

    df = pd.read_csv(data_path)
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state)


    ## data: A key does not contain its extension (i.e. pkl)
    data = {"training_set": df_train, "test_set": df_test, "full": df}

    ## parameters: Do not forget to add/modify "table"
    parameters = {"table": "transactions", "random_state": random_state, "test_size": test_size}

    ## metrics: the size of the retrieved tables are useful.
    metrics = {"ncol": df.shape[1], "nrow_train": df_train.shape[0],
               "nrow_test": df_test.shape[0], "nrow": df.shape[0]}

    return data, parameters, metrics


@click.command(help="Register an original data set, a training set and a test set.")
@click.option("--table", default=None)
@click.option("--random_state", default=42)
@click.option("--test_size", default=0.4)
@click.option("--data_path", default="data/creditcardfraud.zip")
@click.option("--retrieval_time", default=None)
def load(table, random_state, test_size, data_path, retrieval_time):
    print("-" * 20 + " load start")

    client = mlflow.tracking.MlflowClient(tracking_uri=config["mlflow"]["tracking_uri"])
    mlflow.set_experiment("load")

    if retrieval_time != "None": ## not a good logic
        ## A data is specified
        print("Start to look up the specified data. (retrieval_time = %s)" % retrieval_time)

        run_uuid, artifact_uri, _ = enrichment.look_up_run(client, experiment="load", run_time=retrieval_time,
                                                           query="transactions", tz=tz)
        if run_uuid:
            print("you have already the data")

    else:
        ## No data is specified
        print("Start to store the data as artifacts.")
        retrieval_time = int(datetime.now().timestamp())
        print("retrieval_time:", retrieval_time)
        print("in date:", datetime.now())

        with mlflow.start_run():
            ## Check if the original data can be found
            if enrichment.sha256sum(data_path) != config["data"]["checksum"]:
                raise FileNotFoundError("You specified a wrong file. Please read README.md carefully.")

            data, parameters, metrics = retrieve_data(data_path, retrieval_time=retrieval_time,
                                                      test_size=test_size, random_state=random_state)
            parameters["retrieval_time"] = retrieval_time
            enrichment.store_artifact(data, parameters=parameters, metrics=metrics)


if __name__ == "__main__":
    load()

