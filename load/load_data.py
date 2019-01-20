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

import click
from configparser import ConfigParser
from datetime import datetime
from pytz import timezone

import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split

from lib import enrichment

### load configuration
config = ConfigParser()
config.read("config.ini")
tz = timezone(config["general"]["timezone"])

### mlflow initialization
mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])


@click.command(help="Register an original data set, a training set and a test set.")
@click.option("--random_state", default=42)
@click.option("--test_size", default=0.4)
@click.option("--data_path", default="data/creditcardfraud.zip")
@click.option("--retrieval_time", default=None)
def load(random_state, test_size, data_path, retrieval_time):
    print("-" * 20 + " load start")

    client = mlflow.tracking.MlflowClient(tracking_uri=config["mlflow"]["tracking_uri"])

    if retrieval_time:
        ### A data is specified
        print("Start to look up the specified data.")

        run_uuid, _ = enrichment.look_up_run_load(client, tz, retrieval_time=retrieval_time,
                                                  table="transaction", dataset="full")
        if run_uuid:
            print("you have already the data")
            return

    else:
        ### no data is specified
        print("Start to store the data as artifacts.")

        ### check if the original data can be found
        if enrichment.sha256sum(data_path) != config["data"]["checksum"]:
            raise FileNotFoundError("You specified a wrong file. Please read README.md carefully.")

        retrieval_time = int(datetime.now().timestamp())

        df = pd.read_csv(data_path)
        assert df.shape[0] == 284807

        df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state)

        data = {"training_set": df_train, "test_set": df_test, "full": df}
        parameters = {"table": "transactions", "retrieval_time": retrieval_time, "random_state":random_state,
                      "test_size": test_size}
        metrics = {"ncol": df.shape[1], "nrow_train": df_train.shape[0], "nrow_test": df_test.shape[0],
                   "nrow": df.shape[0]}
        enrichment.store_artifact(data, experiment="load", parameters=parameters, metrics=metrics)


if __name__ == "__main__":
    load()

