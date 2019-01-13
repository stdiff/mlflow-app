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
import tempfile
import shutil
from configparser import ConfigParser
from pathlib import Path
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

### TODO: logger (do we need it?)
# import logging
# import sys
# logging.basicConfig(stream=sys.stdout,
#                     level=print(),
#                     format='%(asctime)s %(message)s',
#                     datefmt='%Y-%m-%d %H:%M:%S') ### Logger


def store_data(data:pd.DataFrame, experiment:str, table:str, retrieval_time:int, dataset:str,
               size:float=None, random_state=None) -> bool:
    """
    Store the given DataFrame as an artifact .

    :param data:
    :param experiment:
    :param table:
    :param retrieval_time:
    :param dataset:
    :param size:
    :param random_state:
    :return:
    """

    if experiment not in ["load","processing","model"]:
        raise enrichment.ExperimentNotFoundError("You gave a wrong experiment: %s" % experiment)

    mlflow.set_experiment(experiment)
    artifact_path = "data"
    file_name = "dataset_%s_%s.pkl" % (table, dataset)

    with mlflow.start_run():
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            target_file = tmp_dir.joinpath(file_name)
            print("Saving %s ..." % file_name)
            data.to_pickle(target_file)

            mlflow.log_param("table", table)
            mlflow.log_param("retrieval_time", retrieval_time)
            mlflow.log_param("dataset", dataset)
            mlflow.log_param("size", size)
            if random_state is not None:
                mlflow.log_param("random_state", random_state)

            mlflow.log_metric("nrow", data.shape[0])
            mlflow.log_metric("ncol", data.shape[1])
            mlflow.log_artifact(str(target_file), artifact_path)

        except:
            raise Exception("Failed to save the data. (table=%s, dataset=%s)" % (table,dataset))

        finally:
            shutil.rmtree(str(tmp_dir))

    return True


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

        store_data(df, experiment="load", table="transaction", retrieval_time=retrieval_time,
                   dataset="full", size=1)
        store_data(df_train, experiment="load", table="transaction", retrieval_time=retrieval_time,
                   dataset="train", size=1-test_size, random_state=random_state)
        store_data(df_test, experiment="load", table="transaction", retrieval_time=retrieval_time,
                   dataset="test", size=test_size, random_state=random_state)


if __name__ == "__main__":
    load()

