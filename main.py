"""
Execute the whole data pipeline. More precisely we execute the whole steps so that we avoid any redundant run.

How to work: You have to give
- Which data you use (retrieval_time and table)
- Which logic you use for data processing (logic)
- Which algorithm you use for training a model (algorithm)

Then:

1. Look up a load-run you want to use. If no run is found, we start a load-run.
   So we have a specific load-run now.
2. Look up a processing-run which corresponds to the load-run.
   If no run is found, then we start a new processing-run with the given logic.
   If several runs are found, we take the newest one. Namely no new processing-run is started.
   So we have a specific processing-run.
3. Look up a model-run which corresponds to the processing-run.
   If no run is found, then we start a new model-run with the given algorithm.
   If several runs are found, we take the newest one and no new model-run is started.

"""

import click
from pytz import timezone
from configparser import ConfigParser

import mlflow
from lib import enrichment
from lib.enrichment import RunNotFoundError

### load configuration
config = ConfigParser()
config.read("config.ini")
tz = timezone(config["general"]["timezone"])
target = config["data"]["target"]

### mlflow initialization
mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])


@click.command(help="Execute the whole data pipeline")
@click.option("--retrieval_time", default=None)
@click.option("--table", default=None)
@click.option("--logic", default=None)
@click.option("--algorithm", default=None)
def main(retrieval_time:str, table:str, logic:str, algorithm:str):

    client = mlflow.tracking.MlflowClient(tracking_uri=config["mlflow"]["tracking_uri"])

    ## Step 1: load
    print("-- Checking a run in load")
    try:
        run_uuid, _, retrieval_time_ts = enrichment.look_up_run(
            client, experiment="load", query=table, run_time=retrieval_time, tz=tz)
        print("A run is found.")

    except RunNotFoundError:
        print("We start a new run (load)")
        experiment_id = client.get_experiment_by_name("load").experiment_id
        new_run = mlflow.run(".", entry_point="load", experiment_id=experiment_id,
                             parameters={"retrieval_time":None})
        run = client.get_run(new_run.run_id)
        ## retrieval_time
        param_dict = {param.key: param.value for param in run.data.params}
        retrieval_time_ts = param_dict["retrieval_time"]

    ## Step 2: processing
    print("-- Checking a run in processing")
    try:
        processed_time = enrichment.get_latest_run_time(client, source_experiment="load",
                                                        source_run_time=retrieval_time_ts,
                                                        source_query=table, target_query=logic)
        print("A run is found.")

    except RunNotFoundError:
        print("We start a new run (processing)")
        experiment_id = client.get_experiment_by_name("processing").experiment_id
        new_run = mlflow.run(".", entry_point="processing", experiment_id=experiment_id, use_conda=False,
                             parameters={"table":table, "retrieval_time":retrieval_time})
        run = client.get_run(new_run.run_id)
        ## processed_time
        param_dict = {param.key: param.value for param in run.data.params}
        processed_time = param_dict["processed_time"]

        if logic != param_dict["logic"]:
            raise ValueError("The run has a logic which is different from the logic you gave.")

    ## Step 3: model
    print("-- Checking a run in model")
    try:
        trained_time = enrichment.get_latest_run_time(client, source_experiment="processing",
                                                      source_run_time=processed_time,
                                                      source_query=logic, target_query=algorithm)
        print("A run is found.")

    except RunNotFoundError:
        print("We start a new run (modell)")
        experiment_id = client.get_experiment_by_name("model").experiment_id
        new_run = mlflow.run(".", entry_point="model", experiment_id=experiment_id, use_conda=False,
                             parameters={"logic":logic, "processed_time":processed_time, "random_state":3})
        run = client.get_run(new_run.run_id)

        ## processed_time
        param_dict = {param.key: param.value for param in run.data.params}
        trained_time = param_dict["trained_time"]

        if algorithm != param_dict["algorithm"]:
            raise ValueError("The run has an algorithm which is different from the algorithm you gave.")

    mlflow.log_param("retrieval_time", retrieval_time_ts) ## overwrite
    mlflow.log_metric("processed_time", processed_time)
    mlflow.log_metric("trained_time", trained_time)


if __name__ == "__main__":
    print("-"*20, "workflow start")
    main()
    print("-"*20, "workflow end")