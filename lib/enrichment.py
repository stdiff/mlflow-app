"""
Helper functions to get along with MLflow
"""

from typing import Union, Tuple, Any
import os
import re
import shutil
import hashlib
import tempfile
from pathlib import Path
from shutil import rmtree
from datetime import datetime
from typing import Dict
import dill as pickle
#import pickle

import mlflow
import mlflow.sklearn
import pandas as pd

"""
NOTE: The rule of the names should be decided by team at first and not be changed.    
"""
exp_to_key = {"load": "table", "processing": "logic", "model": "algorithm"}
exp_to_time = {"load": "retrieval_time", "processing": "processed_time", "model": "trained_time"}

class ExperimentNotFoundError(Exception):
    pass


class RunNotFoundError(Exception):
    pass


def sha256sum(path:str) -> str:
    """
    Compute sha256 checksum

    :param path: path to the file (in local)
    :return: sha256 checksum
    """
    if os.path.exists(path):
        with open(path,"rb") as fo:
            check_sum = hashlib.sha256(fo.read())
        return check_sum.hexdigest()
    else:
        return ""


def compute_score(model, X:pd.DataFrame, pos_label:Union[str,float]=1) -> pd.Series:
    col = list(model.classes_).index(pos_label)
    return pd.Series(model.predict_proba(X)[:, col], name="score", index=X.index)


def store_artifact(data:Dict[str, Any],
                   parameters:Dict[str, Any], metrics:Dict[str, Any],
                   model:Any=None):
    """
    Start a run and store the given DataFrame as an artifact.

    :param data: file_name (without .pkl): Python Object
    :param parameters: parameters for a run
    :param metrics: metrics for a run
    :param model: trained model
    """

    tmp_dir = Path(tempfile.mkdtemp())
    try:
        for artifact_path, obj in data.items():
            file_name = "%s.pkl" % artifact_path
            target_file = tmp_dir.joinpath(file_name)
            print("Saving %s ..." % file_name)
            with target_file.open("wb") as fo:
                pickle.dump(obj, fo)
            mlflow.log_artifact(str(target_file), "data")

        if model is not None:
            mlflow.sklearn.log_model(model, "model")

        for key in sorted(parameters.keys()):
            val = parameters[key]
            mlflow.log_param(key,val)

        for key in sorted(metrics.keys()):
            val = metrics[key]
            mlflow.log_metric(key,val)

    except:
        raise Exception("Failed to save the data.")

    finally:
        shutil.rmtree(str(tmp_dir))


def look_up_run(client:mlflow.tracking.MlflowClient, experiment:str,
                query:str, run_time:Union[int,str], tz=None) -> Tuple[str,str,str]:
    """
    find uuid of a run with the given criteria.
    If run_time is a date in ISO 8601 format (i.e. YYYY-MM-DD),
    then the newest run of the given date is returned.
    If no run is found, an error is raised.

    :param client: mlflow.tracking.MlflowClient instance
    :param experiment: "load", "processing" or "model"
    :param query: value of "table", "logic" or "algorithm"
    :param run_time: value of "retrieval_time", "processed_time" or "trained_time" (or ISO date)
    :param tz: pytz timezone (needed only if run_time is in ISO date)
    :return: (uuid of the found run, artifact_uri of the run, retrieval/processed/trained_time)
    """

    try:
        query_dict = {exp_to_key[experiment]: query, exp_to_time[experiment]: str(run_time)}
        experiment_id = client.get_experiment_by_name(experiment).experiment_id
    except KeyError:
        raise ExperimentNotFoundError("The experiment '%s' cannot be found." % experiment)
    except AttributeError:
        raise ExperimentNotFoundError("The experiment '%s' cannot be found." % experiment)

    print("Searching a run in experiment '%s'" % experiment)

    if re.match(r"\d+$", str(run_time)):
        ### retrieval_time is a unixtime
        print("Looking for the exact dataset (run_time=%s, query=%s)" % (run_time,query))

        for run_info in client.list_run_infos(experiment_id):
            run = client.get_run(run_info.run_uuid)
            param_dict = {param.key: param.value for param in run.data.params}

            try:
                param_run_time = param_dict[exp_to_time[experiment]]
            except KeyError:
                raise RunNotFoundError

            print(exp_to_time[experiment],param_run_time)

            if set(query_dict.keys()).issubset(param_dict.keys()):
                if all([query_dict[k] == param_dict[k] for k in query_dict.keys()]):
                    return run_info.run_uuid, run_info.artifact_uri, query_dict[exp_to_time[experiment]]

        raise RunNotFoundError("There is no run with the given condition.")

    elif run_time and isinstance(run_time, str):
        ### retrieval_time is a date in YYYY-MM-DD
        print("Looking for the newest dataset %s on %s" % (query,run_time))
        if tz is None:
            raise ValueError("'tz' object is None.")

        try:
            d = datetime.strptime(run_time, "%Y-%m-%d")
        except ValueError as e:
            print("run_time is not in the ISO format:", run_time)
            raise RunNotFoundError

        query_date = tz.localize(datetime(year=d.year, month=d.month, day=d.day)).date()

        max_param_time = 0
        found_run_uuid = ""
        found_artifact_uri = ""

        for run_info in client.list_run_infos(experiment_id):
            run = client.get_run(run_info.run_uuid)
            param_dict = {param.key: param.value for param in run.data.params}
            if set(query_dict.keys()).issubset(param_dict.keys()):
                param_unixtime = int(param_dict[exp_to_time[experiment]])
                param_date = datetime.fromtimestamp(param_unixtime, tz=tz).date()

                if query_date == param_date and \
                   query_dict[exp_to_key[experiment]] == param_dict[exp_to_key[experiment]]:
                    if max_param_time <= param_unixtime:
                        max_param_time = param_unixtime
                        found_run_uuid = run_info.run_uuid
                        found_artifact_uri = run_info.artifact_uri

        if max_param_time:
            return found_run_uuid, found_artifact_uri, str(max_param_time)
        else:
            raise RunNotFoundError("There is no run with the given condition.")

    else:
        raise RunNotFoundError("The value of run_time is invalid: %s" % run_time)


def get_latest_run_time(client:mlflow.tracking.MlflowClient, source_experiment:str,
                   source_run_time:int, source_query:str, target_query:str) -> int:
    """
    get the latest target run. A run in processing or model is based on a run in load or processing,
    respectively. In such a case we call the former the target run and the latter the source run.
    That is, the target run is based on the source run.

    Given the source run, this function look for the latest target run. If there is no target run,
    then RunNotFoundError is raised.

    :param client: mlflow.tracking.MlflowClient
    :param source_experiment: experiment where you want to look for a run
    :param source_run_time: run_time of the source run
    :param source_query: query (table or logic) of the source run
    :param target_query: query (logic or algorithm) of the source run
    :return: run_time of the target run
    """

    if source_experiment == "load":
        target_experiment = "processing"
    else:
        target_experiment = "model"

    source_run_time = int(source_run_time)
    experiment_id = client.get_experiment_by_name(target_experiment).experiment_id
    found_run_time = 0

    print("Looking for the newest target run having the given source run.")
    print("SOURCE: experiment: %s, query: %s, run_time: %s" % (source_experiment, source_query, source_run_time))
    print("TARGET: experiment: %s, query: %s" % (target_experiment, target_query))

    for run_info in client.list_run_infos(experiment_id):
        run = client.get_run(run_info.run_uuid)

        param_dict = {param.key: param.value for param in run.data.params} ##

        try:
            param_source_query = param_dict[exp_to_key[source_experiment]]
            param_source_run_time = int(param_dict[exp_to_time[source_experiment]])

            param_target_query = param_dict[exp_to_key[target_experiment]]
            param_target_run_time = int(param_dict[exp_to_time[target_experiment]])

        except Exception as e:
            print("There is an invalid run (%s): %s" % (run_info.run_uuid,e))
            continue

        ## filter source_query
        if param_source_query != source_query:
            continue

        ## filter source_run_time
        if param_source_run_time != source_run_time:
            continue

        ## filter target_query
        if param_target_query != target_query:
            continue

        if param_source_run_time < source_run_time:
            continue
        else:
            found_run_time = param_target_run_time

    if found_run_time:
        return found_run_time
    else:
        raise RunNotFoundError


def get_artifact(client:mlflow.tracking.MlflowClient, run_uuid:str, artifact_uri:str,
                 file_name:str) -> Any:
    """
    download the specified artifact and deserialize it.

    :param client: mlflow.tracking.MlflowClient instance
    :param run_uuid: uuid of the run (cf. lib.enrichment.look_up_run_load)
    :param artifact_uri: artifact_url (cf. lib.enrichment.look_up_run_load)
    :param file_name: name of the file (without ".pkl")
    :return: DataFrame or Python function
    """

    print("Downloading the artifact (%s.pkl)" % file_name)
    tmp_dir = Path(client.download_artifacts(run_uuid, artifact_uri))
    artifact_dir = tmp_dir.joinpath("data")
    tmp_dir = tmp_dir.parent

    file_path = artifact_dir.joinpath("%s.pkl" % file_name)
    if file_path.exists():
        print("Deserializing the found pickle data.")
        with file_path.open("rb") as f:
            obj = pickle.load(f)

        print("Deleting the temporary directory %s" % tmp_dir)
        rmtree(str(tmp_dir))

        return obj

    else:
        raise FileNotFoundError("%s can not be found" % file_path)
