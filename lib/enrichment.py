from typing import Union, Tuple, Any

import os
import re
import pickle
import shutil
import hashlib
import tempfile
from pathlib import Path
from shutil import rmtree
from datetime import datetime
from typing import Dict

import mlflow
import pandas as pd


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


def store_artifact(data:Dict[str, Any], experiment:str,
                   parameters:Dict[str, Any], metrics:Dict[str, Any]):
    """
    Start a run and store the given DataFrame as an artifact.

    :param data: file_name (without .pkl): Python Object
    :param experiment: name of the experiment
    :param parameters:
    :param metrics:
    """

    #if experiment not in ["load","processing","model"]:
    #    raise ExperimentNotFoundError("You gave a wrong experiment: %s" % experiment)

    mlflow.set_experiment(experiment)

    with mlflow.start_run():
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            for artifact_path, obj in data.items():
                file_name = "%s.pkl" % artifact_path
                target_file = tmp_dir.joinpath(file_name)
                print("Saving %s ..." % file_name)
                with target_file.open("wb") as fo:
                    pickle.dump(obj, fo)
                mlflow.log_artifact(str(target_file), "data")

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
                query:str, run_time:Union[int,str], tz=None) -> Tuple[str,str]:
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
    :return: (uuid of the found run, artifact_uri of the run)
    """

    """
    NOTE: The rule of the names should be decided by team at first and not be changed.    
    """
    exp_to_key = {"load": "table", "processing": "logic", "model": "algorithm"}
    exp_to_time = {"load": "retrieval_time", "processing": "processed_time", "model": "trained_time"}

    try:
        query_dict = {exp_to_key[experiment]: query, exp_to_time[experiment]: str(run_time)}
        experiment_id = client.get_experiment_by_name(experiment).experiment_id
    except KeyError:
        raise ExperimentNotFoundError("The experiment '%s' cannot be found." % experiment)
    except AttributeError:
        raise ExperimentNotFoundError("The experiment '%s' cannot be found." % experiment)

    #####
    print("Searching a run in expeciment '%s'" % experiment)
    if re.match(r"\d+$", query_dict[exp_to_time[experiment]]):
        ### retrieval_time is a unixtime
        print("Looking for the exact dataset (run_time=%s)" % run_time)

        for run_info in client.list_run_infos(experiment_id):
            run = client.get_run(run_info.run_uuid)
            param_dict = {param.key: param.value for param in run.data.params}

            if set(query_dict.keys()).issubset(param_dict.keys()):
                if all([query_dict[k] == param_dict[k] for k in query_dict.keys()]):
                    return run_info.run_uuid, run_info.artifact_uri

        raise RunNotFoundError("There is no run with the given condition.")

    elif isinstance(run_time, str):
        ### retrieval_time is a date in YYYY-MM-DD
        print("Looking for the newest dataset on %s" % run_time)
        if tz is None:
            raise ValueError("'tz' object is None.")

        d = datetime.strptime(run_time, "%Y-%M-%d")
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
            return found_run_uuid, found_artifact_uri
        else:
            raise RunNotFoundError("There is no run with the given condition.")

    else:
        raise ValueError("The value of retrieval_time is invalid.")


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

    print("Downloading the artifact")
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
