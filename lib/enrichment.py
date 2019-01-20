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



def look_up_run_load(client:mlflow.tracking.MlflowClient, tz=None,
                     retrieval_time:Union[int,str]=None, table:str=None,
                     dataset:str=None) -> Tuple[str,str]:
    """
    find uuid of a run with the given retrieval_time, table and dataset.
    If retrieval_time is a date in ISO 8601 format (i.e. YYYY-MM-DD),
    then the newest run of the given date is returned.
    If no run is found, an error is raised.

    :param client: MlflowClient instance
    :param tz: pytz timezone (needed if retrieval_time is a date.)
    :param retrieval_time: unix time (int) or YYYY-MM-DD (str)
    :param table: name of the table/data
    :param dataset: "train", "test" or "full"
    :return: (uuid of the found run, artifact_uri of the run)
    """

    try:
        experiment_id = client.get_experiment_by_name("load").experiment_id
    except AttributeError:
        raise ExperimentNotFoundError("The experiment 'load' cannot be found.")

    param_keys = ["retrieval_time", "table"]
    param_vals = [str(retrieval_time), table, dataset]
    query_dict = dict(zip(param_keys, param_vals))

    if re.match(r"\d+$", retrieval_time):
        ### retrieval_time is a unixtime
        print("Looking for the exact dataset (retrieval_time=%s)" % retrieval_time)

        for run_info in client.list_run_infos(experiment_id):
            run = client.get_run(run_info.run_uuid)
            param_dict = {param.key: param.value for param in run.data.params}
            if set(param_keys).issubset(param_dict.keys()):
                if all([query_dict[k] == param_dict[k] for k in param_keys]):
                    return run_info.run_uuid, run_info.artifact_uri

        raise RunNotFoundError("There is no run with the given condition.")

    elif isinstance(retrieval_time, str):
        ### retrieval_time is a date in YYYY-MM-DD
        print("Looking for the newest dataset on %s" % retrieval_time)

        d = datetime.strptime(retrieval_time, "%Y-%M-%d")
        query_date = tz.localize(datetime(year=d.year, month=d.month, day=d.day)).date()

        max_retrieval_time = 0
        found_run_uuid = ""
        found_artifact_uri = ""

        for run_info in client.list_run_infos(experiment_id):
            run = client.get_run(run_info.run_uuid)
            param_dict = {param.key: param.value for param in run.data.params}
            if set(param_keys).issubset(param_dict.keys()):
                run_retrieval_unixtime = int(param_dict["retrieval_time"])
                run_retrieval_date = datetime.fromtimestamp(run_retrieval_unixtime, tz=tz).date()

                if query_date == run_retrieval_date and \
                        all([query_dict[k] == param_dict[k] for k in param_keys[1:]]):
                    #print(run_info.run_uuid, run_retrieval_date, param_dict)
                    if max_retrieval_time <= run_retrieval_unixtime:
                        max_retrieval_time = run_retrieval_unixtime
                        found_run_uuid = run_info.run_uuid
                        found_artifact_uri = run_info.artifact_uri

        if max_retrieval_time:
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
