from typing import Union, Tuple
import mlflow

import os
import hashlib
import re
from datetime import datetime


def sha256sum(path) -> str:
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


def look_up_run_load(client:mlflow.tracking.MlflowClient, tz,
                     retrieval_time:Union[int,str]=None, table:str=None,
                     dataset:str=None) -> Tuple[str,str]:
    """
    find uuid of a run with the given retrieval_time, table and dataset.
    If retrieval_time is a date in ISO 8601 format (i.e. YYYY-MM-DD),
    then the newest run of the given date is returned.

    :param client: MlflowClient instance
    :param tz: pytz
    :param retrieval_time: unix time (int) or YYYY-MM-DD (str)
    :param table: name of the table/data
    :param dataset: "train", "test" or "full"
    :return: (uuid of the found run, artifact_uri of the run)
    """

    try:
        experiment_id = client.get_experiment_by_name("load").experiment_id
    except AttributeError:
        print("The experiment 'load' cannot be found.")
        return "",""

    param_keys = ["retrieval_time", "table", "dataset"]
    param_vals = [str(retrieval_time), table, dataset]
    query_dict = dict(zip(param_keys, param_vals))

    if re.search(r"\d+$", retrieval_time):
        ### retrieval_time is a unixtime

        for run_info in client.list_run_infos(experiment_id):
            run = client.get_run(run_info.run_uuid)
            param_dict = {param.key: param.value for param in run.data.params}
            if set(param_keys).issubset(param_dict.keys()):
                if all([query_dict[k] == param_dict[k] for k in param_keys]):
                    return run_info.run_uuid, run_info.artifact_uri
        return "","" ## not found

    elif isinstance(retrieval_time, str):
        ### retrieval_time is a date in YYYY-MM-DD

        try:
            d = datetime.strptime(retrieval_time, "%Y-%M-%d")
            query_date = tz.localize(datetime(year=d.year, month=d.month, day=d.day)).date()
        except ValueError as e:
            print(e)
            return "",""

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
                    print(run_info.run_uuid, run_retrieval_date, param_dict)
                    if max_retrieval_time <= run_retrieval_unixtime:
                        max_retrieval_time = run_retrieval_unixtime
                        found_run_uuid = run_info.run_uuid
                        found_artifact_uri = run_info.artifact_uri

        return found_run_uuid, found_artifact_uri

    elif not retrieval_time:
        return "",""

    else:
        raise ValueError("The value of retrieval_time is invalid.")