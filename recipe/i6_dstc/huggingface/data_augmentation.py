import copy
import itertools
import os
import shutil
import subprocess as sp
import json

from datasets import load_dataset
from sisyphus import *

import i6_core.util as util
from i6_core.returnn.config import instanciate_delayed

Path = setup_path(__package__)


class DataAugmentationJob(Job):
    """
    Augments the training data according to one of the implemented strategies.
    """

    def __init__(
        self,
        code_root, 
        dataset, 
        split,
        strategy="detection",
        model_name_or_path=None,
        dataset_data_files=None,
        time_rqmt=1,
        mem_rqmt=1,
        cpu_rqmt=1,
        gpu_rqmt=0,
        **kwargs
    ):
        self.code_root = code_root
        self.dataset = dataset
        self.split = split
        self.strategy = strategy
        self.model_name_or_path = model_name_or_path
        self.dataset_data_files = dataset_data_files

        self.out_logs_path = self.output_path("out_logs.json")
        self.out_labels_path = self.output_path("out_labels.json")
        self.out_knowledge_path = self.output_path("out_knowledge.json")
        self.out_dataset_data_files_json = self.output_path("out_dataset_data_files.json")

        self.rqmt = {
            "gpu": gpu_rqmt,
            "cpu": cpu_rqmt,
            "mem": mem_rqmt,
            "time": time_rqmt,
        }

    def _get_run_cmd(self):
        args = [
            "--dataset", self.dataset,
            "--split", self.split,
            "--out_logs", self.out_logs_path.get_path(),
            "--out_labels", self.out_labels_path.get_path(),
            "--out_knowledge", self.out_knowledge_path.get_path(),
            "--strategy", self.strategy
        ]
        if self.model_name_or_path is not None:
            args.extend([
                "--model_name_or_path", self.model_name_or_path
            ])
        if self.dataset_data_files is not None:
            import json
            with open(self.out_dataset_data_files_json, "w") as f:
                json.dump(self.dataset_data_files, f)
            args.extend([
                "--dataset_data_files", self.out_dataset_data_files_json.get_path()
            ])

        run_cmd = [
            tk.uncached_path(gs.PYTHON_EXE),
            os.path.join(tk.uncached_path(self.code_root), "i6_dstc10/datasets/data_augmentation.py"),
            *args
        ]
        return run_cmd

    def create_files(self):
        util.create_executable("run.sh", self._get_run_cmd())

    def run(self):
        sp.check_call(self._get_run_cmd())

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task("run", resume="run", rqmt=self.rqmt, mini_task=self.rqmt["gpu"]!=1)