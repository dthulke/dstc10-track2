import copy
import os
import shutil
import subprocess as sp
import copy
import json

from sisyphus import *

import i6_core.util as util
from i6_core.returnn.config import instanciate_delayed

Path = setup_path(__package__)


class HuggingfaceSearchJob(Job):
  """
  Train a Huggingface transformer model

  """

  def __init__(
      self,
      code_root,
      model_path,
      config,
      search_data_config,
      *,  # args below are keyword only
      time_rqmt=4,
      mem_rqmt=4,
      cpu_rqmt=2,
      gpu_rqmt=1,
      python_exe=None,
      **kwargs
  ):
    """

    :param code_root: Root directory for the training scripts. Expected to contain a training script.
    :param config:
    :param num_epochs:
    :param time_rqmt:
    :param mem_rqmt:
    :param cpu_rqmt:
    :param gpu_rqmt:
    """

    self.code_root = code_root
    self.model_path = model_path
    self.config = config
    self.search_data_config = search_data_config
    self.python_exe = (python_exe if python_exe is not None else gs.PYTHON_EXE)

    self.rqmt = {
      "gpu": gpu_rqmt,
      "cpu": cpu_rqmt,
      "mem": mem_rqmt,
      "time": time_rqmt,
    }

    self.out_config_file = self.output_path("search_config.json")
    self.out_search_file = self.output_path("search_output.json")

    self._update_config()

  def _update_config(self):
    fixed_config = {
      'prediction_output_file': self.out_search_file,
      'output_dir': 'trainer_output_dir',
    }
    assert fixed_config.keys().isdisjoint(self.config.keys())
    self.config = copy.deepcopy(self.config)
    self.config.update(fixed_config)
    # Overwrite model path
    self.config['model_name_or_path'] = self.model_path
    self.config['config_name'] = None
    self.config['tokenizer_name'] = None
    assert self.config.keys().isdisjoint(self.search_data_config.keys())

  def _get_run_cmd(self):
      run_cmd = [
          tk.uncached_path(self.python_exe),
          os.path.join(tk.uncached_path(self.code_root), "predict.py"),
          self.out_config_file.get_path(),
      ]
      return run_cmd

  def create_files(self):
    instanciated_config = instanciate_delayed({
      **copy.deepcopy(self.config),
      **copy.deepcopy(self.search_data_config),
    })
    with util.uopen(self.out_config_file, 'wt') as fp:
      json.dump(instanciated_config, fp)

    util.create_executable("run.sh", self._get_run_cmd())

  def run(self):
    sp.check_call(self._get_run_cmd())

  def tasks(self):
    yield Task("create_files", mini_task=True)
    yield Task("run", resume="run", rqmt=self.rqmt)

  @classmethod
  def hash(cls, kwargs):
      hash_kwargs = copy.deepcopy(kwargs)
      excluded_keys = ['time_rqmt', 'mem_rqmt', 'cpu_rqmt', 'gpu_rqmt']
      for key in excluded_keys:
        if key in hash_kwargs:
          del hash_kwargs[key]

      return super().hash(hash_kwargs)


class HuggingfaceBuildIndexJob(Job):
  """
  Train a Huggingface transformer model

  """

  def __init__(
      self,
      code_root,
      model_path,
      config,
      search_data_config,
      *,  # args below are keyword only
      time_rqmt=4,
      mem_rqmt=4,
      cpu_rqmt=2,
      gpu_rqmt=1,
      python_exe=None,
  ):
    """

    :param code_root: Root directory for the training scripts. Expected to contain a training script.
    :param config:
    :param num_epochs:
    :param time_rqmt:
    :param mem_rqmt:
    :param cpu_rqmt:
    :param gpu_rqmt:
    """

    self.code_root = code_root
    self.model_path = model_path
    self.config = config
    self.search_data_config = search_data_config
    self.python_exe = (python_exe if python_exe is not None else gs.PYTHON_EXE)

    self.rqmt = {
      "gpu": gpu_rqmt,
      "cpu": cpu_rqmt,
      "mem": mem_rqmt,
      "time": time_rqmt,
    }

    self.out_config_file = self.output_path("search_config.json")
    self.out_index_path = self.output_path("index.faiss")

    self._update_config()

  def _update_config(self):
    fixed_config = {
      'test_documents_faiss_index_path': self.out_index_path,
      'output_dir': 'trainer_output_dir',
    }
    assert fixed_config.keys().isdisjoint(self.config.keys())
    self.config = copy.deepcopy(self.config)
    self.config.update(fixed_config)
    # Overwrite model path
    self.config['model_name_or_path'] = self.model_path
    self.config['config_name'] = None
    self.config['tokenizer_name'] = None
    assert self.config.keys().isdisjoint(self.search_data_config.keys())

  def _get_run_cmd(self):
      run_cmd = [
          tk.uncached_path(self.python_exe),
          os.path.join(tk.uncached_path(self.code_root), "build_index.py"),
          self.out_config_file.get_path(),
      ]
      return run_cmd

  def create_files(self):
    instanciated_config = instanciate_delayed({
      **copy.deepcopy(self.config),
      **copy.deepcopy(self.search_data_config),
    })
    with util.uopen(self.out_config_file, 'wt') as fp:
      json.dump(instanciated_config, fp)

    util.create_executable("run.sh", self._get_run_cmd())

  def run(self):
    sp.check_call(self._get_run_cmd())

  def tasks(self):
    yield Task("create_files", mini_task=True)
    yield Task("run", resume="run", rqmt=self.rqmt)

  @classmethod
  def hash(cls, kwargs):
      hash_kwargs = copy.deepcopy(kwargs)
      excluded_keys = ['time_rqmt', 'mem_rqmt', 'cpu_rqmt', 'gpu_rqmt']
      for key in excluded_keys:
        if key in hash_kwargs:
          del hash_kwargs[key]

      return super().hash(hash_kwargs)


class EnsembleDetectionByMajorityVoteJob(Job):
  """
  Job to take the majority vote over multiple model outputs.
  """

  def __init__(
      self,
      result_files,
      *,  # args below are keyword only
      time_rqmt=1,
      mem_rqmt=4,
      cpu_rqmt=2,
      gpu_rqmt=0,
      python_exe=None,
      **kwargs
  ):
    """

    :param code_root: Root directory for the training scripts. Expected to contain a training script.
    :param config:
    :param num_epochs:
    :param time_rqmt:
    :param mem_rqmt:
    :param cpu_rqmt:
    :param gpu_rqmt:
    """

    self.result_files = result_files

    self.rqmt = {
      "gpu": gpu_rqmt,
      "cpu": cpu_rqmt,
      "mem": mem_rqmt,
      "time": time_rqmt,
    }

    self.out_search_file = self.output_path("search_output.json")

  def run(self):
    import json
    predictions = []
    out = []
    
    for file in self.result_files:
      with open(file, "r") as f:
        predictions.append(json.load(f))
      
    for sample in zip(*predictions):
      vote = sum([item["target"] for item in sample]) >= (len(sample) / 2)
      out.append({"target": vote})

    with open(self.out_search_file, "w") as f:
      json.dump(out, f)

  def tasks(self):
    yield Task("run", resume="run", rqmt=self.rqmt, mini_task=True)

  @classmethod
  def hash(cls, kwargs):
      hash_kwargs = copy.deepcopy(kwargs)
      excluded_keys = ['time_rqmt', 'mem_rqmt', 'cpu_rqmt', 'gpu_rqmt']
      for key in excluded_keys:
        if key in hash_kwargs:
          del hash_kwargs[key]

      return super().hash(hash_kwargs)

class CombineGenerationPredictionsJob(Job):
  """
  Combines the predictions of multiple jobs which can be used after performing search over
  chunks of the dataset in parallel.
  """

  def __init__(
      self,
      result_files,
      *,  # args below are keyword only
      label=None,
      time_rqmt=1,
      mem_rqmt=4,
      cpu_rqmt=2,
      gpu_rqmt=0,
      python_exe=None,
      **kwargs
  ):
    """

    :param code_root: Root directory for the training scripts. Expected to contain a training script.
    :param config:
    :param num_epochs:
    :param time_rqmt:
    :param mem_rqmt:
    :param cpu_rqmt:
    :param gpu_rqmt:
    """
    self.label = label
    self.result_files = result_files

    self.rqmt = {
      "gpu": gpu_rqmt,
      "cpu": cpu_rqmt,
      "mem": mem_rqmt,
      "time": time_rqmt,
    }

    self.out_search_file = self.output_path("search_output.json")

  def run(self):
    import itertools
    import json
    predictions = []
    
    if self.label is not None:
      with open(self.label, "r") as f:
        labels = json.load(f)

    for file in self.result_files:
      with open(file, "r") as f:
        predictions.append(json.load(f))

    idx = 0
    if self.label is not None:
      out = labels
    else:
      out = predictions[0]
    all_generations = [sample["response"] for sample in itertools.chain.from_iterable(predictions) if sample["target"] and "response" in sample]
      
    for i, sample in enumerate(out):
      if sample["target"]:
        out[i]["response"] = all_generations[idx]
        idx += 1
        if "knowledge" in sample:
          for j, snippet in enumerate(sample["knowledge"]):
            out[i]["knowledge"][j]["entity_id"] = snippet["entity_id"]
      if "source" in out[i]:
        del out[i]["source"]

    with open(self.out_search_file, "w") as f:
      json.dump(out, f)

  def tasks(self):
    yield Task("run", resume="run", rqmt=self.rqmt, mini_task=True)

  @classmethod
  def hash(cls, kwargs):
      hash_kwargs = copy.deepcopy(kwargs)
      excluded_keys = ['time_rqmt', 'mem_rqmt', 'cpu_rqmt', 'gpu_rqmt']
      for key in excluded_keys:
        if key in hash_kwargs:
          del hash_kwargs[key]

      return super().hash(hash_kwargs)