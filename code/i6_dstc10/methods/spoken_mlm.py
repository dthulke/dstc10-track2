from typing import Optional

from datasets import load_dataset, concatenate_datasets, Dataset, load_from_disk
from transformers import PretrainedConfig, EvalPrediction, AutoModelForMaskedLM, DataCollatorForWholeWordMask

from i6_dstc10.methods.base import Method
from i6_dstc10.preprocessing import create_concatenated_model_input, Pipeline


class MaskedLanguageModelingOnSpokenDataMethod(Method):

  def get_model_class(self, config: PretrainedConfig):
      return AutoModelForMaskedLM

  def get_data_collator(self):
      return DataCollatorForWholeWordMask(self.tokenizer)

  def compute_metrics(self, p: EvalPrediction):
      return {}

  name = "spoken_mlm"

  def _collapse_turns(self, turns):
    if len(turns) == 0:
      return turns

    out = []
    current_speaker = turns[0]["speaker"]
    current_turn = turns[0]

    for turn in turns[1:]:
      if turn["speaker"] == current_speaker:
        current_turn["text"] = current_turn["text"] + " " + turn["text"]
      else:
        out.append(current_turn)
        current_speaker = turn["speaker"]
        current_turn = turn
    out.append(current_turn)

    return out

  def preprocess_features(self, features):
    return self.preprocess_features_dstc9(features)

  def preprocess_features_ccpe(self, features):
    dstc_format_input = []
    dstc_format_target = []
    dialogs = features["data"]
    for sample in dialogs:
      dialog = sample["utterances"]
      for i, turn in enumerate(dialog):
        dialog[i] = {
          "text": turn["text"],
          "speaker": "U" if turn["speaker"] == "USER" else "S"
        }

      dialog = self._collapse_turns(dialog)

      for i in range(len(dialog) - 1):
        # only predict agent truns
        if dialog[i + 1]["speaker"] == "S":
          source = dialog[:i + 1]
          target = dialog[i + 1]
          dstc_format_input.append(source)
          dstc_format_target.append(target["text"])

    input_ids = [
      create_concatenated_model_input(self.model_args, turns, self.tokenizer, knowledge=None)
      for turns in dstc_format_input
    ]

    return {
      'input_ids': input_ids,
    }

  def preprocess_features_dstc9(self, features):
    input_ids = [
      create_concatenated_model_input(self.model_args, turns, self.tokenizer, knowledge=None)
      for turns, knowledge in zip(features['turns'], features['knowledge'])
    ]

    return {
      'input_ids': input_ids
    }

  def preprocess_features_taskmaster(self, features):
    if self.data_args.dataset_transformations is not None:
      pipeline = Pipeline(self.data_args.dataset_transformations)
    else:
      pipeline = None

    dstc_format_input = []
    dstc_format_target = []
    utterances = features["utterances"]
    for turns in utterances:
      for i, turn in enumerate(turns):
        turns[i] = {
          "text": pipeline.apply(turn["text"]) if pipeline is not None else turn["text"],
          "speaker": "U" if turn["speaker"] == "USER" else "S"
        }
      turns = self._collapse_turns(turns)

      for i in range(len(turns) - 1):
        # only predict agent truns
        if turns[i + 1]["speaker"] == "S":
          source = turns[:i + 1]
          target = turns[i + 1]
          dstc_format_input.append(source)
          dstc_format_target.append(target["text"])

    assert len(dstc_format_input) == len(dstc_format_target)

    input_ids = [
      create_concatenated_model_input(self.model_args, turns, self.tokenizer, knowledge=None)
      for turns in dstc_format_input
    ]

    return {
      'input_ids': input_ids
    }

  def _get_dataset(self, split):
    if split != 'train':
      return super()._get_dataset(split)

    import getpass
      with open("") as f: #TODO: add path
      ccpe = json.load(f)
      ccpe = {"data": ccpe}
      ccpe = Dataset.from_dict(ccpe)

    ccpe = ccpe.map(
      self.preprocess_features_ccpe,
      batched=True,
      remove_columns=["data"],
    )

    dstc_train: Optional[Dataset] = load_dataset(
      self.data_args.dataset_name,
      self.data_args.dataset_config_name,
      split=split,
      cache_dir=self.model_args.cache_dir,
      data_files=self.data_args.dataset_data_files,
      dataset_filter_dict=self.data_args.dataset_filter_dict
    )
    old_eval_column_names = dstc_train.column_names
    dstc_train = dstc_train.map(
      self.preprocess_features_dstc9,
      batched=True,
      remove_columns=old_eval_column_names
    )

    train = concatenate_datasets([ccpe, dstc_train])

    configs = ["hotels", "restaurant-search", "flights", "food-ordering", "sports", "movies", "music"]
    # configs = ["sports",] #"movies", "music"]
    datasets = [load_dataset("taskmaster2", config)["train"] for config in configs]
    datasets.append(load_dataset("taskmaster1", "woz_dialogs")["train"])
    concatenated = concatenate_datasets(datasets)

    old_eval_column_names = concatenated.column_names
    concatenated = concatenated.map(
      self.preprocess_features_taskmaster,
      batched=True,
      remove_columns=old_eval_column_names
    )

    train = concatenate_datasets([train, concatenated])

    return train
