import abc
import re
from abc import ABCMeta
from typing import Optional, Union

from datasets import Dataset, load_dataset
from transformers import PretrainedConfig, DataCollatorWithPadding, EvalPrediction, Trainer
from transformers.trainer_utils import PredictionOutput

from i6_dstc10.arguments import ModelArguments, DataPredictionArguments, DataTrainingArguments
from i6_dstc10.preprocessing import Pipeline


class Method(abc.ABC):
  def __init__(self, model_args: ModelArguments, data_args: Union[DataTrainingArguments, DataPredictionArguments], config: PretrainedConfig, tokenizer):
    self.model_args = model_args
    self.data_args = data_args
    self.config = config
    self.tokenizer = tokenizer

    tokenizer.add_special_tokens({
      "additional_special_tokens": sorted(self.get_special_tokens())
    })
    self.metrics = []

  def get_special_tokens(self):
    return [
      self.model_args.user_token,
      self.model_args.agent_token,
      self.model_args.knowledge_tag_token,
      self.model_args.knowledge_sep_token,
    ]

  @abc.abstractmethod
  def get_model_class(self, config: PretrainedConfig):
    raise NotImplementedError()

  def get_model(self, run_mode, config: PretrainedConfig):
    model_class = self.get_model_class(config)
    model = model_class.from_pretrained(
      self.model_args.model_name_or_path,
      from_tf=bool(".ckpt" in self.model_args.model_name_or_path),
      config=config,
      cache_dir=self.model_args.cache_dir,
      revision=self.model_args.model_revision,
      use_auth_token=True if self.model_args.use_auth_token else None,
    )
    model.resize_token_embeddings(len(self.tokenizer))
    return model

  @abc.abstractmethod
  def preprocess_features(self, features):
    raise NotImplementedError()

  def preprocess_features_and_maybe_normalize(self, features):
    if self.data_args.dataset_transformations is not None:
      pipeline = Pipeline(self.data_args.dataset_transformations)
      for i, turns in enumerate(features["turns"]):
        for j, turn in enumerate(turns):
          features["turns"][i][j]["text"] = pipeline.apply(turns[j])

    return self.preprocess_features(features)

  def get_data_collator(self):
    return DataCollatorWithPadding(self.tokenizer)

  def get_trainer_class(self):
    return Trainer

  def postprocess_predictions(self, p: PredictionOutput, dataset):
    return p

  @abc.abstractmethod
  def compute_metrics(self, p: EvalPrediction):
    raise NotImplementedError()

  def _get_dataset(self, split, config_name=None):
    if config_name is None:
      config_name = self.data_args.dataset_config_name
  
    dataset: Optional[Dataset] = load_dataset(
      self.data_args.dataset_name,
      config_name,
      split=split,
      cache_dir=self.model_args.cache_dir,
      data_files=self.data_args.dataset_data_files,
      dataset_filter_dict=self.data_args.dataset_filter_dict
    )

    old_eval_column_names = dataset.column_names
    if config_name != "evaluation":
      dataset = dataset.map(
        self.preprocess_features_and_maybe_normalize,
        batched=True,
        remove_columns=old_eval_column_names,
      )
    return dataset

  def get_train_dataset(self):
    return self._get_dataset(self.data_args.dataset_train_split)

  def get_eval_dataset(self):
    if self.data_args.dataset_eval_split is None:
      return None
    return self._get_dataset(self.data_args.dataset_eval_split)

  def get_test_dataset(self):
    return self._get_dataset(self.data_args.dataset_test_split)

  def get_full_dataset(self, config_name="evaluation", split=None):
    if split is None:
      split = self.data_args.dataset_test_split
      split = split.split("[")[0]
    return self._get_dataset(split, config_name=config_name)


class MethodWithDocumentDataset(Method, metaclass=ABCMeta):
  def get_document_dataset(self, split):
    if self.model_args.selection_level == 'domain':
      document_dataset_config_name = "knowledge_domains"
    elif self.model_args.selection_level in ['entity', 'domain_entity']:
      document_dataset_config_name = "knowledge_entities"
    else:
      document_dataset_config_name = "knowledge"
    document_dataset_split = re.sub(r'^(\w+)(\[.*\])?', r'\1', split)
    document_dataset = load_dataset(
      self.data_args.dataset_name,
      document_dataset_config_name,
      split=document_dataset_split,
      cache_dir=self.model_args.cache_dir,
      data_files=self.data_args.dataset_data_files,
    )
    return document_dataset

  @abc.abstractmethod
  def preprocess_documents(self, features):
    raise NotImplementedError()
  