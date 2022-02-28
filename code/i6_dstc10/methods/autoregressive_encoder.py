import itertools
import random
import re
from dataclasses import dataclass
from typing import Optional, Union, List, Dict, Any, Tuple

import numpy as np
import torch
from datasets import load_metric, load_dataset, arrow_dataset
from torch import nn
from torch.cuda.amp import autocast
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset
from transformers import PretrainedConfig, EvalPrediction, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, \
  PreTrainedModel, PreTrainedTokenizerBase, Trainer
from transformers.file_utils import is_sagemaker_dp_enabled, is_sagemaker_mp_enabled, PaddingStrategy
from transformers.trainer_utils import PredictionOutput
from transformers.trainer_pt_utils import nested_detach


if is_sagemaker_dp_enabled():
    import smdistributed.dataparallel.torch.distributed as dist
else:
    import torch.distributed as dist

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat

from i6_dstc10.arguments import ModelArguments
from i6_dstc10.knowledge_utils import build_knowledge_document_register
from i6_dstc10.methods.base import Method
from i6_dstc10.models.autoregressive_encoder import BartForAutoregressiveEncoder
from i6_dstc10.preprocessing import create_concatenated_dialog_knowledge_input, \
  process_input, process_knowledge, wrap_with_special_tokens


class AutoregressiveEncoderDataset(Dataset):
  def __init__(self, args: ModelArguments, tokenizer, query_dataset, documents_dataset, document_register, train_mode=True):
    self.args = args
    self.tokenizer = tokenizer
    self.query_dataset = query_dataset
    self.documents_dataset = documents_dataset
    self.train_mode = train_mode

    # Map: [domain, entity_id, doc_id] -> idx in documents dataset
    self.document_register = document_register
    self._build_data_infos()

    # Plausibility check
    assert args.selection_level in ['all', 'document', 'entity', 'domain', 'domain_entity']
    assert args.selection_level not in ['entity', 'domain', 'domain_entity'] or args.num_doc_negatives == 0
    assert args.selection_level not in ['domain'] or args.num_entity_negatives == 0

  def _build_data_infos(self):
    data_infos = []
    query_slices = []
    query_slice_counter = 0
    for query_idx, query_item in enumerate(self.query_dataset):
      if self.args.selection_level in ["all", "domain", "domain_entity"]:
        data_info = list(range(len(self.documents_dataset)))
      elif self.args.selection_level == 'entity':
        data_info = list(self.document_register[query_item['domain']].values())
      elif self.args.selection_level == 'document':
        data_info = list(self.document_register[query_item['domain']][query_item['entity_id']].values())
      else:
        assert False
      data_infos.extend(list(zip(itertools.cycle([query_idx]), data_info)))
      query_slices.append((query_slice_counter, query_slice_counter + len(data_info)))
      query_slice_counter += len(data_info)
    self.data_infos = data_infos
    self.query_slices = query_slices

  def _get_number_of_documents_per_sample(self):
    return self.args.num_domain_negatives + self.args.num_entity_negatives + self.args.num_doc_negatives + 1

  def _get_document_index(self, domain, entity_id, doc_id):
    entities = self.document_register[domain]
    if self.args.selection_level == 'domain':
      return entities
    docs = entities[entity_id]
    if self.args.selection_level in ['entity', 'domain_entity']:
      return docs
    return docs[doc_id]

  def _sample_negative(self, query_item, document_type):
    # Set the sampling level
    negative_sample_level = None
    if document_type < self.args.num_domain_negatives + 1:
      # Domain negatives
      negative_sample_level = "domain"
    elif document_type < self.args.num_entity_negatives + self.args.num_domain_negatives + 1:
      # Entity negatives
      if len(self.document_register[query_item['domain']]) > 1:
        negative_sample_level = "entity"
      else:
        negative_sample_level = "domain"
    elif document_type < self.args.num_doc_negatives + self.args.num_entity_negatives + self.args.num_domain_negatives + 1:
      # Doc negatives
      negative_sample_level = "document"

    # Randomly select negatives
    if negative_sample_level == "domain":
      negative_domain = random.choice(list(self.document_register.keys() - {query_item['domain']}))
    else:
      negative_domain = query_item['domain']

    if self.args.selection_level in ['domain']:
      negative_entity = None
    elif negative_sample_level == 'entity':
      negative_entity = random.choice(list(self.document_register[negative_domain].keys() - {query_item['entity_id']}))
    elif negative_sample_level == 'domain':
      negative_entity = random.choice(list(self.document_register[negative_domain].keys()))
    else:
      negative_entity = query_item['entity_id']

    if self.args.selection_level in ['domain_entity', 'entity']:
      negative_doc = None
    elif negative_sample_level == 'document':
      possible_documents = list(self.document_register[negative_domain][negative_entity].keys() - {query_item['doc_id']})
      if len(possible_documents) == 0:
        negative_entity = random.choice(list(self.document_register[negative_domain].keys() - {query_item['entity_id']}))
        possible_documents = list(self.document_register[negative_domain][negative_entity].keys())
      negative_doc = random.choice(possible_documents)
    else:
      negative_doc = random.choice(list(self.document_register[negative_domain][negative_entity].keys()))

    return self._get_document_index(negative_domain, negative_entity, negative_doc)

  def __getitem__(self, index):
    document_index: int
    label: int
    if not self.train_mode:
      query_index, document_index = self.data_infos[index]
      query_item = self.query_dataset[query_index]

      try:
        label_index = self._get_document_index(query_item['domain'], query_item['entity_id'], query_item['doc_id'])
        label = int(document_index == label_index)
      except Exception:
        label = 0
    else:

      query_index = index
      query_item = self.query_dataset[query_index]

      positive_document_index = self._get_document_index(query_item['domain'], query_item['entity_id'], query_item['doc_id'])
      negative_document_indices = [
        self._sample_negative(query_item, i + 1) for i in range(self.args.num_domain_negatives + self.args.num_entity_negatives + self.args.num_doc_negatives)
      ]

      query_input_ids = wrap_with_special_tokens(self.tokenizer, query_item['input_ids'])
      positive_document_input_ids = wrap_with_special_tokens(self.tokenizer,
                                                             self.documents_dataset[positive_document_index]['input_ids'])
      negative_document_input_ids = [wrap_with_special_tokens(self.tokenizer,
                                                             self.documents_dataset[index]['input_ids'])
                                     for index in negative_document_indices] if len(negative_document_indices) > 0 else None

      return {
        'input_ids': query_input_ids,
        'labels': positive_document_input_ids,
        **({'negative_labels': negative_document_input_ids} if negative_document_input_ids is not None else {}),
      }

    query_input_ids = wrap_with_special_tokens(self.tokenizer, query_item['input_ids'])
    document_input_ids = wrap_with_special_tokens(self.tokenizer, self.documents_dataset[document_index]['input_ids'])

    return {
      'input_ids': query_input_ids,
      'labels': document_input_ids,
    }

  def __len__(self):
    if not self.train_mode:
      return len(self.data_infos)
    return len(self.query_dataset)


class AutoregressiveEncoderMethod(Method):
  name = "autoregressive_encoder"

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self.metrics = [
      load_metric(metric) for metric in ['accuracy']
    ]

  def get_data_collator(self):
    return DataCollatorForSeq2SeqWithNegativeLabels(self.tokenizer)

  def get_trainer_class(self):
    return AutoregressiveEncoderTrainer

  def preprocess_features(self, features):
    input_ids = [
      process_input(self.model_args, turns, self.tokenizer) for turns in features['turns']
    ]

    data = {
      'input_ids': input_ids,
      'id': features['id'],
      'target': features['target']
    }
    if 'knowledge' in features:
      data.update({
        'domain': [x[0]['domain'] for x in features['knowledge']],
        'entity_id': [x[0]['entity_id'] for x in features['knowledge']],
        'doc_id': [x[0]['doc_id'] for x in features['knowledge']],
      })
    return data

  def preprocess_documents(self, features):
    if self.data_args.dataset_lowercase_entities:
      features = features.copy()
      if 'entity_name' in features and features['entity_name'] is not None:
        features['entity_name'] = features['entity_name'].lower()
    out = {
      'input_ids': process_knowledge(self.model_args, self.tokenizer, features),
      'domain': features['domain'],
    }
    if 'entity_id' in features:
      out['entity_id'] = features['entity_id']
    if 'doc_id' in features:
      out['doc_id'] = features['doc_id']
    return out

  def get_model_class(self, config: PretrainedConfig):
    return BartForAutoregressiveEncoder

  def compute_metrics(self, p: EvalPrediction):
    # TODO better metric (i.e. take the best one)
    return {}

  def postprocess_predictions(self, p: PredictionOutput, dataset):
    # TODO avoid that this is run twice
    scores = torch.tensor(p.predictions)

    def get_documents(query_idx):
      start, end = dataset.query_slices[query_idx]
      k = min(end - start, self.model_args.selection_prediction_topk)
      topk_values, topk_indices = scores[start:end].topk(k)
      return [
        {
          'score': score.item(),
          **{
            k: v for k, v in dataset.documents_dataset[dataset.data_infos[index.item() + start][1]].items()
            if k in ['domain', 'entity_id', 'doc_id']
          },
        }
        for score, index in zip(topk_values, topk_indices)
      ]

    return list(map(get_documents, range(len(dataset.query_dataset))))

  def _get_dataset(self, split, config_name=None):
    query_dataset = super()._get_dataset(split, config_name=config_name)
    if config_name in ["evaluation", "detection"]:
      return query_dataset

    # Remove the slice when loading the document dataset
    if self.model_args.selection_level == 'domain':
      document_dataset_config_name = "knowledge_domains"
    elif self.model_args.selection_level in ['entity', 'domain_entity']:
      document_dataset_config_name = "knowledge_entities"
    else:
      document_dataset_config_name = "knowledge"
    document_dataset_split = re.sub(r'^(\w+)(\[.*\])?', r'\1', split)
    document_dataset: arrow_dataset.Dataset = load_dataset(
      self.data_args.dataset_name,
      document_dataset_config_name,
      split=document_dataset_split,
      cache_dir=self.model_args.cache_dir,
      data_files=self.data_args.dataset_data_files,
    )
    document_register = build_knowledge_document_register(document_dataset)
    old_eval_column_names = document_dataset.column_names
    document_dataset = document_dataset.map(
      self.preprocess_documents,
      batched=False,
      remove_columns=old_eval_column_names,
    )

    return AutoregressiveEncoderDataset(
      self.model_args,
      self.tokenizer,
      query_dataset,
      document_dataset,
      document_register,
      self.data_args.is_training,
    )


@dataclass
class DataCollatorForSeq2SeqWithNegativeLabels:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        model (:class:`~transformers.PreTrainedModel`):
            The model that is being trained. If set and has the `prepare_decoder_input_ids_from_labels`, use it to
            prepare the `decoder_input_ids`

            This is useful when using `label_smoothing` to avoid calculating loss twice.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features):
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                feature["labels"] = (
                    feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                )

        negative_labels = [feature["negative_labels"] for feature in features] if "negative_labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if negative_labels is not None:
            max_label_length = max(len(nl) for nl_batch in negative_labels for nl in nl_batch)
            padding_side = self.tokenizer.padding_side
            for feature in features:
                def get_remainder(nl):
                  return [self.label_pad_token_id] * (max_label_length - len(nl))
                feature["negative_labels"] = [
                    nl + get_remainder(nl) if padding_side == "right" else get_remainder(nl) + nl
                    for nl in feature["negative_labels"]
                ]

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        return features


class AutoregressiveEncoderTrainer(Trainer):
  def prediction_step(
      self,
      model: nn.Module,
      inputs: Dict[str, Union[torch.Tensor, Any]],
      prediction_loss_only: bool,
      ignore_keys: Optional[List[str]] = None,
  ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Perform an evaluation step on :obj:`model` using obj:`inputs`.

    Subclass and override to inject custom behavior.

    Args:
        model (:obj:`nn.Module`):
            The model to evaluate.
        inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
            The inputs and targets of the model.

            The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
            argument :obj:`labels`. Check your model's documentation for all accepted arguments.
        prediction_loss_only (:obj:`bool`):
            Whether or not to return the loss only.

    Return:
        Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
        labels (each being optional).
    """

    if not (not self.args.predict_with_generate or prediction_loss_only):
      assert False, "Not yet implemented."

    has_labels = all(inputs.get(k) is not None for k in self.label_names)
    inputs = self._prepare_inputs(inputs)
    if ignore_keys is None:
      if hasattr(self.model, "config"):
        ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
      else:
        ignore_keys = []

    # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
    if has_labels:
      labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
      if len(labels) == 1:
        labels = labels[0]
    else:
      labels = None

    with torch.no_grad():
      if is_sagemaker_mp_enabled():
        raw_outputs = smp_forward_only(model, inputs)
        if has_labels:
          if isinstance(raw_outputs, dict):
            loss_mb = raw_outputs["loss"]
            logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"])
          else:
            loss_mb = raw_outputs[0]
            logits_mb = raw_outputs[1:]

          loss = loss_mb.reduce_mean().detach().cpu()
          logits = smp_nested_concat(logits_mb)
        else:
          loss = None
          if isinstance(raw_outputs, dict):
            logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
          else:
            logits_mb = raw_outputs
          logits = smp_nested_concat(logits_mb)
      else:
        if has_labels:
          loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
          loss = loss.mean().detach()
          if isinstance(outputs, dict):
            logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
          else:
            logits = outputs[1:]
        else:
          loss = None
          if self.use_amp:
            with autocast():
              outputs = model(**inputs)
          else:
            outputs = model(**inputs)
          if isinstance(outputs, dict):
            logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
          else:
            logits = outputs
          # TODO: this needs to be fixed and made cleaner later.
          if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index - 1]

    loss_fct = CrossEntropyLoss(reduction='none')
    log_probs = -loss_fct(outputs.logits.view(-1, model.config.vocab_size), labels.view(-1)).view(*labels.shape).sum(-1)
    logits = (log_probs, )

    if prediction_loss_only:
      return (loss, None, None)

    logits = nested_detach(logits)
    if len(logits) == 1:
      logits = logits[0]

    return (loss, logits, labels)
