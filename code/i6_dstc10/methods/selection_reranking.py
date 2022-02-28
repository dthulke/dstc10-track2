import itertools
import random
import re
from collections import defaultdict

import numpy as np
import torch
from datasets import load_metric, load_dataset, arrow_dataset
from torch.utils.data import Dataset
from transformers import PretrainedConfig, EvalPrediction, AutoModelForSequenceClassification, \
  AutoModelForMultipleChoice, RobertaForSequenceClassification
from transformers.trainer_utils import PredictionOutput

from i6_dstc10.arguments import ModelArguments
from i6_dstc10.knowledge_utils import build_knowledge_document_register
from i6_dstc10.methods.base import Method
from i6_dstc10.models.selection_reranking import RobertaForReranking
from i6_dstc10.preprocessing import create_concatenated_dialog_knowledge_input, \
  process_input, process_knowledge, create_input_for_reranking


class SelectionRerankingDataset(Dataset):
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
    data_scores = []
    for query_idx, query_item in enumerate(self.query_dataset):
      if self.args.selection_level in ["all", "domain", "domain_entity"]:
        data_info = list(range(len(self.documents_dataset)))
      elif self.args.selection_level == 'entity':
        if isinstance(query_item['domain'], str):
          data_info = list(self.document_register[query_item['domain']].values())
        else:
          data_info_with_scores = [(doc, score) for domain, score in zip(query_item['domain'], query_item['score']) for doc in self.document_register[domain].values()]
          data_info, scores = zip(*data_info_with_scores)
          data_scores.extend(scores)
      elif self.args.selection_level == 'document':
        if isinstance(query_item['domain'], str):
          data_info = list(self.document_register[query_item['domain']][query_item['entity_id']].values())
        else:
          data_info_with_scores = [
            (doc, score) for domain, entity, score in zip(query_item['domain'], query_item['entity_id'], query_item['score'])
            for doc in self.document_register[domain][entity].values()
          ]
          data_info, scores = zip(*data_info_with_scores)
          data_scores.extend(scores)
      else:
        assert False
      data_infos.extend(list(zip(itertools.cycle([query_idx]), data_info)))
      query_slices.append((query_slice_counter, query_slice_counter + len(data_info)))
      query_slice_counter += len(data_info)
    self.data_infos = data_infos
    self.query_slices = query_slices
    self.data_scores = data_scores

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
      possible_domains = self.document_register.keys()
      negative_domain = random.choice(list(possible_domains - {query_item['domain']}))
    else:
      negative_domain = query_item['domain']

    if self.args.selection_level in ['domain']:
      negative_entity = None
    elif negative_sample_level in ['entity', 'domain']:
      possible_entities = self.document_register[negative_domain].keys()
      if negative_sample_level == 'entity':
        negative_entity = random.choice(list(possible_entities - {query_item['entity_id']}))
      elif negative_sample_level == 'domain':
        negative_entity = random.choice(list(possible_entities))
      else:
        assert False
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

    return {
      'domain': negative_domain,
      'entity_id': negative_entity,
      'doc_id': negative_doc,
    }

  def __getitem__(self, index):
    query_index = index
    query_item = self.query_dataset[query_index]

    correct_index = None
    if self.train_mode:
      # Sample from candidates based on score
      correct_index = 0
      num_random_samples = self.args.selection_reranking_topk - int(query_item['target'])
      all_candidates = query_item['knowledge_preds']
      if query_item['target']:
        ref_knowledge = query_item['knowledge'][0]
        all_candidates = [c for c in all_candidates if c['domain'] != ref_knowledge['domain'] and c['entity_id'] != ref_knowledge['entity_id'] and c['doc_id'] != ref_knowledge['doc_id']]

      if len(all_candidates) >= num_random_samples:
        scores = np.array([k['score'] for k in all_candidates])
        scores = scores / np.sum(scores)

        candidates = list(np.random.choice(all_candidates, size=num_random_samples, replace=False, p=scores))
      else:
        candidates = all_candidates + [
          self._sample_negative(query_item['knowledge'][0], random.randrange(1, self._get_number_of_documents_per_sample()))
          for _ in range(num_random_samples - len(all_candidates))
        ]
      if query_item['target']:
        ref_index = random.randrange(0, self.args.selection_reranking_topk)
        candidates.insert(ref_index, query_item['knowledge'][0])
        correct_index = ref_index + int(self.args.selection_reranking_include_non_ks)
      if not self.args.selection_reranking_include_non_ks:
        assert query_item['target']
    else:
      # Take n best samples
      candidates = query_item['knowledge_preds'][:self.args.selection_reranking_topk]
      if len(candidates) < self.args.selection_reranking_topk:
        candidates += [
          self._sample_negative(query_item['knowledge_preds'][0], 1)  # Sample domain negative
          for _ in range(self.args.selection_reranking_topk - len(candidates))
        ]

    query_input_ids = query_item['input_ids']
    document_input_ids = [
      self.documents_dataset[self._get_document_index(candidate_item['domain'], candidate_item['entity_id'], candidate_item['doc_id'])]['input_ids']
      for candidate_item in candidates
    ]
    input_ids, start_indices = create_input_for_reranking(self.args, self.tokenizer, query_input_ids, document_input_ids)

    return {
      'input_ids': input_ids,
      **({
        'labels': correct_index,
      } if correct_index is not None else {}),
      **({
        'candidate_positions': start_indices,
      } if self.args.selection_reranking_model_type == 'multiple_choice' else {})
    }

  def __len__(self):
    return len(self.query_dataset)


class SelectionRerankingMethod(Method):
  name = "selection_reranking"

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.config.num_labels = \
      self.model_args.selection_reranking_topk + int(self.model_args.selection_reranking_include_non_ks)

    self.metrics = [
      load_metric(metric) for metric in ['accuracy']
    ]

  def preprocess_features(self, features):
    input_ids = [
      process_input(self.model_args, turns, self.tokenizer) for turns in features['turns']
    ]

    data = {
      'input_ids': input_ids,
      'id': features['id'],
      'target': features['target'],
      'knowledge': features['knowledge'],
      'knowledge_preds': features['knowledge_preds'],
    }

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
    if self.model_args.selection_reranking_model_type == 'seq_class':
      return RobertaForSequenceClassification
    elif self.model_args.selection_reranking_model_type == 'multiple_choice':
      return RobertaForReranking
    assert False

  def compute_metrics(self, p: EvalPrediction):
    # TODO better metric (i.e. take the best one)
    prediction_ids = np.argmax(p.predictions, axis=-1)
    results = {}
    for metric in self.metrics:
      results.update(
        metric.compute(predictions=prediction_ids, references=p.label_ids)
      )
    return results

  def postprocess_predictions(self, p: PredictionOutput, dataset):
    # TODO avoid that this is run twice
    scores = torch.softmax(torch.tensor(p.predictions), -1)

    def get_documents(query_idx):
      query_item = dataset.query_dataset[query_idx]
      argmax = scores[query_idx].argmax()
      if self.model_args.selection_reranking_include_non_ks and argmax == 0:
        return {
          'target': False,
        }

      candidates = query_item['knowledge_preds'][:self.model_args.selection_reranking_topk]
      for c, s in zip(candidates, scores[query_idx][int(self.model_args.selection_reranking_include_non_ks):]):
        c['score'] = s.item()

      sorted_candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)

      return {
        'knowledge': sorted_candidates,
        'target': True,
      }

    return list(map(get_documents, range(len(dataset.query_dataset))))

  def _get_dataset(self, split, config_name=None):
    query_dataset = super()._get_dataset(split, config_name=config_name)

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
      dataset_filter_dict=self.data_args.dataset_filter_dict,
    )
    document_register = build_knowledge_document_register(document_dataset)
    old_eval_column_names = document_dataset.column_names
    document_dataset = document_dataset.map(
      self.preprocess_documents,
      batched=False,
      remove_columns=old_eval_column_names,
    )

    return SelectionRerankingDataset(
      self.model_args,
      self.tokenizer,
      query_dataset,
      document_dataset,
      document_register,
      self.data_args.is_training,
    )
