import re
from copy import deepcopy

import inflect
import numpy as np
import torch
from nltk.tokenize import TweetTokenizer
from datasets import load_metric
from transformers import PretrainedConfig, EvalPrediction, AutoModelForSequenceClassification
from transformers.trainer_utils import PredictionOutput

from i6_dstc10.methods.base import Method
from i6_dstc10.preprocessing import create_concatenated_model_input, Pipeline


class BaselineDetectionMethod(Method):
  name = "baseline_detection"

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.config.num_labels = 2

    self.metrics = [
      load_metric(metric) for metric in ['precision', 'recall', 'f1']
    ]

  def preprocess_features(self, features):
    if self.model_args.process_all_in_nbest_last_turn:
      if 'nbest' in features['turns'][0][-1]:
        ids = []
        input_ids = []
        labels = []
        asr_scores = []
        for id, turns, target in zip(features['id'], features['turns'], features['target']):
          label = int(target)
          for hyp in turns[-1]['nbest']:
            current_turns = deepcopy(turns)
            current_turns[-1]['text'] = hyp['hyp']

            ids.append(id)
            input_ids.append(create_concatenated_model_input(self.model_args, current_turns, self.tokenizer))
            labels.append(label)
            asr_scores.append(hyp['score'])
        return {
          'input_ids': input_ids,
          'labels': labels,
          'id': ids,
          'asr_score': asr_scores,
        }

    input_ids = [
      create_concatenated_model_input(self.model_args, turns, self.tokenizer) for turns in features['turns']
    ]
    labels = [int(t) for t in features['target']]
    return {
      'input_ids': input_ids,
      'labels': labels,
    }

  def get_model_class(self, config: PretrainedConfig):
    return AutoModelForSequenceClassification

  def compute_metrics(self, p: EvalPrediction):
    assert self.config.num_labels == 2
    prediction_ids = np.argmax(p.predictions, axis=-1)
    results = {}
    for metric in self.metrics:
      results.update(
        metric.compute(predictions=prediction_ids, references=p.label_ids)
      )
    return results

  def postprocess_predictions(self, p: PredictionOutput, dataset):
    predictions_scores = torch.softmax(torch.tensor(p.predictions), dim=-1)[:, 1].numpy()

    if self.model_args.process_all_in_nbest_last_turn and 'asr_score' in dataset.column_names:
      list_of_scores = []
      list_of_asr_scores = []
      labels = []
      prev_id = None
      for pred_score, row_id, row_asr_score, label in zip(predictions_scores, dataset['id'], dataset['asr_score'], dataset['labels']):
        if row_id != prev_id:
          prev_id = row_id
          list_of_scores.append([])
          list_of_asr_scores.append([])
          labels.append(label)
        list_of_scores[-1].append(pred_score)
        list_of_asr_scores[-1].append(row_asr_score)

      labels = np.array(labels)

      if self.model_args.detection_nbest_combination_stratey == 'max':
        target_scores = np.array([max(scores) for scores in list_of_scores])
      elif self.model_args.detection_nbest_combination_stratey == 'average':
        target_scores = np.array([sum(scores)/len(scores) for scores in list_of_scores])
      elif self.model_args.detection_nbest_combination_stratey == 'weighted_by_asr':
        target_scores = np.array([np.sum(np.array(scores) * (np.exp(np.array(asr_scores)) / np.sum(np.exp(np.array(asr_scores))))) for scores, asr_scores in zip(list_of_scores, list_of_asr_scores)])
      else:
        assert False, f'Unsupported nbest detection combination strategy: {self.model_args.detection_nbest_combination_stratey}'

      return [
        {
          "target": bool(target_score >= 0.5),
          "target_score": float(target_score),
          "target_nbest_scores": all_scores,
        }
        for target_score, all_scores in zip(target_scores, list_of_scores)
      ]

    predictions = np.argmax(p.predictions, axis=-1)
    predictions = [
      {
        "target": bool(prediction),
        "target_score": score,
      } for prediction, score in zip(predictions, predictions_scores)
    ]
    return predictions

class DetectionOnLastUserTurnMethod(BaselineDetectionMethod):
  name = "detection_single_turn"

  def preprocess_features(self, features):
    input_ids = [
      create_concatenated_model_input(self.model_args, turns[-1:], self.tokenizer) for turns in features['turns']
    ]
    
    labels = [int(t) for t in features['target']]
    return {
      'input_ids': input_ids,
      'labels': labels,
    }
