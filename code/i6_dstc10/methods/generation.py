import itertools
import json

from datasets import load_metric, load_dataset, concatenate_datasets
from transformers import PretrainedConfig, EvalPrediction, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from transformers.trainer_utils import PredictionOutput

from i6_dstc10.methods.asr_errors import SimpleSeq2SeqDataset
from i6_dstc10.methods.base import Method
from i6_dstc10.methods.trainer_seq2seq import CustomSeq2SeqTrainer, DataCollatorForSeq2SeqWithLMInputs
from i6_dstc10.preprocessing import create_concatenated_model_input


class BaselineGenerationMethod(Method):
  name = "baseline_generation"

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self.metrics = [
      load_metric(metric) for metric in ['sacrebleu']
    ]

  def preprocess_features(self, features):
    input_ids = [
      create_concatenated_model_input(self.model_args, turns, self.tokenizer, knowledge=knowledge[0])
      for turns, knowledge in zip(features['turns'], features['knowledge'])
    ]

    return_dict = {
      'input_ids': input_ids,
      'id': features['id'],
      'target': features['target']
    }

    if self.data_args.is_training:
      return_dict["labels"] = self.tokenizer(features["response"])["input_ids"]

    return return_dict

  def get_model_class(self, config: PretrainedConfig):
    return AutoModelForSeq2SeqLM

  def get_trainer_class(self):
    return CustomSeq2SeqTrainer

  def get_data_collator(self):
    return DataCollatorForSeq2SeqWithLMInputs(self.tokenizer)

  def compute_metrics(self, p: EvalPrediction):
    predictions_strings = self.tokenizer.batch_decode(p.predictions, skip_special_tokens=True)
    label_ids = p.label_ids
    label_ids[label_ids == -100] = self.tokenizer.pad_token_id
    reference_strings = [[ref] for ref in self.tokenizer.batch_decode(p.label_ids, skip_special_tokens=True)]
    results = {}
    for metric in self.metrics:
      results.update(
        metric.compute(predictions=predictions_strings, references=reference_strings)
      )
    return results

  def postprocess_predictions(self, p: PredictionOutput, dataset):
    generations = self.tokenizer.batch_decode(p.predictions, skip_special_tokens=True)

    full_dataset = self.get_full_dataset(config_name="evaluation")

    idx = 0
    out = []

    for sample in full_dataset:
      item = {}
      item["target"] = sample["target"]

      if idx < len(generations) and item["target"]:
        item["response"] = generations[idx]
        for key in ["title", "body"]:
          for knowledge_item in sample["knowledge"]:
            knowledge_item.pop(key)
        item["knowledge"] = sample["knowledge"]
        idx += 1

      out.append(item)

    return out
