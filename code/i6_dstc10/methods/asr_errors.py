import os

from i6_dstc10.methods.base import Method

import torch
from transformers import AutoModelForSeq2SeqLM, EvalPrediction, PretrainedConfig, DataCollatorForSeq2Seq
from datasets import load_dataset
from torch.utils.data import Dataset

class SimpleSeq2SeqDataset(Dataset):

    def __init__(self, tokenizer, examples=None):
        self.tokenizer = tokenizer

        if examples is None:
            examples = []
        self.examples = examples

    def __getitem__(self, index):
        example = self.examples[index]

        return {
            "input_ids": example["source"],
            "labels": example["target"]
        }

    def __len__(self):
        return len(self.examples)


class Seq2SeqOnSpokenDataMethod(Method):
    """
    Superclass for sequence-to-sequence modeling on the spoken data subsets
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_model_class(self, config: PretrainedConfig):
        return AutoModelForSeq2SeqLM

    def get_data_collator(self):
        return DataCollatorForSeq2Seq(self.tokenizer)

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

    def preprocess_features():
        pass

    def _get_dataset(self, split):
        examples = []

        dataset_root = self.data_args.dataset_name
        sf_spoken_human = load_dataset(
            os.path.join(dataset_root, "dstc9_track1.py"),
            "generation",
            split="test",
            dataset_filter_dict={"source": "sf_spoken"}
        )

        sf_spoken_asr = load_dataset(
            os.path.join(dataset_root, "dstc10_track2.py"),
            "generation",
            split="validation",
        )

        for source, target in zip(sf_spoken_human, sf_spoken_asr):
            for source_turn, target_turn in zip(source["turns"], target["turns"]):
                if source_turn["speaker"] == "U":
                    for target_hyp in target_turn["nbest"]:
                        if target_hyp["hyp"] != source_turn["text"]:
                            source = self.tokenizer(source)["input_ids"]
                            target = self.tokenizer(target_hyp["hyp"])["input_ids"]
                            examples.append({
                                "source": source,
                                "target": target
                            })

        train_dataset = SimpleSeq2SeqDataset(self.tokenizer, examples=examples)

        return train_dataset


class AsrErrorSimulationMethod(Seq2SeqOnSpokenDataMethod):

    name = "asr_error_simulation"


class AsrErrorCorrectionMethod(Seq2SeqOnSpokenDataMethod):

    name = "asr_error_correction"

