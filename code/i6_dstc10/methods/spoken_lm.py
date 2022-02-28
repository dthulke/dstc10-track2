import itertools
import json

import numpy as np
from datasets import load_metric, load_dataset, concatenate_datasets, Dataset, load_from_disk
from transformers import PretrainedConfig, EvalPrediction, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, AutoTokenizer
from transformers.trainer_utils import PredictionOutput

from i6_dstc10.methods.base import Method
from i6_dstc10.methods.generation import BaselineGenerationMethod, PretrainGenerationOnWizardOfWikipediaMethod
from i6_dstc10.methods.trainer_seq2seq import CustomSeq2SeqTrainer, DataCollatorForSeq2SeqForPartialInputs, CustomSeq2SeqTrainerWithPartialInputs
from i6_dstc10.models.density_ratio import DensityRatioMethodModelForConditionalGeneration
from i6_dstc10.models.spoken_lm import ShallowFusionModelForConditionalGeneration
from i6_dstc10.models.noisy_channel import NoisyChannelRerankingModelForConditionalGeneration, OnlineNoisyChannelModelForConditionalGeneration
from i6_dstc10.preprocessing import create_concatenated_model_input, Pipeline, process_knowledge
from i6_dstc10.methods.asr_errors import SimpleSeq2SeqDataset


class GenerationWithShallowFusionMethod(BaselineGenerationMethod):

    name = "shallow_fusion_lm"

    def get_model_class(self, config: PretrainedConfig):
        return ShallowFusionModelForConditionalGeneration

    def preprocess_features(self, features):
        return_dict = super().preprocess_features(features)
        return_dict["lm_input_ids"] = [
            create_concatenated_model_input(self.model_args, turns, self.tokenizer, knowledge=None)
            for turns in features['turns']
        ]

        return return_dict


class GenerationWithDensityRatioMethod(BaselineGenerationMethod):

    name = "density_ratio"

    def get_model_class(self, config: PretrainedConfig):
        return DensityRatioMethodModelForConditionalGeneration

    def preprocess_features(self, features):
        return_dict = super().preprocess_features(features)
        return_dict["lm_input_ids"] = [
            create_concatenated_model_input(self.model_args, turns, self.tokenizer, knowledge=None)
            for turns in features['turns']
        ]

        return return_dict


class InternalLMEstimationMethod(BaselineGenerationMethod):

    name = "internal_lm"

    def preprocess_features(self, features):
        input_ids = [
        create_concatenated_model_input(self.model_args, turns, self.tokenizer, knowledge=None)
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


class ChannelModelMethod(BaselineGenerationMethod):

    name = "channel_model"

    def preprocess_features(self, features):
        features["response"] =  [[{"speaker": "S", "text": response}] for response in features["response"]]
        input_ids = [
        create_concatenated_model_input(self.model_args, turns + response, self.tokenizer, knowledge=None)
            for turns, response in zip(features["turns"], features['response'])
        ]
        target = [
        create_concatenated_model_input(self.model_args, [], self.tokenizer, knowledge=knowledge[0])
            for turns, knowledge in zip(features['turns'], features['knowledge'])
        ]

        return_dict = {
            'input_ids': input_ids,
            'labels': target
        }

        return return_dict


class OnlineChannelModelMethod(BaselineGenerationMethod):

    name = "online_channel_model"

    def get_data_collator(self):
        return DataCollatorForSeq2SeqForPartialInputs(self.tokenizer)

    def get_trainer_class(self):
        return CustomSeq2SeqTrainerWithPartialInputs

    def preprocess_features(self, features):
        features["response"] =  [[{"speaker": "S", "text": response}] for response in features["response"]]
        input_ids = [
        create_concatenated_model_input(self.model_args, turns + response, self.tokenizer, knowledge=None)
            for turns, response in zip(features["turns"], features['response'])
        ]
        target = [
        create_concatenated_model_input(self.model_args, [], self.tokenizer, knowledge=knowledge[0])
            for turns, knowledge in zip(features['turns'], features['knowledge'])
        ]
        
        return_dict = {
            'input_ids': input_ids, 
            'labels': target 
        }

        return return_dict


class NoisyChannelModelMethod(BaselineGenerationMethod):

    name = "noisy_channel"

    def get_model_class(self, config: PretrainedConfig):
        return NoisyChannelRerankingModelForConditionalGeneration

    def preprocess_features(self, features):
        return_dict = super().preprocess_features(features)
        return_dict["lm_input_ids"] = [
            create_concatenated_model_input(self.model_args, turns, self.tokenizer, knowledge=None)
            for turns in features['turns']
        ]

        return_dict["cm_labels"] = [
            create_concatenated_model_input(self.model_args, [], self.tokenizer, knowledge=knowledge[0])
            for turns, knowledge in zip(features['turns'], features['knowledge'])
        ]

        return return_dict


class GenerationWithStyleTokenOnFinalDatasetMethod(BaselineGenerationMethod):

    name = "generation_style_token_final"

    def get_special_tokens(self):
        return [
            self.model_args.user_token,
            self.model_args.agent_token,
            self.model_args.knowledge_tag_token,
            self.model_args.knowledge_sep_token,
            "<spoken>",
            "<written>"
        ]

    def preprocess_features(self, features):
        def style_token(source):
            return "<spoken>"

        input_ids = [
            self.tokenizer.convert_tokens_to_ids([self.tokenizer.cls_token, style_token(source)]) + create_concatenated_model_input(self.model_args, turns, self.tokenizer, knowledge=knowledge[0])[1:]
            for turns, knowledge, source in zip(features['turns'], features['knowledge'], features["source"])
        ]

        return_dict = {
            'input_ids': input_ids,
        }

        if self.data_args.is_training:
            return_dict["labels"] = self.tokenizer(features["response"])["input_ids"]

        return return_dict


class GenerationWithStyleTokenMethod(BaselineGenerationMethod):

    name = "generation_style_token"

    def get_special_tokens(self):
        return [
            self.model_args.user_token,
            self.model_args.agent_token,
            self.model_args.knowledge_tag_token,
            self.model_args.knowledge_sep_token,
            "<spoken>",
            "<written>"
        ]

    def preprocess_features_and_maybe_normalize_with_special_token(self, features, style):
        style_to_token = {
            "spoken": "<spoken>",
            "written": "<written>"
        }
        style_token = style_to_token[style]
        input_ids = [
            self.tokenizer.convert_tokens_to_ids([self.tokenizer.cls_token, style_token]) + create_concatenated_model_input(self.model_args, turns, self.tokenizer, knowledge=knowledge[0])[1:]
            for turns, knowledge in zip(features['turns'], features['knowledge'])
        ]

        return_dict = {
            'input_ids': input_ids,
        }

        if self.data_args.is_training:
            return_dict["labels"] = self.tokenizer(features["response"])["input_ids"]

        return return_dict

    def preprocess_features_and_maybe_normalize_spoken(self, features):
        return self.preprocess_features_and_maybe_normalize_with_special_token(features, "spoken")

    def preprocess_features_and_maybe_normalize_written(self, features):
        return self.preprocess_features_and_maybe_normalize_with_special_token(features, "written")

    def _get_dataset(self, split, config_name=None):
        if config_name is None:
            config_name = self.data_args.dataset_config_name
        spoken_dataset = load_dataset(
            self.data_args.dataset_name,
            config_name,
            split=split,
            cache_dir=self.model_args.cache_dir,
            data_files=self.data_args.dataset_data_files,
            dataset_filter_dict=self.data_args.dataset_filter_dict
        )

        if config_name != "evaluation":
            old_eval_column_names = spoken_dataset.column_names
            spoken_dataset = spoken_dataset.map(
                self.preprocess_features_and_maybe_normalize_spoken,
                batched=True,
                remove_columns=old_eval_column_names,
            )

        if self.data_args.is_training:
            written_dataset = load_dataset(
                "/".join(self.data_args.dataset_name.split("/")[:-1] + ["dstc9_track1.py"]),
                config_name,
                split="train",
                cache_dir=self.model_args.cache_dir,
                data_files=self.data_args.dataset_data_files,
                dataset_filter_dict=self.data_args.dataset_filter_dict
            )

            old_eval_column_names = written_dataset.column_names
            written_dataset = written_dataset.map(
                self.preprocess_features_and_maybe_normalize_written,
                batched=True,
                remove_columns=old_eval_column_names,
            )
            written_dataset_val = load_dataset(
                "/".join(self.data_args.dataset_name.split("/")[:-1] + ["dstc9_track1.py"]),
                config_name,
                split="validation",
                cache_dir=self.model_args.cache_dir,
                data_files=self.data_args.dataset_data_files,
                dataset_filter_dict=self.data_args.dataset_filter_dict
            )
            old_eval_column_names = written_dataset_val.column_names
            written_dataset_val = written_dataset_val.map(
                self.preprocess_features_and_maybe_normalize_written,
                batched=True,
                remove_columns=old_eval_column_names,
            )

            return concatenate_datasets([spoken_dataset, written_dataset, written_dataset_val])
        else:
            return spoken_dataset


class GenerationWithStyleTokenAndOnlineNoisyChannelMethod(BaselineGenerationMethod):

    name = "style_token_online"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.cm_tokenizer = AutoTokenizer.from_pretrained(self.config.channel_model_tokenizer_name_or_path)
        self.lm_tokenizer = AutoTokenizer.from_pretrained(self.config.language_model_tokenizer_name_or_path)

    def get_special_tokens(self):
        return [
            self.model_args.user_token,
            self.model_args.agent_token,
            self.model_args.knowledge_tag_token,
            self.model_args.knowledge_sep_token,
            "<spoken>",
            "<written>"
        ]

    def get_model_class(self, config: PretrainedConfig):
        return OnlineNoisyChannelModelForConditionalGeneration

    def preprocess_features(self, features):
        style_token = "<spoken>"
        input_ids = [
            self.tokenizer.convert_tokens_to_ids([self.tokenizer.bos_token, style_token]) + create_concatenated_model_input(self.model_args, turns, self.tokenizer, knowledge=knowledge[0])[1:]
            for turns, knowledge in zip(features['turns'], features['knowledge'])
        ]

        return_dict = {
            'input_ids': input_ids,
        }

        if self.data_args.is_training:
            return_dict["labels"] = self.tokenizer(features["response"])["input_ids"]

        return_dict["lm_input_ids"] = [
            create_concatenated_model_input(self.model_args, turns, self.lm_tokenizer, knowledge=None)
            for turns in features['turns']
        ]

        return_dict["cm_labels"] = [
            create_concatenated_model_input(self.model_args, [], self.cm_tokenizer, knowledge=knowledge[0])
            for turns, knowledge in zip(features['turns'], features['knowledge'])
        ]

        return return_dict
        
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
        return model
