# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""
DSTC9 Track 1 - Beyond Domain APIs: Task-oriented Conversational Modeling with Unstructured Knowledge Access - Dataset
"""

from __future__ import absolute_import, division, print_function

import json
from typing import List, Optional

import datasets

from .base import DSTCBase


logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@article{kim2020domain,
  title={Beyond Domain APIs: Task-oriented Conversational Modeling with Unstructured Knowledge Access},
  author={Seokhwan Kim and Mihail Eric and Karthik Gopalakrishnan and Behnam Hedayatnia and Yang Liu and Dilek Hakkani-Tur},
  journal={arXiv preprint arXiv:2006.03533}
  year={2020}
}
"""

_DESCRIPTION = """\

"""

_HOMEPAGE = "https://github.com/alexa/alexa-with-dstc9-track1-dataset"


_BASE_URL = "https://raw.githubusercontent.com/alexa/alexa-with-dstc9-track1-dataset/master"
_URLs = {
    'train': {
        'logs': f'{_BASE_URL}/data/train/logs.json',
        'labels': f'{_BASE_URL}/data/train/labels.json',
        'knowledge': f'{_BASE_URL}/data/knowledge.json',
    },
    'val': {
        'logs': f'{_BASE_URL}/data/val/logs.json',
        'labels': f'{_BASE_URL}/data/val/labels.json',
        'knowledge': f'{_BASE_URL}/data/knowledge.json',
    },
    'test': {
        'logs': f'{_BASE_URL}/data_eval/test/logs.json',
        'labels': f'{_BASE_URL}/data_eval/test/labels.json',
        'knowledge': f'{_BASE_URL}/data_eval/knowledge.json',
    }
}

class DSTC9Track1(DSTCBase, datasets.GeneratorBasedBuilder):


    def _info(self):

        if self.config.name == "detection":
            features = datasets.Features(
                {
                    "id": datasets.Value('string'),
                    "target": datasets.Value("bool"),
                    "turns": [
                        {
                            "speaker": datasets.Value("string"),
                            "text": datasets.Value("string"),
                        }
                    ],
                    "source": datasets.Value('string'),
                }
            )
        elif self.config.name == "selection":
            features = datasets.Features(
                {
                    "id": datasets.Value('string'),
                    "target": datasets.Value("bool"),
                    "turns": [
                        {
                            "speaker": datasets.Value("string"),
                            "text": datasets.Value("string"),
                        }
                    ],
                    "knowledge": [
                        {
                            "domain": datasets.Value("string"),
                            "entity_id": datasets.Value("string"),
                            "doc_id": datasets.Value("int32"),
                            "score": datasets.Value("float"),
                        }
                    ],
                    "entity_candidates": [{
                        "id": datasets.Value('string'),
                        "domain": datasets.Value("string"),
                        "city": datasets.Value("string"),
                        "entity_id": datasets.Value("string"),
                        "entity_name": datasets.Value("string"),
                    }],
                    "source": datasets.Value('string'),
                }
            )
        elif self.config.name == "selection_search":
            features = datasets.Features(
                {
                    "id": datasets.Value('string'),
                    "target": datasets.Value("bool"),
                    "turns": [
                        {
                            "speaker": datasets.Value("string"),
                            "text": datasets.Value("string"),
                        }
                    ],
                    "entity_candidates": [{
                        "id": datasets.Value('string'),
                        "domain": datasets.Value("string"),
                        "city": datasets.Value("string"),
                        "entity_id": datasets.Value("string"),
                        "entity_name": datasets.Value("string"),
                    }],
                    "source": datasets.Value('string'),
                }
            )
        elif self.config.name == "selection_reranking":
            features = datasets.Features(
                {
                    "id": datasets.Value('string'),
                    "target": datasets.Value("bool"),
                    "turns": [
                        {
                            "speaker": datasets.Value("string"),
                            "text": datasets.Value("string"),
                        }
                    ],
                    "knowledge": [
                        {
                            "domain": datasets.Value("string"),
                            "entity_id": datasets.Value("string"),
                            "doc_id": datasets.Value("int32"),
                            "score": datasets.Value("float"),
                        }
                    ],
                    "knowledge_preds": [
                        {
                            "domain": datasets.Value("string"),
                            "entity_id": datasets.Value("string"),
                            "doc_id": datasets.Value("int32"),
                            "score": datasets.Value("float"),
                        }
                    ],
                    "entity_candidates": [{
                        "id": datasets.Value('string'),
                        "domain": datasets.Value("string"),
                        "city": datasets.Value("string"),
                        "entity_id": datasets.Value("string"),
                        "entity_name": datasets.Value("string"),
                    }],
                    "source": datasets.Value('string'),
                }
            )
        elif self.config.name == "generation_search":
            features = datasets.Features(
                {
                    "id": datasets.Value('string'),
                    "target": datasets.Value("bool"),
                    "turns": [
                        {
                            "speaker": datasets.Value("string"),
                            "text": datasets.Value("string"),
                        }
                    ],
                    "knowledge": [
                        {
                            "domain": datasets.Value("string"),
                            "entity_id": datasets.Value("string"),
                            "doc_id": datasets.Value("int32"),
                            "entity_name": datasets.Value("string"),
                            "title": datasets.Value("string"),
                            "body": datasets.Value("string"),
                            "score": datasets.Value("float"),
                        }
                    ],
                    "entity_candidates": [{
                        "id": datasets.Value('string'),
                        "domain": datasets.Value("string"),
                        "city": datasets.Value("string"),
                        "entity_id": datasets.Value("string"),
                        "entity_name": datasets.Value("string"),
                    }],
                    "source": datasets.Value('string'),
                }
            )
        elif self.config.name in ["evaluation", "generation"]:
            features = datasets.Features(
                {
                    "id": datasets.Value('string'),
                    "target": datasets.Value("bool"),
                    "turns": [
                        {
                            "speaker": datasets.Value("string"),
                            "text": datasets.Value("string"),
                        }
                    ],
                    "knowledge": [
                        {
                            "domain": datasets.Value("string"),
                            "entity_id": datasets.Value("string"),
                            "doc_id": datasets.Value("int32"),
                            "entity_name": datasets.Value("string"),
                            "title": datasets.Value("string"),
                            "body": datasets.Value("string"),
                            "score": datasets.Value("float"),
                        }
                    ],
                    "entity_candidates": [{
                        "id": datasets.Value('string'),
                        "domain": datasets.Value("string"),
                        "city": datasets.Value("string"),
                        "entity_id": datasets.Value("string"),
                        "entity_name": datasets.Value("string"),
                    }],
                    "response": datasets.Value("string"),
                    "source": datasets.Value('string'),
                }
            )
        elif self.config.name in ["knowledge", "knowledge_entities", "knowledge_domains"]:
            features = datasets.Features(
                {
                    "id": datasets.Value('string'),
                    "domain": datasets.Value("string"),
                    **({
                        "city": datasets.Value("string"),
                        "entity_id": datasets.Value("string"),
                        "entity_name": datasets.Value("string"),
                    } if self.config.name in ["knowledge", "knowledge_entities"] else {}),
                    **({
                        "doc_id": datasets.Value("int32"),
                        "title": datasets.Value("string"),
                        "body": datasets.Value("string"),
                    } if self.config.name == "knowledge" else {}),
                }
            )
        else:
            assert False, f"Unexpected config name: {self.config.name}"

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )


    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        urls_to_download = _URLs
        downloaded_files = self._download_files(urls_to_download, self.config.data_files, dl_manager)

        return [
            datasets.SplitGenerator(name=ds_split, gen_kwargs=downloaded_files[split])
            for ds_split, split in (
                (datasets.Split.TRAIN, 'train'),
                (datasets.Split.VALIDATION, 'val'),
                (datasets.Split.TEST, 'test')
            )
        ]
