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
from typing import List, Optional

from .base import DSTCBase

import datasets

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

_HOMEPAGE = "https://github.com/alexa/alexa-with-dstc10-track2-dataset"


_BASE_URL = "https://raw.githubusercontent.com/alexa/alexa-with-dstc10-track2-dataset/main/task2/"
_URLs = {
    'val': {
        'logs': f'{_BASE_URL}/data/val/logs.json',
        'labels': f'{_BASE_URL}/data/val/labels.json',
        'knowledge': f'{_BASE_URL}/data/knowledge.json',
    },
    'test': {
        'logs': f'{_BASE_URL}/data/test/logs.json',
        'labels': f'{_BASE_URL}/data/test/labels.json',
        'knowledge': f'{_BASE_URL}/data/knowledge.json',
    }
}


class DSTC10Track2(DSTCBase, datasets.GeneratorBasedBuilder):

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        urls_to_download = _URLs
        downloaded_files = self._download_files(urls_to_download, self.config.data_files, dl_manager)

        return [
            datasets.SplitGenerator(name=ds_split, gen_kwargs=downloaded_files[split])
            for ds_split, split in (
                (datasets.Split.VALIDATION, 'val'),
                (datasets.Split.TEST, 'test'),
            )
        ]
