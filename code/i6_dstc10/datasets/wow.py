import itertools
import json
import os
from typing import List

import datasets
from datasets import load_dataset
from base import DSTCBase

_URL = "http://parl.ai/downloads/wizard_of_wikipedia/wizard_of_wikipedia.tgz"

class WizardOfWikipedia(DSTCBase, datasets.GeneratorBasedBuilder):

    def _info(self):

        if self.config.name == "generation":
            features = datasets.Features(
                {
                    "id": datasets.Value('string'),
                    "target": datasets.Value("bool"),
                    "knowledge": [
                        {
                            "domain": datasets.Value("string"),
                            "entity_id": datasets.Value("string"),
                            "doc_id": datasets.Value("int32"),
                            "entity_name": datasets.Value("string"),
                            "title": datasets.Value("string"),
                            "body": datasets.Value("string"),
                        }
                    ],
                    "response": datasets.Value("string"),
                    "turns": [
                        {
                            "speaker": datasets.Value("string"),
                            "text": datasets.Value("string"),
                        }
                    ],
                    "source": datasets.Value('string'),
                }
            )
        else:
            raise NotImplementedError()

        return datasets.DatasetInfo(
            description="",
            features=features,
            supervised_keys=None,
            homepage="",
            citation="",
        )

    def _map_to_dstc_format(self, dialog):
        out = []

        for i, turn in enumerate(dialog["dialog"]):
            turn["speaker"] = "S" if turn["speaker"] == "0_Wizard" else "U"
            if i > 0 and turn["speaker"] == "S":
                if turn["checked_sentence"] != {} and turn["checked_sentence"] != {'no_passages_used': 'no_passages_used'}:
                    source = [{
                        "text": turn["text"],
                        "speaker": turn["speaker"]
                    } for turn in dialog["dialog"][:i]]

                    knowledge_items = list(turn["checked_sentence"].keys())[0].split("_")
                    domain, entity_name, doc_id = dialog["chosen_topic"], " ".join(knowledge_items[1:-1]), knowledge_items[-1]
                    sample = { 
                        "target": True,
                        "knowledge": [
                        {
                            "domain": domain,
                            "entity_id": None,
                            "doc_id": int(doc_id),
                            "entity_name": entity_name,
                            "title": None,
                            "body": list(turn["checked_sentence"].values())[0]
                        }
                        ],
                        "response": turn["text"],
                        "turns": source,
                        "source": "wow"
                    }
                    out.append(sample)

        return out

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        url_to_download = _URL
        data_path = self._download_files(url_to_download, self.config.data_files, dl_manager)
        splits = ["train", "test", "val"]
        file_names = ["train.json", "test_random_split.json", "valid_random_split.json"]
        data_files = {
            split: os.path.join(file_name) for split, file_name in zip(splits, file_names)
        }

        return [
            datasets.SplitGenerator(
                name=ds_split, gen_kwargs={
                    "filepath": os.path.join(data_path, file_name),
                })
            for ds_split, file_name in zip([datasets.Split.TRAIN, datasets.Split.VALIDATION, datasets.Split.TEST], file_names)
        ]

    def _generate_examples(self, filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
        data = list(itertools.chain.from_iterable([self._map_to_dstc_format(dialog) for dialog in data]))

        for idx, sample in enumerate(data):
            sample["id"] = str(idx)
            yield idx, sample

