import json
import zipfile

from sisyphus import *

from i6_core.tools.download import DownloadJob


class AddMultiWozFilenamesToDSTCData(Job):
  def __init__(self, dataset_name, config_name, split, mwoz_file=None, num_cpu=4):
    self.dataset_name = dataset_name
    self.config_name = config_name
    self.split = split

    self.mwoz_file = mwoz_file if mwoz_file is not None else DownloadJob(
      "https://github.com/budzianowski/multiwoz/raw/master/data/MultiWOZ_2.1.zip"
    )
    self.num_cpu = num_cpu

    self.out_list_of_ids = self.output_path("list_of_ids.json")

  def run(self):
    import datasets
    dstc_dataset = datasets.load_dataset(self.dataset_name, self.config_name, split=self.split)

    with zipfile.ZipFile(self.mwoz_file.get_path(), "r") as z:
      with z.open("data.json") as f:
        mwoz = json.load(f)

    # Get all knowledge seeking turns
    real_knowledge_seeking_turns = {
      dstc_dial['turns'][-1]['text']
      for dstc_dial in dstc_dataset if dstc_dial['target']
    }
    knowledge_seeking_turns = real_knowledge_seeking_turns.copy()

    def map_func(dstc_dial):
      mwoz_id = None
      longest_match = 0
      longest_match_num_snippets = 0

      for mwoz_key, mwoz_dial in mwoz.items():
        mwoz_turns = iter(mwoz_dial['log'])
        dstc_turns = iter(dstc_dial['turns'])
        match_length = 0
        num_snippets = 0

        try:
          curr_mwoz_turn = next(mwoz_turns)
          curr_dstc_turn = next(dstc_turns)
          while True:
            if curr_mwoz_turn['text'].strip() != curr_dstc_turn['text'].strip():
              if curr_dstc_turn['text'] in knowledge_seeking_turns:
                next(dstc_turns)
                curr_dstc_turn = next(dstc_turns)
                num_snippets += 1
                continue
              break
            match_length += 1
            curr_mwoz_turn = next(mwoz_turns)
            curr_dstc_turn = next(dstc_turns)
        except Exception:
          # Reached end of dstc turns
          mwoz_id = mwoz_key
          break
        if longest_match < match_length:
          longest_match = match_length
          longest_match_num_snippets = num_snippets

      if mwoz_id is None:
        extra_knowledge_seeking_turn = dstc_dial['turns'][longest_match + longest_match_num_snippets * 2]['text']
        print(extra_knowledge_seeking_turn)
        knowledge_seeking_turns.add(extra_knowledge_seeking_turn)
        return map_func(dstc_dial)
        
      return {'mwoz_id': mwoz_id}

    dstc_dataset = dstc_dataset.map(map_func, batched=False, num_proc=self.num_cpu)

    with open(self.out_list_of_ids.get_path(), 'w') as fp:
      json.dump(dstc_dataset['mwoz_id'], fp)

  def tasks(self):
    yield Task('run', rqmt={'cpu': self.num_cpu, 'mem': 2 * self.num_cpu, 'time': 2})
