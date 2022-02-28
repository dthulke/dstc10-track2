from datasets import load_dataset, DownloadManager
from collections import defaultdict
import re
import numpy as np
from tqdm import tqdm
import json
import argparse
import sys

import numba as nb
from numba import jit, njit
from numba import types

import i6_dstc10.datasets.dstc9_track1


_BASE_URL_DSTC9 = "https://raw.githubusercontent.com/alexa/alexa-with-dstc9-track1-dataset/master"
_BASE_URL_DSTC10 = "https://raw.githubusercontent.com/alexa/alexa-with-dstc10-track2-dataset/main/task2/"


_DATA_FILES = {
    "dstc9": {
        'train': {
            'logs': f'{_BASE_URL_DSTC9}/data/train/logs.json',
            'labels': f'{_BASE_URL_DSTC9}/data/train/labels.json',
            'knowledge': f'{_BASE_URL_DSTC9}/data_eval/knowledge.json',
        },
        'val': {
            'logs': f'{_BASE_URL_DSTC9}/data/val/logs.json',
            'labels': f'{_BASE_URL_DSTC9}/data/val/labels.json',
            'knowledge': f'{_BASE_URL_DSTC9}/data/knowledge.json',
        },
        'test': {
            'logs': f'{_BASE_URL_DSTC9}/data_eval/test/logs.json',
            'labels': f'{_BASE_URL_DSTC9}/data_eval/test/labels.json',
            'knowledge': f'{_BASE_URL_DSTC9}/data_eval/knowledge.json',
        }
    },
    "dstc10": {
        'validation': {
            'logs': f'{_BASE_URL_DSTC10}/data/val/logs.json',
            'labels': f'{_BASE_URL_DSTC10}/data/val/labels.json',
            'knowledge': f'{_BASE_URL_DSTC10}/data/knowledge.json',
        },
        'test': {
            'logs': f'{_BASE_URL}/data_eval/test/logs.json',
            'labels': f'{_BASE_URL}/data_eval/test/labels.json',
            'knowledge': f'{_BASE_URL}/data_eval/knowledge.json',
        }
    }
}


def preprocess_common(text):
  text = text.lower()
  text = text.replace('0', ' zero ')
  text = text.replace('1', ' one ')
  text = text.replace('2', ' two ')
  text = text.replace('3', ' three ')
  text = text.replace('4', ' four ')
  text = text.replace('5', ' five ')
  text = text.replace('6', ' six ')
  text = text.replace('7', ' seven ')
  text = text.replace('8', ' eight ')
  text = text.replace('9', ' nine ')
  text = text.replace('.', ' ')
  text = text.replace('?', ' ')
  text = text.replace('!', ' ')
  text = text.replace('\'s', ' #s ')
  text = text.replace('\'', ' ')
  text = text.replace('#s', '\'s')
  text = text.replace(',', ' ')
  text = text.replace(';', ' ')
  text = text.replace('-', ' ')
  text = text.replace('&', ' and ')
  text = text.replace('guest house', 'guesthouse')
  text = text.replace('san francisco', ' ')
  text = text.replace('downtown', ' ')
  return text


def preprocess_dialog(dialog):
  def preprocess_turn(turn):
    turn = preprocess_common(turn)
    return re.sub(r' {2,}', ' ', turn).strip()

  return [
    preprocess_turn(turn['text']) for turn in dialog
  ]


def preprocess_entity(document):
  entity_name = document['entity_name']
  entity_name = entity_name.split(' - ')[0]
  entity_name = entity_name.split('/')[0]
  entity_name = entity_name.split(', ')[0]
  entity_name = preprocess_common(entity_name)
  return re.sub(r' {2,}', ' ', entity_name).strip()


@jit(nb.float32(types.unicode_type, types.unicode_type))
def best_match(turn, entity_name):
  grid = np.zeros((len(turn) + 1, len(entity_name) + 1), dtype=np.float32)
  grid[0, 0] = 0
  for i, c in enumerate(turn):
    grid[i + 1, 0] = 0 if c == ' ' else grid[i, 0] + 1
  for i, c in enumerate(entity_name):
    grid[0, i + 1] = grid[0, i] + 1

  for i, c_t in enumerate(turn):
    for j, c_e in enumerate(entity_name):
      if c_t == c_e:
        grid[i + 1, j + 1] = grid[i, j]
      else:
        sub_cost = grid[i, j] + 1
        del_cost = grid[i + 1, j] + 1
        ins_cost = grid[i, j + 1] + 0.2
        swap_cost = 9999999.9
        if i > 0 and j > 0 and c_t == entity_name[j - 1] and c_e == turn[i - 1]:
          swap_cost = grid[i - 1, j - 1] + 0.2

        grid[i + 1, j + 1] = min(sub_cost, del_cost, ins_cost, swap_cost)
  return np.min(grid[np.concatenate((grid[1:, 0] == 0, np.array([True]))), len(entity_name)])


def create_unigram_vocab():
  full_dataset = load_dataset(i6_dstc10.datasets.dstc9_track1.__file__, name='detection')
  vocab_count = defaultdict(int)
  for dialog in tqdm([d for s in full_dataset for d in full_dataset[s]]):
      for turn in preprocess_dialog(dialog['turns']):
        for word in turn.split():
          vocab_count[word] += 1
  return vocab_count


def get_entities_from_knowledge(knowledge_data):
  for domain, domain_knowledge in knowledge_data.items():
    for entity_id, entity_knowledge in domain_knowledge.items():
      entity_name = entity_knowledge['name']
      city = str(entity_knowledge.get('city', '*'))
      yield {
        'id': f"{domain}__{entity_id}",
        'domain': domain,
        'city': city,
        'entity_id': entity_id,
        'entity_name': entity_name,
      }


def entity_tracking(logs, labels, knowledge, rare_word_factor=1.5, edits_threshold=0.4):
  vocab_count = create_unigram_vocab()
  entity_dataset = get_entities_from_knowledge(knowledge)

  def preprocess_entity_and_keep_rare_words(entity):
    entity_words = entity.split()
    try:
      min_count = min([vocab_count[word] for word in entity_words if vocab_count[word] > 0])
    except Exception:
      return None
    while True:
      if len(entity_words) == 1:
        break
      counts = [vocab_count[word] for word in entity_words]
      if counts[0] > rare_word_factor * min_count or counts[0] == 0:
        entity_words = entity_words[1:]
      elif counts[-1] > rare_word_factor * min_count or counts[-1] == 0:
        entity_words = entity_words[:-1]
      else:
        break
    return entity_words

  def split_entity_name(entity_name):
    return re.split(r' (and|\'s) ', f" {entity_name} ")

  def check_entity_in_dial(entity_name, processed_dialog):
    found = False
    orig_entity_name = entity_name
    for sub_name in entity_name:
      if sub_name is None or sub_name.strip() == '':
        continue
      entity_words = preprocess_entity_and_keep_rare_words(sub_name.strip())
      if entity_words is None:
        continue
      entity_name = " ".join(entity_words)
      min_edits = min(best_match(turn, entity_name) for turn in processed_dialog)
      if min_edits / len(entity_name) <= edits_threshold:
        found = True
      else:
        for entity_name in entity_words:
          if len(entity_name) > 5:
            min_edits = min(best_match(turn, entity_name) for turn in processed_dialog)
            if min_edits / len(entity_name) <= edits_threshold:
              found = True
              break
    return found

  preprocessed_entities = [x for x in [
    (entity, split_entity_name(preprocess_entity(entity))) for entity in entity_dataset if entity['domain'] not in {'train', 'taxi'}
  ] if x[1] is not None]

  for log, label in tqdm(zip(logs, labels), total=len(logs)):
    label['entity_candidates'] = []
    if not label['target']:
      continue
    dialog_turns = preprocess_dialog(log)
    for entity, entity_name in preprocessed_entities:
      if check_entity_in_dial(entity_name, dialog_turns):
        label['entity_candidates'].append(entity)
    if 'knowledge' in label and len(label['knowledge']) == 1:
      if label['knowledge'][0]['domain'] not in {'train', 'taxi'}:
        assert int(label['knowledge'][0]['entity_id']) in [int(c['entity_id']) for c in label['entity_candidates']]

  return logs, labels, knowledge

def main(argv):
  print(argv)
  parser = argparse.ArgumentParser()

  parser.add_argument('--split', dest='split', action='store', metavar='DATASET',
                      choices=['train', 'validation', 'test'], required=True, help='The dataset to analyze')
  parser.add_argument('--dataset', dest='dataset', action='store', metavar='PATH', required=True,
                      help='Will look for corpus in <dataroot>/<dataset>/...')
  parser.add_argument('--out_logs', dest='out_logs', action='store', metavar='JSON_FILE', required=True,
                      help='File containing output JSON')
  parser.add_argument('--out_labels', dest='out_labels', action='store', metavar='JSON_FILE', required=True,
                      help='File containing output JSON')
  parser.add_argument('--out_knowledge', dest='out_knowledge', action='store', metavar='JSON_FILE', required=True,
                      help='File containing output JSON')
  parser.add_argument('--dataset_data_files', dest='dataset_data_files', default=None)

  args = parser.parse_args()

  data_files = _DATA_FILES[args.dataset]
  if args.dataset_data_files is not None:
    with open(args.dataset_data_files, "r") as f:
      new_data_files = json.load(f)

    for split, update_dict in new_data_files.items():
      for key, value in update_dict.items():
        data_files[split][key] = value

  dm = DownloadManager()

  data_files = dm.download(data_files)[args.split]

  data = {}
  for key, path in data_files.items():
    with open(path, "r") as f:
      data[key] = json.load(f)

  logs, labels, knowledge = entity_tracking(data["logs"], data["labels"], data["knowledge"])

  for path, obj in zip([args.out_logs, args.out_labels, args.out_knowledge], [logs, labels, knowledge]):
    with open(path, "w") as f:
      json.dump(obj, f, indent=2)


if __name__ == "__main__":
  main(sys.argv)
