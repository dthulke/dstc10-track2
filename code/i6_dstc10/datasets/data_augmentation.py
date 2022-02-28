import argparse
import itertools
import json
import os
import re
import sys

import numpy as np
import torch
from tqdm import tqdm
from datasets import DownloadManager, load_dataset

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

_BASE_URL_DSTC9 = "https://raw.githubusercontent.com/alexa/alexa-with-dstc9-track1-dataset/master"
_BASE_URL_DSTC10 = "https://raw.githubusercontent.com/alexa/alexa-with-dstc10-track2-dataset/main/task2/"

# TODO: How to abstract this?
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


def _truncate_sequences(sequences, max_length):
  words_to_cut = sum(list(map(len, sequences))) - max_length
  if words_to_cut <= 0:
    return sequences

  while words_to_cut > len(sequences[0]):
    words_to_cut -= len(sequences[0])
    sequences = sequences[1:]

  sequences[0] = sequences[0][words_to_cut:]
  return sequences


def _has_slots_of_domain(dialog, domain):
    pattern = re.compile(f"{domain}-", re.IGNORECASE)
    
    for turn in dialog["log"]:
        if "span_info" in turn and any([pattern.match(slot_value[0]) for slot_value in turn["span_info"]]):
            return True
    
    return False


def _last_dialog_domain(dialog, domains=None):
    if domains is None:
        domains = ["hotel", "attraction", "taxi", "restaurant", "train"]
    
    for turn in dialog["log"][::-1]:
        if "span_info" in turn:
            for slot in turn["span_info"]:
                for domain in domains:
                    if domain in slot[0].lower():
                        return domain
    
    return None


def _entity_name_in_dialog_with_domain(dialog, domain):
    for turn in dialog["log"][::-1]:
        if "span_info" in turn:
            for slot in turn["span_info"]:
                if domain in slot[0].lower() and slot[1] == "Name":
                    return slot[2]
    
    return None


def _ends_with_slots_of_domain(dialog, domain):
    pattern = re.compile(f"{domain}-", re.IGNORECASE)
    
    for turn in dialog["log"][::-1]:
        if "span_info" in turn:
            return any([pattern.match(slot_value[0]) for slot_value in turn["span_info"]])
    
    return False


def _replace_entities_and_reformat(log, knowledge):
    sampled_entity = np.random.choice(list(knowledge["attraction"].keys()))
    entity_name = knowledge["attraction"][sampled_entity]["name"]

    new_log = []

    entity_replaced = False
    
    for turn in log:
        for slot in turn["span_info"]:
            if slot[1] == "Name":
                to_replace = slot[2] # corresponds to the entity name
                turn["text"] = turn["text"].replace(to_replace, entity_name)
                entity_replaced = True
    
        new_turn = {
            "text": turn["text"],
            "speaker": "U" if turn["metadata"] == {} else "S"
        }
        new_log.append(new_turn)

    return new_log, sampled_entity, entity_replaced


def _replace_last_user_utterance(dialog, entity_knowledge):
    for turn in dialog[::-1]:
        if turn["speaker"] == "U":
            snippet_key = np.random.choice(list(entity_knowledge["docs"].keys()))
            turn["text"] = entity_knowledge["docs"][snippet_key]["title"]
            break
        else:
            dialog.pop()

    return dialog, snippet_key


def _label_from_snippet(snippet_key, sampled_entity, domain="attraction"):
    return {
        "target": True,
        "knowledge": [{
            "domain": domain,
            "entity_id": sampled_entity,
            "doc_id": int(snippet_key)
        }],
        "source": "simulated"
    }


def _prepare_knowledge(knowledge):
    snippet = knowledge
    knowledge_sep_token = "<knowledge_sep>"
    join_str = " %s " % knowledge_sep_token
    knowledge_parts = [snippet["domain"], snippet['entity_name'], snippet['title'], snippet['body']]
    knowledge_to_use = join_str.join([k for k in knowledge_parts if k is not None])
    return knowledge_to_use


def _process_knowledge(tokenizer, knowledge):
    knowledge_text = _prepare_knowledge(knowledge)
    tokenized_knowledge = tokenizer(knowledge_text, add_special_tokens=False)["input_ids"]
    truncated_knowledge = tokenized_knowledge[:512]
    return truncated_knowledge


def _wrap_with_special_tokens(tokenizer, input_ids):
  return [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]


def _process_input(turns, tokenizer):
  history = [
    tokenizer.convert_tokens_to_ids(tokenizer.tokenize(turn["text"]))
    for turn in turns
  ]

  # apply history threshold at an utterance-level (a large value can be used to nullify its effect)
  truncated_history = history[12:]

  # perform token-level truncation of history from the left
  truncated_history = _truncate_sequences(truncated_history, 384)
  truncated_speaker = [turn['speaker'] for turn in turns[-len(truncated_history):]]

  # Add speaker tags to the history and response
  # the response is always by speaker2
  # and the current_turn always by speaker1
  truncated_history = [
    tokenizer.convert_tokens_to_ids(tokenizer.tokenize("<user>" if s == 'U' else "<agent>")) + t
    for i, t, s in zip(range(len(truncated_history)), truncated_history, truncated_speaker)
  ]

  return list(itertools.chain.from_iterable(truncated_history))


def _create_concatenated_dialog_knowledge_input(tokenizer, dialog_input_ids, knowledge_input_ids):
  return [tokenizer.cls_token_id] + knowledge_input_ids + tokenizer.convert_tokens_to_ids(
      ["<knowledge_tag>"]) + dialog_input_ids + [tokenizer.sep_token_id]


def _create_concatenated_model_input(turns, tokenizer, knowledge=None):
  input_ids = _process_input(turns, tokenizer)

  if knowledge is not None:
    truncated_knowledge = _process_knowledge(tokenizer, knowledge)

  if knowledge is None:
    input_ids = _wrap_with_special_tokens(tokenizer, input_ids)
  else:
    input_ids = _create_concatenated_dialog_knowledge_input(tokenizer, input_ids, truncated_knowledge)

  return input_ids

def _get_dialogs_with_slots_of_domain(dialogs, domain):
    out = []
    for dialog in dialogs:
        if _ends_with_slots_of_domain(dialog, domain):
            out.append(dialog)
    return out


def augment_from_multiwoz_full_context(logs, labels, knowledge, detection=False):
    path = "" # TODO: add correct path
    with open(os.path.join(path, "data.json"), "r") as f:
        dialogs = json.load(f)
    
    domains = knowledge.keys()

    for domain in domains:

        domain_dialogs = _get_dialogs_with_slots_of_domain(dialogs.values(), domain)
        domain_knowledge = knowledge[domain]
        
        for entity_id in domain_knowledge.keys():
            entity_name = domain_knowledge[entity_id]["name"]
            for doc_id, snippet in domain_knowledge[entity_id]["docs"].items():
                new_log = []
                sampled_dialog = np.random.choice(domain_dialogs)["log"]

                for turn in sampled_dialog:
                    for slot in turn["span_info"]:
                        if slot[1] == "Name" and domain in slot[0].lower():
                            to_replace = re.compile(slot[2].replace(" 's", "'s"), re.IGNORECASE) # corresponds to the entity name
                            turn["text"] = to_replace.sub(entity_name, turn["text"])
                            entity_replaced = True
            
                    new_turn = {
                        "text": turn["text"],
                        "speaker": "U" if turn["metadata"] == {} else "S"
                    }
                    new_log.append(new_turn)

                for i, turn in enumerate(new_log[::-1]):
                    if turn["speaker"] == "S":
                        if any(["bye" in act for act in sampled_dialog[::-1][i]["dialog_act"].keys()]):
                            new_log.pop()
                        else:
                            break
                    else:
                        new_log.pop()

                snippet["domain"] = domain
                snippet["entity_name"] = entity_name
                label = _label_from_snippet(doc_id, entity_id, domain=domain)
                turns = [snippet["title"]]
                if entity_name is not None:
                    pattern = re.compile(entity_name, re.IGNORECASE)
                    turns.append(pattern.sub("it", snippet["title"]))

                for turn in set(turns):
                    logs.append(new_log + [{
                        'speaker': 'U',
                        'text': turn
                    }])
                    labels.append(label)

    return logs, labels, knowledge


def filter_selection_errors(logs, labels, knowledge, version=1):
    blacklist = ['992', '6393', '6817', '8248', '10809', '10857', '20516', '29237', '39136', '39389', '39878', '42646', '44947',
     '47617', '48418', '50175', '51788', '51934', '52241', '52243', '52244', '59613', '67348', '67509', '67949',
     '68491']
    blacklist = {int(i) for i in blacklist}

    assert(len(logs) == 71348), "Should be the original train files so that the ids match"

    logs = [log for i, log in enumerate(logs) if i not in blacklist]
    labels = [log for i, log in enumerate(labels) if i not in blacklist]

    return logs, labels, knowledge


def filter_detection_errors(logs, labels, knowledge, version=1):
    wrong_ids = {'65007', '65447', '4863', '10731', '18351', '34681', '41863', '42007', '47513', '49755', '51878',
                 '53374', '14320', '23079', '25271', '34489', '40333', '42755', '43116', '47525', '53101', '63932',
                 '22583', '44254', '53984', '65447', '348', '8372', '9761', '9766', '11058', '11353', '11494', '17041',
                 '17046', '17954', '18317', '18575', '18847', '18949', '21151', '21333', '24665', '24873', '24945',
                 '25314', '25441', '25509', '25553', '33959', '34233', '34319', '34558', '34870', '35181', '35357',
                 '35980', '40360', '40733', '41165', '42012'}
    unclear_filter = {'73', '92', '111', '142', '532', '1050', '1141'}

    if version > 1:
        new_wrong_ids = {'824', '5919', '5921', '5929', '9643'}
        wrong_entrance_fee = {'470', '629', '661', '688', '699', '985', '1043', '1113', '1176', '1177', '1251', '1332',
                              '1931', '2050', '2133', '2134', '2679', '2935', '4045', '4146', '4188', '4830', '4963',
                              '5042', '5058', '5104', '5272', '5278', '5308', '5522', '5554', '5801', '5912', '5913',
                              '6368', '6370', '6371', '6950', '6951', '6952', '7061'}
        wrong_ids |= new_wrong_ids | wrong_entrance_fee

    unclear_filter = {int(i) for i in unclear_filter}

    assert(len(logs) == 71348), "Should be the original train files so that the ids match"

    if version == 3:
        unclear_filter |= wrong_ids
    else:
        for wrong_id in wrong_ids:
            labels[int(wrong_id)]['target'] = not labels[int(wrong_id)]['target']

    logs = [log for i, log in enumerate(logs) if i not in unclear_filter]
    labels = [log for i, log in enumerate(labels) if i not in unclear_filter]

    return logs, labels, knowledge


def augment_from_multiwoz_full_context_by_simulation(model, tokenizer, logs, labels, knowledge, detection=False):
    path = "" # TODO: add correct path
    with open(os.path.join(path, "data.json"), "r") as f:
        dialogs = json.load(f)
    
    domains = knowledge.keys()

    for domain in domains:

        domain_dialogs = _get_dialogs_with_slots_of_domain(dialogs.values(), domain)
        domain_knowledge = knowledge[domain]
        
        for entity_id in domain_knowledge.keys():
            entity_name = domain_knowledge[entity_id]["name"]
            for doc_id, snippet in domain_knowledge[entity_id]["docs"].items():
                new_log = []
                sampled_dialog = np.random.choice(domain_dialogs)["log"]

                for turn in sampled_dialog:
                    for slot in turn["span_info"]:
                        if slot[1] == "Name" and domain in slot[0].lower():
                            to_replace = re.compile(slot[2].replace(" 's", "'s"), re.IGNORECASE) # corresponds to the entity name
                            turn["text"] = to_replace.sub(entity_name, turn["text"])
                            entity_replaced = True
            
                    new_turn = {
                        "text": turn["text"],
                        "speaker": "U" if turn["metadata"] == {} else "S"
                    }
                    new_log.append(new_turn)

                for i, turn in enumerate(new_log[::-1]):
                    if turn["speaker"] == "S":
                        if "bye" in sampled_dialog[::-1][i]["dialog_act"]:
                            new_log.pop()
                        else:
                            break
                    else:
                        new_log = new_log[:-1]
                        break

                snippet["domain"] = domain
                snippet["entity_name"] = entity_name
                source = _create_concatenated_model_input(new_log, tokenizer, knowledge=snippet)
                source = torch.tensor(source).unsqueeze(dim=0).to("cuda:0")
                label = _label_from_snippet(doc_id, entity_id, domain=domain)
                turns = tokenizer.batch_decode(
                    model.generate(
                        source, 
                        num_beams=12, 
                        max_length=60,
                        no_repeat_ngram_size=2, 
                        early_stopping=True, 
                        num_return_sequences=2,
                        diversity_penalty=0.3,
                        num_beam_groups=6,
                        #bad_words_ids=bad_word_ids
                        ),
                    skip_special_tokens=True
                )
                if domain == "attraction":
                    for i in range(len(turns)):
                        new = np.random.choice(["attraction", "place", "location"])
                        for d in ["restaurant", "hotel", "taxi", "train"]:
                            turns[i] = turns[i].replace(d, new)

                for turn in set(turns):
                    logs.append(new_log + [{
                        'speaker': 'U',
                        'text': turn
                    }])
                    labels.append(label)

    return logs, labels, knowledge


def _add_new_knowledge_and_return_label(snippet, knowledge, entity_name, last_domain):
    existing_doc = None
    existing_entity = None if entity_name is not None else "*"
    for entity_id, entity_info in knowledge[last_domain].items():
        if existing_entity != "*":
            if entity_info["name"].lower() == entity_name.lower():
                existing_entity = entity_id
    if existing_entity is None:
        new_entity_id = str(max([int(entity_id) for entity_id in knowledge[last_domain].keys()]) + 1)
        knowledge[last_domain][new_entity_id] = {
            "name": entity_name,
            "docs": {}
        }
        existing_entity = new_entity_id

    for doc_id, doc in knowledge[last_domain][existing_entity]["docs"].items():
        if doc["title"] == snippet["title"]:
            existing_doc = doc_id
            break

    if existing_doc is None:
        if len(knowledge[last_domain][existing_entity]["docs"].keys()) == 0:
            new_doc_id = '0'
        else:
            new_doc_id = str(max([int(doc_id) for doc_id in knowledge[last_domain][existing_entity]["docs"].keys()]) + 1)

        existing_doc = new_doc_id
        knowledge[last_domain][existing_entity]["docs"][new_doc_id] = {
            "title": snippet["title"],
            "body": snippet["body"]
        }

    label = {
        "target": True,
        "knowledge": [{
            "domain": last_domain,
            "entity_id": existing_entity,
            "doc_id": existing_doc
        }],
        "source": 'simulated',
    }
    return knowledge, label

        

def change_entity_in_snippet(logs, labels, knowledge, city=None):
    mwoz_path = "" # TODO: add path
    with open(os.path.join(mwoz_path, "data.json"), "r") as f:
        dialogs = json.load(f)

    if city is None:
        domains = knowledge.keys()
    else:
        domains = {key for key in knowledge if any(entity.get('city', None) == city for entity in knowledge[key].values())}

    for key, dialog in dialogs.items():
        last_domain = _last_dialog_domain(dialog, domains)
        if last_domain is not None:
            entity_name = None
            if last_domain not in ["train", "taxi"]:
                entity_name = _entity_name_in_dialog_with_domain(dialog, last_domain)


            if entity_name is not None or last_domain in ["train", "taxi"]:

                new_log = []

                for turn in dialog["log"]:
                    new_turn = {
                        "text": turn["text"],
                        "speaker": "U" if turn["metadata"] == {} else "S"
                    }
                    new_log.append(new_turn)

                for i, turn in enumerate(new_log[::-1]):
                    if turn["speaker"] == "S":
                        if any(["bye" in act for act in dialog["log"][::-1][i]["dialog_act"].keys()]):
                            new_log.pop()
                        else:
                            break
                    else:
                        new_log.pop()

                if city is None:
                    entity_id = np.random.choice(list(knowledge[last_domain].keys()))
                else:
                    entity_id = np.random.choice([entity_id for entity_id, entity in knowledge[last_domain].items() if
                                                  entity.get('city') == city])

                old_entity_name = knowledge[last_domain][entity_id]["name"]
                doc_id = np.random.choice(list(knowledge[last_domain][entity_id]["docs"].keys()))
                snippet = knowledge[last_domain][entity_id]["docs"][doc_id].copy()

                snippet["domain"] = last_domain
                snippet["entity_name"] = entity_name
                label = _label_from_snippet(doc_id, entity_id, domain=last_domain)
                orig_snippet = snippet.copy()
                if old_entity_name is not None:
                    pattern = re.compile(old_entity_name, re.IGNORECASE)
                    snippet["title"] = pattern.sub(entity_name, snippet["title"])
                    snippet["body"] = pattern.sub(entity_name, snippet["body"])
                
                knowledge, label = _add_new_knowledge_and_return_label(snippet, knowledge, entity_name, last_domain)
                turns = [snippet["title"]]
                if old_entity_name is not None:
                    turns.append(pattern.sub("it", orig_snippet["title"]))

                for turn in set(turns):
                    logs.append(new_log + [{
                        'speaker': 'U',
                        'text': turn
                    }])
                    labels.append(label)


    return logs, labels, knowledge


def augment_detection_data(logs, labels, knowledge):
    for _, domain in knowledge.items():
        for _, entity_dict in domain.items():
            entity_name = entity_dict["name"]
            if entity_name is not None:
                pattern = re.compile(entity_name, re.IGNORECASE)
            
            for _, doc in entity_dict["docs"].items():
                texts = [doc["title"]]
                if entity_name is not None and entity_name.lower() in doc["title"].lower():
                    text_replaced_entity = pattern.sub("it", doc["title"])
                    texts.append(text_replaced_entity)
                
                for text in texts:
                    logs.append([{
                        'speaker': 'U',
                        'text': text
                    }])
                    labels.append({"target": True, "source": "sf_written"})

    return logs, labels, knowledge


def seq2seq_on_user_utterance(model, tokenizer, logs, labels, knowledge):
    for i, dialog in enumerate(logs):
        for j, turn in enumerate(dialog):
            if turn["speaker"] == "U":
                source = torch.tensor(tokenizer(turn["text"])["input_ids"]).unsqueeze(dim=0).to("cuda:0")
                out = model.generate(source, num_beams=6)
                out = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
                logs[i][j]["text"] = out
            
    return logs, labels, knowledge


def augment_by_simulated_last_turn(model, tokenizer, logs, labels, knowledge):
    domains = list(knowledge.keys())
    for domain in domains:
        for entity_id, entity_dict in tqdm(knowledge[domain].items()):
            entity_name = entity_dict["name"]
            for doc_id, doc in entity_dict["docs"].items():
                snippet = doc
                snippet["domain"] = domain
                snippet["entity_name"] = entity_name
                
                source = _process_knowledge(tokenizer, snippet)
                source = torch.tensor(source).unsqueeze(dim=0).to("cuda:0")
                turns = tokenizer.batch_decode(
                    model.generate(
                        source, 
                        num_beams=12, 
                        max_length=60,
                        no_repeat_ngram_size=2, 
                        early_stopping=True, 
                        num_return_sequences=6,
                        diversity_penalty=0.3,
                        num_beam_groups=6,
                        ),
                    skip_special_tokens=True
                )

                if domain == "attraction":
                    for i in range(len(turns)):
                        new = np.random.choice(["attraction", "place", "location"])
                        for d in ["restaurant", "hotel", "taxi", "train"]:
                            turns[i] = turns[i].replace(d, new)

                for turn in set(turns):
                    logs.append([{
                        'speaker': 'U',
                        'text': turn
                    }])
                    labels.append({
                        "target": True,
                        "source": "sf_written",
                        "knowledge": [{
                            "domain": domain,
                            "entity_id": entity_id,
                            "doc_id": doc_id
                        }]
                    })

    return logs, labels, knowledge


def augment_by_adding_validation_data(logs, labels, knowledge, additional_data=None):
    dm = DownloadManager()

    if additional_data is None:
        additional_data = ['test', 'validation', 'sf_spoken_asr']

    assert set(additional_data).issubset({'test', 'validation', 'sf_spoken_asr', 'sf_written', 'sf_spoken', 'test_mwoz'})

    data_files_paths = dm.download(_DATA_FILES)
    data_files = {}
    for ds_name in data_files_paths:
        data_files[ds_name] = {}
        for split in data_files_paths[ds_name]:
            data_files[ds_name][split] = {}
            for key in data_files_paths[ds_name][split]:
                with open(data_files_paths[ds_name][split][key], "r") as f:
                    data_files[ds_name][split][key] = json.load(f)

    all_logs = logs
    all_labels = labels

    # Add source to training data
    for label in labels:
        if 'source' not in label:
            label['source'] = 'multiwoz'

    if 'validation' in additional_data:
        all_logs += data_files['dstc9']['val']['logs']
        val_labels = data_files['dstc9']['val']['labels']
        for label in val_labels:
            label['source'] = 'multiwoz'
        all_labels += val_labels
    if 'test' in additional_data:
        all_logs += data_files['dstc9']['test']['logs']
        all_labels += data_files['dstc9']['test']['labels']
    if 'sf_spoken_asr' in additional_data:
        all_logs += data_files['dstc10']['validation']['logs']
        dstc10_labels = data_files['dstc10']['validation']['labels']
        for label in dstc10_labels:
            label['source'] = 'sf_spoken_asr'
        all_labels += dstc10_labels
    if 'sf_written' in additional_data:
        test_logs = data_files['dstc9']['test']['logs']
        test_labels = data_files['dstc9']['test']['labels']
        for log, label in zip(test_logs, test_labels):
            if 'sf_written' == label['source']:
                all_logs.append(log)
                all_labels.append(label)
    if 'sf_spoken' in additional_data:
        test_logs = data_files['dstc9']['test']['logs']
        test_labels = data_files['dstc9']['test']['labels']
        for log, label in zip(test_logs, test_labels):
            if 'sf_spoken' == label['source']:
                all_logs.append(log)
                all_labels.append(label)
    if 'test_mwoz' in additional_data:
        test_logs = data_files['dstc9']['test']['logs']
        test_labels = data_files['dstc9']['test']['labels']
        for log, label in zip(test_logs, test_labels):
            if 'multiwoz' == label['source']:
                all_logs.append(log)
                all_labels.append(label)

    for i, dialog in enumerate(all_logs):
        for j, turn in enumerate(dialog):
            if not "nbest" in turn:
                all_logs[i][j]["nbest"] = []

    return all_logs, all_labels, knowledge


def main(argv):
    print(argv)
    parser = argparse.ArgumentParser()

    parser.add_argument('--split', dest='split', action='store', metavar='DATASET', choices=['train', 'validation', 'test'], required=True, help='The dataset to analyze')
    parser.add_argument('--dataset',dest='dataset',action='store', metavar='PATH', required=True,
                        help='Will look for corpus in <dataroot>/<dataset>/...')
    parser.add_argument('--out_logs',dest='out_logs',action='store',metavar='JSON_FILE',required=True,
                        help='File containing output JSON')
    parser.add_argument('--out_labels',dest='out_labels',action='store',metavar='JSON_FILE',required=True,
                        help='File containing output JSON')
    parser.add_argument('--out_knowledge',dest='out_knowledge',action='store',metavar='JSON_FILE',required=True,
                        help='File containing output JSON')
    parser.add_argument('--strategy',dest='strategy',action='store',required=True)
    parser.add_argument('--model_name_or_path',dest='model_name_or_path',default=None)
    parser.add_argument('--dataset_data_files',dest='dataset_data_files',default=None)

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

    if args.strategy == "detection":
        # Question from knowledge snippet is added as a new knowledge seeking dialog
        logs, labels, knowledge = augment_detection_data(data["logs"], data["labels"], data["knowledge"])
    elif args.strategy == "multiwoz":
        # Question from knowledge snippet is appended to a dialog sampled from multiwoz
        logs, labels, knowledge = augment_from_multiwoz_full_context(data["logs"], data["labels"], data["knowledge"])
    elif args.strategy == "multiwoz_detection":
        # Question from knowledge snippet is appended to a dialog sampled from multiwoz, entity names in the question are replaced with "it"
        logs, labels, knowledge = augment_from_multiwoz_full_context(data["logs"], data["labels"], data["knowledge"], detection=True)
    elif args.strategy.startswith("filter_detection_errors"):
        strategy_match = re.match(r'filter_detection_errors_v(\d+)', args.strategy)
        version = int(strategy_match.group(1)) if strategy_match is not None else 1
        # Removes detection samples with annotation errors
        logs, labels, knowledge = filter_detection_errors(data["logs"], data["labels"], data["knowledge"], version=version)
    elif args.strategy.startswith("filter_selection_errors"):
        strategy_match = re.match(r'filter_selection_errors_v(\d+)', args.strategy)
        version = int(strategy_match.group(1)) if strategy_match is not None else 1
        # Removes selection samples with annotation errors
        logs, labels, knowledge = filter_selection_errors(data["logs"], data["labels"], data["knowledge"], version=version)
    elif args.strategy == "simulate_turn":
        # A new user turn is generated by a model from the question of the knowledge snippet
        # the new turn is added as a new dialog (without context)
        if args.model_name_or_path is not None:
            model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path).to("cuda:0")
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
            logs, labels, knowledge = augment_by_simulated_last_turn(model, tokenizer, data["logs"], data["labels"], data["knowledge"])
        else:
            raise Exception("No model provided")
    elif args.strategy == "simulate_turn_full_context":
        # A new user turn is generated by a model from the question of the knowledge snippet and a given dialog context (sampled from multiwoz)
        # the new turn is added as a new dialog with the context
        if args.model_name_or_path is not None:
            model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path).to("cuda:0")
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
            logs, labels, knowledge = augment_from_multiwoz_full_context_by_simulation(model, tokenizer, data["logs"], data["labels"], data["knowledge"])
        else:
            raise Exception("No model provided")
    elif args.strategy == "seq2seq_on_user_utterance":
        # Applies a given seq2seq model to each user utterance in the dataset (e.g. a model to generate asr errors)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path).to("cuda:0")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        logs, labels, knowledge = seq2seq_on_user_utterance(model, tokenizer, data["logs"], data["labels"], data["knowledge"])
    elif args.strategy == "replace_in_snippet":
        logs, labels, knowledge = change_entity_in_snippet(data["logs"], data["labels"], data["knowledge"])
    elif args.strategy == "replace_in_snippet_sf":
        logs, labels, knowledge = change_entity_in_snippet(data["logs"], data["labels"], data["knowledge"], city="San Francisco")
    elif args.strategy.startswith('add_validation_data'):
        additional_data = None  # add all additional data by default
        if args.strategy.startswith('add_validation_data_'):
            additional_data_match = re.match(r'add_validation_data_(?:([\w_]+)-)*([\w_]+)', args.strategy)
            additional_data = [d for d in additional_data_match.groups() if d is not None]
        logs, labels, knowledge = augment_by_adding_validation_data(data["logs"], data["labels"], data["knowledge"], additional_data=additional_data)
    else:
        assert False, f'Unsupported strategy: {args.strategy}'


    for path, obj in zip([args.out_logs, args.out_labels, args.out_knowledge], [logs, labels, knowledge]):
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)

if __name__ =="__main__":
    main(sys.argv)        
