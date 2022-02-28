import sys

from i6_core.tools import CloneGitRepositoryJob
from i6_dstc.huggingface.evaluation import CalculateDSTC10MetricsJob

sys.setrecursionlimit(2500)

# ------------------------------ Sisyphus -------------------------------------

import sisyphus.toolkit as tk
from i6_dstc.huggingface.training import *
from i6_dstc.huggingface.search import *

Path = tk.Path

# ------------------------------ Recipes --------------------------------------


async def detection():
  code_root = ""#TODO

  config = {
    'model_name_or_path': 'roberta-large',
    'method': 'baseline_detection',
    'learning_rate': 6.25e-6,
    'per_device_train_batch_size': 2,
    'gradient_accumulation_steps': 8,
  }
  train_data_config = {
    'dataset_name': os.path.join(code_root, 'i6_dstc10/datasets/dstc9_track1.py'),
    'dataset_config_name': 'detection',
    'dataset_train_split': 'train',
    'dataset_eval_split': 'validation',
  }
  search_data_config = {
    'dataset_name': os.path.join(code_root, 'i6_dstc10/datasets/dstc9_track1.py'),
    'dataset_config_name': 'detection',
    'dataset_test_split': 'test',
  }

  # Train the model
  train_job = HuggingfaceTrainingJob(
    code_root=code_root,
    config=config,
    train_data_config=train_data_config,
    num_epochs=10,
    mem_rqmt=12,
    time_rqmt=24,
  )
  train_job.add_alias("train_job-detection_baseline")

  search_job = HuggingfaceSearchJob(
    code_root=code_root,
    model_path=train_job.out_best_model,
    config=config,
    search_data_config=search_data_config,
    mem_rqmt=8,
  )

  tk.register_output('detection_baseline_out.json', search_job.out_search_file)

  scoring_job = CalculateDSTC10MetricsJob(
    code_root,
    "test",
    search_job.out_search_file,  # search_job.out_search_file
  )
  tk.register_output('detection_baseline_scores.json', scoring_job.out_results_file)


async def hierarchical_selection(name, config_update, train_commit, search_commit, train_config_update={}, search_config_update={}, train_use_gitlab=False):
  def get_train_code_root():
    url = "" # TODO: add path
    return CloneGitRepositoryJob(url, commit=train_commit).out_repository

  def get_search_code_root():
    return CloneGitRepositoryJob('', commit=search_commit).out_repository #TODO: add path

  config = {
    'model_name_or_path': 'roberta-large',
    'method': 'cross_encoder',
    'learning_rate': 6.25e-6,
    'per_device_train_batch_size': 2,
    'gradient_accumulation_steps': 8,
    'per_device_eval_batch_size': 8,
  }
  config.update(config_update if config_update is not None else {})
  train_data_config = {
    'dataset_name': os.path.join(get_train_code_root(), 'i6_dstc10/datasets/dstc9_track1.py'),
    'dataset_config_name': 'selection',
    'dataset_train_split': 'train',
    'dataset_eval_split': 'validation',
    **train_config_update,
  }
  base_search_data_config = {
    'dataset_config_name': 'selection',
    **search_config_update,
  }

  domain_entity_config = {
    'selection_level': 'domain_entity',
    'num_domain_negatives': 1,
    'num_entity_negatives': 2,
    'num_doc_negatives': 0,
  }
  doc_config = {
    'selection_level': 'document',
    'num_domain_negatives': 0,
    'num_entity_negatives': 0,
    'num_doc_negatives': 3,
  }

  ############
  # TRAINING #
  ############

  best_models = {}
  for selection_config in [domain_entity_config, doc_config]:
    final_config = {
      **config,
      **selection_config,
    }

    # Train the model
    train_job = HuggingfaceTrainingJob(
      code_root=get_train_code_root(),
      config=final_config,
      train_data_config=train_data_config,
      num_epochs=5,
      mem_rqmt=12,
      time_rqmt=24,
      keep_only_best=True,
    )
    best_models[selection_config['selection_level']] = train_job.out_best_model

  ##############
  # EVALUATION #
  ##############

  eval_datasets = {
    'validation': {
      'dataset_name': os.path.join(get_search_code_root(), 'i6_dstc10/datasets/dstc9_track1.py'),
      'dataset_test_split': 'validation',
    },
    'test_mwoz': {
      'dataset_name': os.path.join(get_search_code_root(), 'i6_dstc10/datasets/dstc9_track1.py'),
      'dataset_test_split': 'test',
      'dataset_filter_dict': {"source": "multiwoz"},
    },
    'sf_written': {
      'dataset_name': os.path.join(get_search_code_root(), 'i6_dstc10/datasets/dstc9_track1.py'),
      'dataset_test_split': 'test',
      'dataset_filter_dict': {"source": "sf_written"},
    },
    'sf_spoken': {
      'dataset_name': os.path.join(get_search_code_root(), 'i6_dstc10/datasets/dstc9_track1.py'),
      'dataset_test_split': 'test',
      'dataset_filter_dict': {"source": "sf_spoken"},
    },
    'sf_spoken_asr': {
      'dataset_name': os.path.join(get_search_code_root(), 'i6_dstc10/datasets/dstc10_track2.py'),
      'dataset_test_split': 'validation',
    },
  }

  pred_files = defaultdict(dict)
  for data_name, eval_data_config in eval_datasets.items():
    for selection_config in [domain_entity_config, doc_config]:
      search_data_config = {
        **base_search_data_config,
        **eval_data_config,
      }
      final_config = {
        **config,
        **selection_config,
      }

      if selection_config['selection_level'] == 'domain_entity':
        search_data_config['dataset_data_files'] = None
      elif selection_config['selection_level'] == 'document':
        search_data_config['dataset_config_name'] = 'generation'

        data_file_split_name = search_data_config['dataset_test_split'] \
          if search_data_config['dataset_test_split'] != 'validation' else 'val'
        search_data_config['dataset_data_files'] = {
          data_file_split_name: pred_files['domain_entity'][data_name]
        }
      else:
        assert False, 'Not implemented'

      search_job = HuggingfaceSearchJob(
        code_root=get_search_code_root(),
        model_path=best_models[selection_config['selection_level']],
        config=final_config,
        search_data_config=search_data_config,
        mem_rqmt=8,
        time_rqmt=12,
      )

      concat_search_file = MergePredictionsJob(
        [search_job.out_search_file],
        search_data_config['dataset_name'],
        search_data_config['dataset_test_split'],
        data_files=search_data_config.get('dataset_data_files', None),
        dataset_filter_dict=search_data_config.get('dataset_filter_dict', None),
      ).out_predictions_file

      pred_files[selection_config['selection_level']][data_name] = concat_search_file

    scoring_job = CalculateDSTC10MetricsJob(
      get_search_code_root(),
      eval_data_config["dataset_name"],
      eval_data_config['dataset_test_split'],
      pred_files['document'][data_name],
      dataset_filter_dict=eval_data_config.get('dataset_filter_dict', None),
    )
    tk.register_output(f'selection/hierarchical/selection_hierarchical_cross_encoder_{name}__{data_name}_scores.json', scoring_job.out_results_file)

  merged_test_splits = MergeDSTC9TestSplitsJob(
    mwoz_predictions_file=pred_files['document']['test_mwoz'],
    sf_spoken_predictions_file=pred_files['document']['sf_spoken'],
    sf_written_predictions_file=pred_files['document']['sf_written'],
  ).out_predictions_file
  scoring_job = CalculateDSTC10MetricsJob(
    get_search_code_root(),
    dataset_name=os.path.join(get_search_code_root(), 'i6_dstc10/datasets/dstc9_track1.py'),
    split='test',
    model_output_file=merged_test_splits,
  )
  tk.register_output(f'selection/hierarchical/selection_hierarchical_cross_encoder_{name}__test_scores.json',
                     scoring_job.out_results_file)

  async def generation():
    pass


async def async_main():
  await detection()

  await asyncio.gather(
    hierarchical_selection("large", config_update={}, train_commit='66b90d5', search_commit='335de9b'),
  )


async def py():
  await async_main()
