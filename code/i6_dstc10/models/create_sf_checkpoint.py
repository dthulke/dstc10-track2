import argparse

from transformers import AutoConfig
from spoken_lm import PretrainedShallowFusionModel, ShallowFusionConfig


parser = argparse.ArgumentParser()
parser.add_argument("--dm_model_name_or_path", type=str)
parser.add_argument("--lm_model_name_or_path", type=str)
parser.add_argument("--scaling_factor", type=str)
parser.add_argument("--checkpoint_path", type=str)

args, additional_args = parser.parse_known_args()

lm_config = AutoConfig.from_pretrained(args.lm_model_name_or_path)
config = ShallowFusionConfig(
    direct_model_tokenizer_name_or_path=args.dm_model_name_or_path,
    direct_model_name_or_path=args.dm_model_name_or_path,
    language_model_tokenizer_name_or_path=args.lm_model_name_or_path,
    language_model_name_or_path=args.lm_model_name_or_path,
    scaling_factor=float(args.scaling_factor),
    **lm_config.to_diff_dict()
)

model = PretrainedShallowFusionModel(config=config)
model.save_pretrained(args.checkpoint_path)