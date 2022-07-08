import argparse
from train import train
from generate import generate
import config

parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true', required=False)
parser.add_argument('--generate', action='store_true', required=False)
test_flag = parser.parse_args()
parser.add_argument('--carla_dir', type=str, required=False)
parser.add_argument('--darwin_dir', type=str, required=False)
parser.add_argument('--generate_dataset_dir', type=str, required=False)
parser.add_argument('--generate_dir', type=str, required=False)
parser.add_argument('--load_model', type=str, required=False if test_flag.train else True)
parser.add_argument('--save_model', type=str, required=False)

if __name__ == "__main__":
    args = parser.parse_args()
    config.LOAD_MODEL = True if len(args.load_model) > 0 else False
    config.SAVE_MODEL = True if len(args.save_model) > 0 else False

    config.CARLA_DIR = args.carla_dir
    config.DARWIN_DIR = args.darwin_dir

    config.CHECKPOINT_GEN_C = args.save_model + config.CHECKPOINT_GEN_C
    config.CHECKPOINT_GEN_D = args.save_model + config.CHECKPOINT_GEN_D
    config.CHECKPOINT_CRITIC_C = args.save_model + config.CHECKPOINT_CRITIC_C
    config.CHECKPOINT_CRITIC_D = args.save_model + config.CHECKPOINT_CRITIC_D

    config.LOAD_CHECKPOINT_GEN_C = args.load_model + config.LOAD_CHECKPOINT_GEN_C
    config.LOAD_CHECKPOINT_GEN_D = args.load_model + config.LOAD_CHECKPOINT_GEN_D
    config.LOAD_CHECKPOINT_CRITIC_C = args.load_model + config.LOAD_CHECKPOINT_CRITIC_C
    config.LOAD_CHECKPOINT_CRITIC_D = args.load_model + config.LOAD_CHECKPOINT_CRITIC_D

    config.GEN_DIR = args.generate_dir
    config.GEN_DATASET_DIR = args.generate_dataset_dir

    if args.train:
        train()

    if args.generate:
        generate()
