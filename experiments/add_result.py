""" This script adds the "result.json" to an output directory if it cannot find one.
"""

import argparse
import os

# Register all hooks for the models.
# noinspection PyUnresolvedReferences
import time
from math import ceil
from multiprocessing import Process
from tqdm import tqdm

"""
Expects the following dir:
<attack_name>
    - 00000_attack
        - best.pth
    - 00001_attack
        ...
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="../outputs/cifar10/attacks/knockoff")
    parser.add_argument("--filename", type=str, default="best.pth", help="Filepath to the pretrained model.")
    parser.add_argument("--gpu", type=str, default=None, help="Which GPU to use. Defaults to GPU with least memory.")
    return parser.parse_args()


def file_with_suffix_exists(dirname, suffix, not_contains="", raise_error=False) -> str:
    for file in os.listdir(dirname):
        if file.endswith(suffix) and (not not_contains in file or len(not_contains) == 0):
            return os.path.abspath(os.path.join(dirname, file))
    if raise_error:
        raise FileNotFoundError(f"No file found with suffix '{suffix}' in '{dirname}")
    # return False
    return ''


def run_tasks(folder, wm_folder, wm_config, pth_file, output_filename, atk_config):
    os.system(f"cd .. && python validate_wm.py -f {os.path.join(wm_folder, pth_file)} -w {wm_config} "
              f"-o {output_filename} -a {atk_config} -d {os.path.join(folder,'checkpoint.pth')}")


def main():
    args = parse_args()

    _, attack_name = os.path.split(args.output_dir)
    # Collect all folder paths.
    folders, wm_configs, wm_bases, atk_configs = [], [], [], []
    for folder in os.listdir(args.output_dir):
        full_path = os.path.abspath(os.path.join(args.output_dir, folder))

        wm_config = file_with_suffix_exists(full_path, suffix=".yaml", not_contains=attack_name)
        atk_config = file_with_suffix_exists(full_path, suffix=".yaml", not_contains=wm_config)
        wm_base = os.path.basename(wm_config).split('.')[0]
        if os.path.exists(os.path.join(full_path, args.filename)):
            folders.append(full_path)
            wm_configs.append(wm_config)
            wm_bases.append(wm_base)
            atk_configs.append(atk_config)

    # Create the result.json
    for folder, wm_config, wm_base, atk_config in zip(tqdm(folders), wm_configs, wm_bases, atk_configs):
        wm_folder = f"./outputs/cifar10/wm/{wm_base}/00000_{wm_base}"
        run_tasks(folder, wm_folder, wm_config, pth_file=args.filename, output_filename="result.json",atk_config=atk_config)


if __name__ == "__main__":
    main()
