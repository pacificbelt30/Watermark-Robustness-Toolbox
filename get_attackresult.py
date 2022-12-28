""" This script runs the "embed.py" script on a list of watermarking configurations
"""

import argparse
import os

# Register all hooks for the models.
# noinspection PyUnresolvedReferences
import time
from math import ceil
from multiprocessing import Process


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', type=str, default='./outputs/cifar10/atk_json/', help="Path to output file for all watermarking scheme result")
    parser.add_argument('-r', '--result_dir', type=str, default='./outputs/cifar10/attack_experiments/',
                        help="Path to config file for the watermarking scheme.")
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument("--pretrained_dir", default="./outputs/cifar10/null_models/vit/00000_null_model")
    parser.add_argument("--filename", type=str, default="best.pth", help="Filepath to the pretrained model.")
    parser.add_argument("-n", "--num_processes", type=int, default=1, help="Number of concurrent processes.")
    parser.add_argument("--gpu", type=str, default=None, help="Which GPU to use. Defaults to GPU with least memory.")
    return parser.parse_args()

"""
ModelArch1
    - scheme1_dir
        - result.json
    - scheme2_dir
        - result.json
ModelArch2
    - scheme1_dir
        - result.json
    - scheme2_dir
        - result.json
"""
def main():
    args = parse_args()
    attack_result = {}
    # attack_config
    all_result_dir = [os.path.abspath(os.path.join(args.result_dir, x)) for x in os.listdir(args.result_dir)]
    for i, dir in enumerate(all_result_dir):
        if not os.path.isdir(dir):
            print(dir,'is not a directory.')
            continue
        # attack_config/0000x_attack_config
        attack_dirs = [os.path.abspath(os.path.join(dir, x)) for x in os.listdir(dir)]
        attack_result_dict = {}
        print(attack_dirs)
        for j, attack_dir in enumerate(attack_dirs):
            if not os.path.isdir(attack_dir):
                print(attack_dir,'is not a directory.')
                continue
            files = [os.path.abspath(os.path.join(attack_dir, x)) for x in os.listdir(attack_dir)]
            if os.path.abspath(os.path.join(attack_dir, 'args.json')) not in files:
                print('args.json not found in', )
                continue
            elif os.path.abspath(os.path.join(attack_dir, 'result.json')) not in files:
                print('result.json not found in', )
                continue
            args_dict = {}
            try:
                with open(os.path.join(attack_dir,'args.json'),'r') as f:
                    import json
                    args_dict = json.loads(f.read())
            except:
                import traceback
                traceback.print_exc()
                continue
            wm_name = os.path.basename(args_dict['wm_dir']).split('_')[1]
            arch = os.path.basename(os.path.dirname(args_dict['wm_dir']))
            wm_result_path = os.path.join(args_dict['wm_dir'],'result.json')
            attack_name = os.path.basename(args_dict['attack_config']).split('.')[0]
            if arch not in attack_result:
                attack_result[arch] = {}
            if wm_name not in attack_result[arch]:
                attack_result[arch][wm_name] = {}
            if 'atk' not in attack_result[arch][wm_name]:
                attack_result[arch][wm_name]['atk'] = {}
            # attack_result[arch][wm_name] = wm_result_path
            wm_result = {}
            attack_result_list = []
            try:
                with open(wm_result_path,'r') as f:
                    import json
                    wm_result = json.loads(f.read())
            except:
                import traceback
                traceback.print_exc()
                continue
            try:
                with open(wm_result_path,'r') as f:
                    import json
                    attack_result_dict[attack_name] = json.loads(f.read())
            except:
                import traceback
                traceback.print_exc()
                continue
            # attack_result[arch][wm_name] = {'wm':wm_result,'atk':attack_result_dict}
            attack_result[arch][wm_name]['wm'] = wm_result
            attack_result[arch][wm_name]['atk'][attack_name] = attack_result_dict[attack_name]
    print(attack_result)
    num_files = len(os.listdir(args.output_dir))
    pre_zero_str = '0'*(5-len(str(num_files)))
    try:
        with open(os.path.join(args.output_dir,f'result_{pre_zero_str}{num_files}.json'),'w') as f:
            import json
            print('save json in:', os.path.join(args.output_dir,f'result_{pre_zero_str}{num_files}.json'))
            json.dump(attack_result,f,indent=2)
            # print(result_json,file=f)
    except:
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
