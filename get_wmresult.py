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
    parser.add_argument('-o', '--output_dir', type=str, default='./outputs/cifar10/json/', help="Path to output file for all watermarking scheme result")
    parser.add_argument('-r', '--result_dir', type=str, default='./configs/cifar10/wm_configs_vit/',
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
def main2():
    args = parse_args()

    all_result_dir = [os.path.abspath(os.path.join(args.result_dir, x)) for x in os.listdir(args.result_dir)]
    all_result_dir = sorted(all_result_dir,reverse=True)
    print(all_result_dir)
    n = len(all_result_dir)
    check_dir = []
    for i, dir in enumerate(all_result_dir):
        wmdir = [os.path.abspath(os.path.join(dir, x)) for x in os.listdir(dir)]
        # print((os.path.basename(dir),sorted(wmdir,reverse=True)[0]))
        check_dir.append((os.path.basename(dir),sorted(wmdir,reverse=True)[0]))

    result_json = {'result':[],'name':os.path.basename(args.result_dir)}
    for i, dir in enumerate(all_result_dir):
        if not os.path.isfile(os.path.join(dir,'result.json')):
            print(f'result.json does not exist in {dir}.')
            continue
        result_json['result'].append({})
        result_json['result'][-1]['name'] = os.path.basename(dir).split('_')[1]
        try:
            with open(os.path.join(dir,'result.json'),'r') as f:
                import json
                result_json['result'][-1]['result'] = json.loads(f.read())
        except:
            import traceback
            traceback.print_exc()
            continue
    # print(result_json)
    num_files = len(os.listdir(args.output_dir))
    pre_zero_str = '0'*(5-len(str(num_files)))
    try:
        with open(os.path.join(args.output_dir,f'result_{pre_zero_str}{num_files}.json'),'w') as f:
            import json
            print('save json in:', os.path.join(args.output_dir,f'result_{pre_zero_str}{num_files}.json'))
            json.dump(result_json,f,indent=2)
            # print(result_json,file=f)
    except:
        import traceback
        traceback.print_exc()

def main():
    args = parse_args()

    all_result_dir = [os.path.abspath(os.path.join(args.result_dir, x)) for x in os.listdir(args.result_dir)]
    all_result_dir = sorted(all_result_dir,reverse=True)
    print(all_result_dir)
    n = len(all_result_dir)
    check_dir = []
    for i, dir in enumerate(all_result_dir):
        wmdir = [os.path.abspath(os.path.join(dir, x)) for x in os.listdir(dir)]
        print((os.path.basename(dir),sorted(wmdir,reverse=True)[0]))
        check_dir.append((os.path.basename(dir),sorted(wmdir,reverse=True)[0]))

    result_json = {'result':[]}
    for i, dir in enumerate(check_dir):
        if not os.path.isfile(os.path.join(dir[1],'result.json')):
            print(f'result.json does not exist in {dir[1]}.')
            continue
        result_json['result'].append({})
        result_json['result'][-1]['name'] = dir[0]
        try:
            with open(os.path.join(dir[1],'result.json'),'r') as f:
                import json
                result_json['result'][-1]['result'] = json.loads(f.read())
        except:
            import traceback
            traceback.print_exc()
            continue
    print(result_json)
    num_files = len(os.listdir(args.output_dir))
    pre_zero_str = '0'*(5-len(str(num_files)))
    try:
        with open(os.path.join(args.output_dir,f'result_{pre_zero_str}{num_files}.json'),'w') as f:
            import json
            json.dump(result_json,f,indent=2)
            # print(result_json,file=f)
    except:
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main2()
