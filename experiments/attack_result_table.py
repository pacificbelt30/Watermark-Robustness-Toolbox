import json
import os
import pandas as pd
import matplotlib.pyplot as plt

def judge(json_path):
    result = {}
    table = {}
    steel_losses = {}
    decision_thresholds = {}

    try:
        with open(json_path,'r') as f:
            result = json.loads(f.read())
    except:
        import traceback
        traceback.print_exc()

    print(result.keys())
    print(result['vit'].keys())
    for key in result['vit'].keys():
        if key == 'acc':
            continue
        table[key] = {}
        steel_losses[key] = {}
        decision_thresholds[key] = {}
        for atk in result['vit'][key]['atk'].keys():
            table[key][atk] = ''
            steel_losses[key][atk] = []
            decision_thresholds[key][atk] = []
    print(table.keys())

    for arch in result.keys():
        for key in result[arch].keys():
            if key == 'acc':
                continue
            for atk in result[arch][key]['atk'].keys():
                print(atk)
                before = result[arch][key]['wm']['test_acc']
                after = result[arch][key]['atk'][atk]['test_acc_after']
                steel_losses[key][atk].append(after-before)
                dt = result[arch][key]['decision_thresholds']['y'][-1] / result[arch][key]['decision_thresholds']['x'][-1]
                wmacc = result[arch][key]['atk'][atk]['wm_acc_after']
                decision_thresholds[key][atk].append(wmacc - dt)
                print(after-before,wmacc)

    tmark = '' # Success attack and watermarking method is vuln.
    fmark = '' # Failed attack and watermarking method is safe.
    loss_thresholds = 0.05
    arch = 'vit'
    for key in result[arch].keys():
        if key == 'acc':
            continue
        for atk in result[arch][key]['atk'].keys():
            for loss,dt in zip(steel_losses[key][atk],decision_thresholds[key][atk]):
                # print(arch,key,atk,loss,dt,(loss>loss_thresholds and dt > 0.0))
                # table[key][atk].append(tmark if ((loss <= loss_thresholds) and (dt >= 0.0)) else fmark)
                if atk == 'cross_architecture_retraining':
                    table[key][atk] += (tmark if ((dt < 0.0)) else fmark) + ' / '
                else:
                    table[key][atk] += (tmark if ((loss+loss_thresholds > 0.0) and (dt < 0.0)) else fmark) + ' / '
                    # table[key][atk] += (tmark if ((dt < 0.0)) else fmark) + ' / '
            table[key][atk] = table[key][atk][:-2]

    print(pd.DataFrame(data=table))
    print(table.keys())
    # print(json.dumps(table,indent=2))
    decision_thresholds = {}
    for key in result['vit'].keys():
        if key == 'acc':
            continue
        decision_thresholds[key] = {}
        for arch in result.keys():
            decision_thresholds[key][arch] = round(result[arch][key]['decision_thresholds']['y'][-1] / result[arch][key]['decision_thresholds']['x'][-1], 5)
    judge_plot(table, result.keys())
    decision_thresholds_plot(decision_thresholds, result.keys())

def judge_plot(table:dict, archs):
    # plot
    df = pd.DataFrame(data=table)
    legend = ''    
    for key in archs:
        legend += key + ' / '
    legend = legend[:-2]
    w, h = 10,5
    plt.rcParams["font.family"] = 'HackGen35Nerd'
    # plt.rcParams["font.size"] = 20
    fig, ax = plt.subplots(figsize=(w,h))
    ax.axis('off')
    fig.subplots_adjust(left=0.26,right=0.97)
    ax.text(0.6,-0.1,legend)
    ax.table(
        df.values,
        colLabels = df.columns,
        rowLabels = df.index,
        rowLoc = 'right',
        loc = 'center',
        bbox=[0,0,1,1]
    )
    plt.savefig('test.png')
# plt.show()

def decision_thresholds_plot(table:dict, archs):
    # plot
    df = pd.DataFrame(data=table)
    print(df)
    legend = ''    
    for key in archs:
        legend += key + ' / '
    legend = legend[:-2]
    w, h = 10,5
    plt.rcParams["font.family"] = 'HackGen35Nerd'
    # plt.rcParams["font.size"] = 20
    fig, ax = plt.subplots(figsize=(w,h))
    ax.axis('off')
    fig.subplots_adjust(left=0.26,right=0.97)
    ax.text(0.6,-0.1,legend)
    ax.table(
        df.values,
        colLabels = df.columns,
        rowLabels = df.index,
        rowLoc = 'right',
        loc = 'center',
        bbox=[0,0,1,1]
    )
    plt.savefig('test_dt.png')
    # plt.show()

def main():
    file = './outputs/cifar10/atk_json/result4.json'
    judge(file)

if __name__ == "__main__":
    main()
