import matplotlib.pyplot as plt
import json
import os
import numpy as np

def load_result_from_jsons(json_paths,arch=['WideResNet','ViT','R50+ViT','R50+ViT_WPM']):
    arch = ['resnet','vit','r50vit','r50vit_wpm']
    arch_name = ['WideResNet','ViT-B/16','R50+ViT-B/16','R50+ViT-B/16 not pretrained']
    labels = ['adi','content','unrelated','noise','blackmarks','frontier','jia']
    if len(json_paths) != len(arch):
        print('Arg length not suitable')
        return

    exist_path = []
    for f in json_paths:
        if os.path.isfile(f):
            exist_path.append(f)
    print(exist_path)
    result = []
    for file in exist_path:
        try:
            with open(file,'r') as f:
                result.append(json.loads(f.read()))
        except:
            import traceback
            traceback.print_exc()
    base = 'outputs/cifar10/png/'
    print(result)
    result = {}
    result2 = {}
    result3 = {}
    result4 = {}
    file = './outputs/cifar10/wm_experiments/vit/jia/history.json'
    # file2 = './outputs/cifar10/wm_experiments/vit/jia/history2.json'
    # file2 = './outputs/cifar10/wm_experiments/vit/resnet/history.json'
    file2 = './outputs/cifar10/wm_experiments/vit/r50/history.json'
    file3 = './outputs/cifar10/wm_experiments/vit/wpm/history.json'
    file4 = './outputs/cifar10/wm_experiments/vit/jia/history2.json'
    try:
        with open(file,'r') as f:
            result = json.loads(f.read())
        with open(file2,'r') as f:
            result2 = json.loads(f.read())
        with open(file3,'r') as f:
            result3 = json.loads(f.read())
        with open(file4,'r') as f:
            result4 = json.loads(f.read())
    except:
        import traceback
        traceback.print_exc()
    plot_jia_loss('test.png',result,result2,result3,result4)

# def plot_jia_loss(filename, archs:list, wms:list, result_dict:dict, archs_name):
def plot_jia_loss(filename,result_dict:dict,result_dict2:dict,result_dict3:dict,result_dict4:dict):
    print(result_dict['train_loss'])
    print(result_dict['wm_loss'])
    margin = 0.01
    total_width = 1 - margin
    print('plot')
    # plt.figure(figsize=(8,5))
    plt.figure(figsize=(8,7))
    plt.title('jia_loss in ViT',fontsize=16)
    # labels = plt.get_xticklabels()
    # plt.setp(labels, rotation=30, fontsize=16);
    # plt.set_ylim(lim[0],lim[1])
    # plt.plot([i for i in range(len(result_dict['train_loss']))],result_dict['train_loss'])
    plt.plot([i for i in range(len(result_dict['wm_loss']))],result_dict['wm_loss'])
    plt.plot([i for i in range(len(result_dict2['wm_loss']))],result_dict2['wm_loss'])
    plt.plot([i for i in range(len(result_dict3['wm_loss']))],result_dict3['wm_loss'])
    plt.plot([i for i in range(len(result_dict4['wm_loss']))],result_dict4['wm_loss'])
    # plt.xticks([i+1 for i in range(len(wms))],wms,fontsize=12.5)
    plt.ylabel('loss',fontsize=16)
    plt.xlabel('epoch',fontsize=16)
    # plt.legend(archs_name,bbox_to_anchor=(1.0, 0),loc='lower right',borderaxespad=1,fontsize=16)
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    path = ['./outputs/cifar10/json/result_00016.json','./outputs/cifar10/json/result_00017.json','./outputs/cifar10/json/result_00018.json','./outputs/cifar10/json/result_00019.json']
    load_result_from_jsons(path)
