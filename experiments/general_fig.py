import matplotlib.pyplot as plt
import json
import os
import numpy as np

def load_result_from_jsons(json_paths,arch=['WideResNet','ViT','R50+ViT','R50+ViT_WPM']):
    arch = ['resnet','vit','r50vit','r50vit_wpm']
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
    file = './outputs/cifar10/atk_json/result_00004.json'
    try:
        with open(file,'r') as f:
            result = json.loads(f.read())
    except:
        import traceback
        traceback.print_exc()
    plot_embed_loss(os.path.join(base,'embed_loss.png'),arch,labels,result)
    # plot_steal_loss(os.path.join(base,'steal_loss.png'),arch[0],labels,result[arch[0]][labels[0]]['atk'].keys(),result)
    plot_steal_losses(os.path.join(base,'steal_loss.png'),base ,arch,labels,result[arch[0]][labels[0]]['atk'].keys(),result)
    plot_wm_acc(os.path.join(base,'wm_acc.png'),arch,labels,result)
    plot_steal_wm_accs(os.path.join(base,'steal_loss.png'),base ,arch,labels,result[arch[0]][labels[0]]['atk'].keys(),result)

def plot_embed_loss(filename, archs:list, wms:list, result_dict:dict):
    margin = 0.01
    total_width = 1 - margin
    print('plot')
    # plt.figure(figsize=(8,5))
    plt.figure(figsize=(8,7))
    plt.title('embed loss')
    # labels = plt.get_xticklabels()
    # plt.setp(labels, rotation=30, fontsize=10);
    # plt.set_ylim(lim[0],lim[1])
    for i,t in enumerate(archs):
        new_y = []
        for j,wm in enumerate(wms):
            # print('tetete',t)
            print(wm)
            new_y.append(result_dict[t]['acc']['test_acc']/100.0-result_dict[t][wm]['wm']['test_acc'])
        pos = np.array([i+1 for i in range(len(wms))]) - total_width * ( 1 - (2*i+4)/len(wms) )/2
        print(new_y)
        plt.bar(pos,new_y,width=total_width/len(wms))
    plt.xticks([i+1 for i in range(len(wms))],wms)
    plt.ylabel('embed_loss')
    plt.legend(archs,bbox_to_anchor=(1.0, 0),loc='lower right',borderaxespad=1,fontsize=10)
    plt.savefig(filename)
    plt.close()

def plot_steal_losses(filename, base, archs:str, wms:list, attacks:list, result_dict:dict):
    for arch in archs:
        plot_steal_loss(os.path.join(base,f'steal_loss_{arch}.png'),arch,wms,result_dict[arch][wms[0]]['atk'].keys(),result_dict)

def plot_steal_loss(filename, arch:str, wms:list, attacks:list, result_dict:dict):
    margin = 0.01
    total_width = 1 - margin
    print('plot')
    # plt.figure(figsize=(8,5))
    plt.figure(figsize=(8,9))
    plt.title(f'{arch} steal loss')
    # labels = plt.get_xticklabels()
    # plt.setp(labels, rotation=30, fontsize=10);
    # plt.set_ylim(lim[0],lim[1])
    for i,wm in enumerate(wms):
        new_y = []
        for j,attack in enumerate(attacks):
            # print('tetete',t)
            print(result_dict[arch][wm]['atk'].keys())
            print(len(result_dict[arch][wm]['atk'].keys()))
            new_y.append(result_dict[arch][wm]['wm']['test_acc']-result_dict[arch][wm]['atk'][attack]['test_acc_after'])
        pos = np.array([i+1 for i in range(len(attacks))]) - total_width * ( 1 - (2*i+4)/len(attacks) )/2
        print(new_y)
        plt.bar(pos,new_y,width=total_width/len(attacks))
    plt.xticks([i+1 for i in range(len(attacks))],attacks,rotation=20)
    plt.ylabel('steal_loss')
    plt.legend(wms,bbox_to_anchor=(1.0, 0),loc='lower right',borderaxespad=1,fontsize=10)
    plt.savefig(filename)
    plt.close()

def plot_steal_wm_accs(filename, base, archs:str, wms:list, attacks:list, result_dict:dict):
    for arch in archs:
        plot_steal_wm_acc(os.path.join(base,f'steal_wm_acc_{arch}.png'),arch,wms,result_dict[arch][wms[0]]['atk'].keys(),result_dict)

def plot_steal_wm_acc(filename, arch:str, wms:list, attacks:list, result_dict:dict):
    margin = 0.01
    total_width = 1 - margin
    print('plot')
    # plt.figure(figsize=(8,5))
    plt.figure(figsize=(8,9))
    plt.title(f'{arch} Watermark Accuracy after removal attacks')
    # labels = plt.get_xticklabels()
    # plt.setp(labels, rotation=30, fontsize=10);
    # plt.set_ylim(lim[0],lim[1])
    for i,wm in enumerate(wms):
        new_y = []
        for j,attack in enumerate(attacks):
            # print('tetete',t)
            print(result_dict[arch][wm]['atk'].keys())
            print(len(result_dict[arch][wm]['atk'].keys()))
            new_y.append(result_dict[arch][wm]['atk'][attack]['wm_acc_after'])
        pos = np.array([i+1 for i in range(len(attacks))]) - total_width * ( 1 - (2*i+4)/len(attacks) )/2
        print(new_y)
        plt.bar(pos,new_y,width=total_width/len(attacks))
    plt.xticks([i+1 for i in range(len(attacks))],attacks,rotation=20)
    plt.ylabel('Watermark Accuracy')
    plt.legend(wms,bbox_to_anchor=(1.0, 0),loc='lower right',borderaxespad=1,fontsize=10)
    plt.savefig(filename)
    plt.close()

def plot_wm_acc(filename, archs:list, wms:list, result_dict:dict):
# def plot_wm_acc(filename, arch:str, wms:list, attacks:list, result_dict:dict):
    margin = 0.01
    total_width = 1 - margin
    print('plot')
    plt.figure(figsize=(8,5))
    plt.title('Watermark Accuracy')
    # labels = plt.get_xticklabels()
    # plt.setp(labels, rotation=30, fontsize=10);
    # plt.set_ylim(lim[0],lim[1])
    for i,t in enumerate(archs):
        new_y = []
        for j,wm in enumerate(wms):
            # print('tetete',t)
            print(wm)
            new_y.append(result_dict[t][wm]['wm']['wm_acc'])
        pos = np.array([i+1 for i in range(len(wms))]) - total_width * ( 1 - (2*i+4)/len(wms) )/2
        print(new_y)
        plt.bar(pos,new_y,width=total_width/len(wms))
    plt.xticks([i+1 for i in range(len(wms))],wms)
    plt.ylabel('Watermark Accuracy')
    plt.legend(archs,bbox_to_anchor=(1.0, 0),loc='lower right',borderaxespad=1,fontsize=10)
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    path = ['./outputs/cifar10/json/result_00016.json','./outputs/cifar10/json/result_00017.json','./outputs/cifar10/json/result_00018.json','./outputs/cifar10/json/result_00019.json']
    load_result_from_jsons(path)
