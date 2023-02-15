import matplotlib.pyplot as plt
import json
import os
import numpy as np

def load_result_from_jsons(json_paths,arch=['WideResNet','ViT','R50+ViT','R50+ViT_WPM']):
    arch = ['resnet','vit','r50vit','r50vit_wpm']
    arch_name = ['WideResNet','ViT-B/16','R50+ViT-B/16','R50+ViT-B/16 not pretrained']
    labels = ['adi','content','unrelated','noise','blackmarks','frontier','jia']
    wmschemes = ['content', 'noise', 'unrelated', 'adi', 'jia', 'frontier', 'blackmarks']
    labels = ['content', 'noise', 'unrelated', 'adi', 'jia', 'frontier', 'blackmarks']
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
    file = './outputs/cifar10/atk_json/result6.json'
    try:
        with open(file,'r') as f:
            result = json.loads(f.read())
    except:
        import traceback
        traceback.print_exc()
    attacks = result[arch[0]][labels[0]]['atk'].keys()
    # attacks = ['input_gaussian_smoothing',"regularization","neural_cleanse_unlearning","feature_shuffling","weight_shifting","retraining","cross_architecture_retraining","transfer_learning"]
    attacks = ['input_gaussian_smoothing',"regularization","neural_cleanse_unlearning","weight_shifting","retraining","cross_architecture_retraining","transfer_learning"]
    plot_decision_threshold_dist(os.path.join(base,'dt_hist'),arch,labels,result,arch_name)
    plot_embed_loss(os.path.join(base,'embed_loss.png'),arch,labels,result,arch_name)
    plot_embed_acc(os.path.join(base,'embed_acc.png'),arch,labels,result,arch_name)
    plot_steal_losses(os.path.join(base,'steal_loss.png'),base ,arch,labels,attacks,result,arch_name)
    plot_wm_acc(os.path.join(base,'wm_acc.png'),arch,labels,result,arch_name)
    plot_steal_wm_accs(os.path.join(base,'steal_loss.png'),base ,arch,labels,attacks,result,arch_name)
    plot_wm_acc_extend_jia('./outputs/cifar10/atk_json/extend.json','test.png')

def plot_embed_acc(filename, archs:list, wms:list, result_dict:dict, archs_name):
    margin = 0.01
    total_width = 1 - margin
    print('plot')
    plt.figure(figsize=(12,7))
    # plt.figure(figsize=(16,9))
    plt.title('Watermarked Model Accuracy',fontsize=20)
    # labels = plt.get_xticklabels()
    # plt.setp(labels, rotation=30, fontsize=16);
    # plt.set_ylim(lim[0],lim[1])
    for i,t in enumerate(archs):
        new_y = []
        for j,wm in enumerate(wms):
            # print('tetete',t)
            print(wm)
            new_y.append(result_dict[t][wm]['wm']['test_acc'])
        pos = np.array([i+1 for i in range(len(wms))]) - total_width * ( 1 - (2*i+4)/len(wms) )/2
        print(new_y)
        plt.bar(pos,new_y,width=total_width/len(wms))
    plt.xticks([i+1 for i in range(len(wms))],wms,fontsize=12.5)
    plt.ylabel('Accuracy',fontsize=16)
    plt.xlabel('Watermark Scheme',fontsize=16)
    plt.legend(archs_name,bbox_to_anchor=(1.0, 0),loc='lower right',borderaxespad=1,fontsize=16)
    plt.savefig(filename)
    plt.close()

def plot_embed_loss(filename, archs:list, wms:list, result_dict:dict, archs_name):
    margin = 0.01
    total_width = 1 - margin
    print('plot')
    # plt.figure(figsize=(8,5))
    # plt.figure(figsize=(16,9))
    plt.figure(figsize=(12,7))
    plt.title('The Embedding loss',fontsize=20)
    # labels = plt.get_xticklabels()
    # plt.setp(labels, rotation=30, fontsize=16);
    # plt.set_ylim(lim[0],lim[1])
    for i,t in enumerate(archs):
        new_y = []
        for j,wm in enumerate(wms):
            # print('tetete',t)
            print(wm)
            # before-after
            new_y.append(result_dict[t]['acc']['test_acc']/100.0-result_dict[t][wm]['wm']['test_acc'])
        pos = np.array([i+1 for i in range(len(wms))]) - total_width * ( 1 - (2*i+4)/len(wms) )/2
        print(new_y)
        plt.bar(pos,new_y,width=total_width/len(wms))
    plt.xticks([i+1 for i in range(len(wms))],wms,fontsize=14)
    plt.ylabel('Embedding Loss',fontsize=16)
    plt.xlabel('Watermark Scheme',fontsize=16)
    # plt.ylim([,])
    plt.legend(archs_name,bbox_to_anchor=(1.0, 0.9),loc='upper right',borderaxespad=1,fontsize=16)
    plt.hlines(0.05, 0, len(wms) + 0.1, linestyle="--", label="Limit of EmbeddingLoss")
    plt.savefig(filename)
    plt.close()

def plot_steal_losses(filename, base, archs:str, wms:list, attacks:list, result_dict:dict, archs_name):
    for arch in archs:
        plot_steal_loss(os.path.join(base,f'steal_loss_{arch}.png'),arch,wms,attacks,result_dict, archs_name)

def plot_steal_loss(filename, arch:str, wms:list, attacks:list, result_dict:dict, archs_name):
    margin = 0.01
    total_width = 1 - margin
    print('plot')
    # plt.figure(figsize=(8,5))
    plt.figure(figsize=(12,7))
    # plt.figure(figsize=(16,9))
    plt.title(f'{arch} stealing loss',fontsize=20)
    # labels = plt.get_xticklabels()
    # plt.setp(labels, rotation=30, fontsize=16);
    # plt.set_ylim(lim[0],lim[1])
    for i,wm in enumerate(wms):
        new_y = []
        for j,attack in enumerate(attacks):
            # print('tetete',t)
            print(result_dict[arch][wm]['atk'].keys())
            print(len(result_dict[arch][wm]['atk'].keys()))
            # before-after
            new_y.append(result_dict[arch][wm]['wm']['test_acc']-result_dict[arch][wm]['atk'][attack]['test_acc_after'])
        pos = np.array([i+1 for i in range(len(attacks))]) - total_width * ( 1 - (2*i+0)/(1+len(attacks)) )/2 -0.4
        print('thisis pos',pos)
        print(new_y)
        plt.bar(pos,new_y,width=total_width/len(attacks))
    new_attacks = [attack.replace('_',' ',1).replace('_','\n') for i, attack in enumerate(attacks)]
    plt.xticks([i+1-0.53 for i in range(len(attacks))],new_attacks,rotation=20,fontsize=14)
    # plt.subplots_adjust(left=0.15, right=0.85, bottom=0.20, top=0.85)
    plt.subplots_adjust(left=0.07, right=0.98, bottom=0.20, top=0.90)
    plt.ylabel('Stealing Loss',fontsize=16)
    plt.xlabel('RemovalAttack Scheme',fontsize=16)
    if arch == 'resnet':
        plt.legend(wms,bbox_to_anchor=(0.0, 0.0),loc='lower left',borderaxespad=1,fontsize=16)
    else:
        plt.legend(wms,bbox_to_anchor=(0.0, 1.0),loc='upper left',borderaxespad=1,fontsize=16)
    plt.hlines(0.05, 0, len(attacks) + 0.1, linestyle="--", label="Limit of StealingLoss")
    plt.savefig(filename)
    plt.close()

def plot_steal_wm_accs(filename, base, archs:str, wms:list, attacks:list, result_dict:dict, archs_name):
    for i,arch in enumerate(archs):
        plot_steal_wm_acc(os.path.join(base,f'stealing_wm_acc_{arch}.png'),arch,wms,attacks,result_dict, archs_name[i])

def plot_steal_wm_acc(filename, arch:str, wms:list, attacks:list, result_dict:dict, arch_name):
    margin = 0.01
    total_width = 1 - margin
    print('plot')
    # plt.figure(figsize=(8,5))
    plt.figure(figsize=(12,7))
    # plt.figure(figsize=(16,9))
    plt.title(f'{arch_name} Watermark Accuracy after removal attacks',fontsize=20)
    # labels = plt.get_xticklabels()
    # plt.setp(labels, rotation=30, fontsize=16);
    # plt.set_ylim(lim[0],lim[1])
    for i,wm in enumerate(wms):
        new_y = []
        for j,attack in enumerate(attacks):
            # print('tetete',t)
            print(result_dict[arch][wm]['atk'].keys())
            print(len(result_dict[arch][wm]['atk'].keys()))
            new_y.append(result_dict[arch][wm]['atk'][attack]['wm_acc_after'])
        pos = np.array([i+1 for i in range(len(attacks))]) - total_width * ( 1 - (2*i+0)/(1+len(attacks)) )/2 - 0.4
        print(new_y)
        plt.bar(pos,new_y,width=total_width/len(attacks))
    new_attacks = [attack.replace('_',' ',1).replace('_','\n') for i, attack in enumerate(attacks)]
    plt.xticks([i+1-0.53 for i in range(len(attacks))],new_attacks,rotation=20,fontsize=14)
    plt.subplots_adjust(left=0.07, right=0.98, bottom=0.20, top=0.90)
    plt.ylabel('Watermark Accuracy',fontsize=16)
    plt.xlabel('RemovalAttack Scheme',fontsize=16)
    # plt.legend(wms,bbox_to_anchor=(1.0, 0),loc='lower right',borderaxespad=1,fontsize=16)
    plt.legend(wms,bbox_to_anchor=(1.0, 1.0),loc='upper right',borderaxespad=1,fontsize=16)
    plt.savefig(filename)
    plt.close()

def plot_wm_acc(filename, archs:list, wms:list, result_dict:dict, archs_name):
# def plot_wm_acc(filename, arch:str, wms:list, attacks:list, result_dict:dict):
    margin = 0.01
    total_width = 1 - margin
    print('plot')
    # plt.figure(figsize=(8,5))
    plt.figure(figsize=(12,7))
    plt.title('Watermark Accuracy',fontsize=20)
    # labels = plt.get_xticklabels()
    # plt.setp(labels, rotation=30, fontsize=16);
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
    plt.xticks([i+1 for i in range(len(wms))],wms,fontsize=14)
    plt.ylabel('Watermark Accuracy',fontsize=16)
    plt.xlabel('Watermark Scheme',fontsize=16)
    plt.legend(archs_name,bbox_to_anchor=(0.0, 0.0),loc='lower left',borderaxespad=1,fontsize=16)
    plt.savefig(filename)
    plt.close()

def plot_wm_acc_extend_jia(input,output):
    result = {}
    try:
        with open(input) as f:
            result = json.load(f)
    except:
        import traceback
        traceback.print_exc()

    margin = 0.01
    total_width = 1 - margin
    print('plot')
    # plt.figure(figsize=(8,5))
    plt.figure(figsize=(12,7))
    # plt.figure(figsize=(16,9))
    plt.title(f'Watermark Accuracy after embedding',fontsize=20)
    keys = result.keys()
    new_y = []
    for i, key in enumerate(result.keys()):
        if 'wm_acc' not in result[key]:
            continue
        new_y.append(result[key]['wm_acc'])
        plt.bar(i,new_y[i],tick_label=key)
    # plt.bar([i for i in range(len(new_y))],new_y)
    plt.xticks([i for i in range(len(keys))],[key for key in keys],rotation=20,fontsize=13)
    plt.subplots_adjust(left=0.15, right=0.85, bottom=0.20, top=0.85)
    plt.ylabel('Watermark Accuracy',fontsize=13)
    plt.legend(keys,bbox_to_anchor=(1.0, 0),loc='lower right',borderaxespad=1,fontsize=16)
    plt.savefig(output)
    plt.close()
    
def plot_embed_loss_extend_jia(input,output):
    result = {}
    try:
        with open(input) as f:
            result = json.load(f)
    except:
        import traceback
        traceback.print_exc()

    margin = 0.01
    total_width = 1 - margin
    print('plot')
    plt.figure(figsize=(12,7))
    # plt.figure(figsize=(16,9))
    plt.title(f'Embedding Loss after embedding',fontsize=20)
    keys = result.keys()
    new_y = []
    new_key = []
    for i, key in enumerate(result.keys()):
        if 'wm_acc' not in result[key]:
            continue
        if 'before_acc' not in result[key]:
            continue
        if 'test_acc' not in result[key]:
            continue
        new_y.append(result[key]['before_acc']/100.0-result[key]['test_acc'])
        new_key.append(key)
        # plt.bar([j for j in range(len(new_y))],new_y)
        plt.bar(i,new_y[i],tick_label=key)
    plt.xticks([i for i in range(len(new_key))],[key for key in new_key],rotation=20,fontsize=14)
    # plt.subplots_adjust(left=0.15, right=0.85, bottom=0.20, top=0.85)
    plt.ylabel('Embedding Loss',fontsize=16)
    # plt.xlabel('Watermark Scheme',fontsize=16)
    print('new_key is ',new_key)
    plt.legend(new_key,bbox_to_anchor=(1.0, 0),loc='lower right',borderaxespad=1,fontsize=16)
    plt.hlines(0.05, -0.4, len(new_y) + 0.1, linestyle="--", label="Limit of EmbeddingLoss")
    plt.savefig(output)
    plt.close()

def plot_wm_acc_extend_jia(input,output):
    result = {}
    try:
        with open(input) as f:
            result = json.load(f)
    except:
        import traceback
        traceback.print_exc()

    margin = 0.01
    total_width = 1 - margin
    print('plot')
    plt.figure(figsize=(12,7))
    # plt.figure(figsize=(16,9))
    plt.title(f'Watermark Accuracy after embedding',fontsize=20)
    keys = result.keys()
    new_y = []
    for i, key in enumerate(result.keys()):
        if 'wm_acc' not in result[key]:
            continue
        new_y.append(result[key]['wm_acc'])
        plt.bar(i,new_y[i],tick_label=key)
    # plt.bar([i for i in range(len(new_y))],new_y)
    plt.xticks([i for i in range(len(keys))],[key.replace(' ','\n') for key in keys],rotation=20,fontsize=13)
    # plt.subplots_adjust(left=0.15, right=0.85, bottom=0.20, top=0.85)
    # plt.ylabel('Watermark Accuracy',fontsize=16)
    plt.legend(keys,bbox_to_anchor=(1.0, 0),loc='lower right',borderaxespad=1,fontsize=16)
    plt.savefig(output)
    plt.close()
    
def plot_decision_threshold_dist(filename, archs:list, wms:list, result_dict:dict, archs_name):
# def plot_wm_acc(filename, arch:str, wms:list, attacks:list, result_dict:dict):
    ext = '.png'
    margin = 0.01
    total_width = 1 - margin
    print('plot')
    plt.figure(figsize=(8,5.2))
    # plt.title('Watermark Accuracy',fontsize=16)
    # labels = plt.get_xticklabels()
    # plt.setp(labels, rotation=30, fontsize=16);
    # plt.set_ylim(lim[0],lim[1])
    for j,wm in enumerate(wms):
        for i,t in enumerate(archs):
            if (wm!='frontier' and wm!='blackmarks') and t != 'resnet':
                continue
            new_y = [[],[],[],[]]
            print(result_dict[t][wm]['decision_thresholds'].keys())
            y = result_dict[t][wm]['decision_thresholds']['wm_acc']
            y = np.array(y[0])
            y = y.reshape(20,100)
            # print(y)
            # print(y.shape)
            array = [[],[],[],[]]
            idx = [0,1,0,1,0,1,0,1,0,1,2,2,2,2,2,3,3,3,3,3]
            idx = [3,2,3,2,3,2,3,2,3,2,1,1,1,1,1,0,0,0,0,0]
            for k,index in enumerate(idx):
                new_y[index] += list(y[k])
            print(len(idx))
            # print(new_y)
            plt.figure(figsize=(8,5))
            if (wm=='frontier' or wm=='blackmarks'):
                plt.title(f'decision_thresholds distribution in {wm} \n using {archs_name[i]}: {round(result_dict[t][wm]["decision_thresholds"]["y"][0]/100.0,5)}',fontsize=19)
                print(result_dict[t][wm]["decision_thresholds"]["y"])
            else:
                plt.title(f'decision_thresholds distribution in {wm}\n: {round(result_dict[t][wm]["decision_thresholds"]["y"][0]/100.0,5)}',fontsize=19)
                print('result_dict',result_dict[t][wm]["decision_thresholds"]["y"])
            plt.hist(new_y, stacked=True,label=archs_name,bins=100,range=(0,100))
            plt.xlabel('Watermark Accuracy',fontsize=13)
            plt.legend(fontsize=16)
            # plt.hist(x=new_y,bins=20)
            # plt.show()
            plt.savefig(filename+f'{t}_{wm}'+ext)
            plt.close()
            # return
            # print('tetete',t)
            print(wm)
            new_y.append(result_dict[t][wm]['wm']['wm_acc'])
        pos = np.array([i+1 for i in range(len(wms))]) - total_width * ( 1 - (2*i+4)/len(wms) )/2
        # plt.bar(pos,new_y,width=total_width/len(wms))
    plt.xticks([i+1 for i in range(len(wms))],wms,fontsize=12.5)
    plt.ylabel('Watermark Accuracy',fontsize=16)
    plt.legend(archs_name,bbox_to_anchor=(0.5, 0.0),loc='lower center',borderaxespad=1,fontsize=16)
    plt.savefig(filename)
    plt.close()

def plot_decision_threshold_dist_baseline(input,output):
    result = {}
    try:
        with open(input,'r') as f:
            result = json.loads(f.read())
    except:
        import traceback
        traceback.print_exc()
    ext = '.png'
    margin = 0.01
    print('plot')
    plt.figure(figsize=(8,5))
    for j,wm in enumerate(result.keys()):
        y = result[wm]['data']
        # print(y)
        # print(y.shape)
        plt.figure(figsize=(8,5))
        plt.title(f'decision_thresholds distribution in {wm} \n Baseline: {round(result[wm]["y"]/100.0,5)}', fontsize=19)
        plt.xlabel('Watermark Accuracy',fontsize=13)
        plt.hist(y, stacked=True,bins=100,range=(0,100))
        # plt.legend(fontsize=16)
        # plt.hist(x=new_y,bins=20)
        # plt.show()
        plt.savefig(output+f"{wm}"+ext)
        plt.close()
        # return
        # print('tetete',t)
        print(wm)
    plt.close()

if __name__ == "__main__":
    path = ['./outputs/cifar10/json/result_00016.json','./outputs/cifar10/json/result_00017.json','./outputs/cifar10/json/result_00018.json','./outputs/cifar10/json/result_00019.json']
    load_result_from_jsons(path)
    plot_decision_threshold_dist_baseline('./outputs/cifar10/atk_json/alldt.json','outputs/cifar10/png/dtbaseline')
    plot_wm_acc_extend_jia('./outputs/cifar10/atk_json/extend.json','test2.png')
    plot_embed_loss_extend_jia('./outputs/cifar10/atk_json/extend.json','test.png')
