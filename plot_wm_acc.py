import matplotlib.pyplot as plt
import json
import os
import numpy as np

def load_result_from_jsons(json_paths,arch=['WideResNet','ViT','R50+ViT','R50+ViT_WPM']):
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
    plot_wm_acc(os.path.join(base,'wm_acc.png'),'WatermarkAccuracy',labels,result,'WatermarkAccuracy','wm_acc',arch)
    plot_wm_acc(os.path.join(base,'acc.png'),'Accuracy',labels,result,'Accuracy','test_acc',arch)
    plot_wm_acc(os.path.join(base,'time.png'),'Time',labels,result,'time[s]','time',arch)
    return

# def plot_wm_acc(filename,main_title,x,y,titles,xlabel,ylabel,key,lim=(0.0,1.0)):
def plot_wm_acc(filename,main_title,x,y,ylabel,key,arch,lim=(0.0,1.0)):
    margin = 0.01
    total_width = 1 - margin
    print('plot')
    plt.figure(figsize=(8,5))
    plt.title(main_title)
    # labels = plt.get_xticklabels()
    # plt.setp(labels, rotation=30, fontsize=10);
    # plt.set_ylim(lim[0],lim[1])
    for i,t in enumerate(y):
        new_y = []
        for j in range(len(x)):
            # print('tetete',t)
            for k,res in enumerate(t['result']):
                if res['name'] == x[j]:
                    new_y.append(res['result'][key])
                    break
        pos = np.array([i+1 for i in range(len(x))]) - total_width * ( 1 - (2*i+4)/len(x) )/2
        plt.bar(pos,new_y,width=total_width/len(x))
    plt.xticks([i+1 for i in range(len(x))],x)
    plt.ylabel(ylabel)
    plt.legend(arch,bbox_to_anchor=(1.0, 0),loc='lower right',borderaxespad=1,fontsize=10)
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    path = ['./outputs/cifar10/json/result_00016.json','./outputs/cifar10/json/result_00017.json','./outputs/cifar10/json/result_00018.json','./outputs/cifar10/json/result_00019.json']
    load_result_from_jsons(path)
