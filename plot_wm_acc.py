import matplotlib.pyplot as plt
import json
import os

def load_result_from_json(json_path):
    if not os.path.isfile(json_path):
        return
    
    try:
        with open(json_path,'r') as f:
            result = json.loads(f.read())
    except:
        import traceback
        traceback.print_exc()

    vit_result = []
    res_result = []
    for i, res in enumerate(result['result']):
        if 'vit' in res['name']:
            vit_result.append((res['name'].replace('_vit',''),res['result']['test_acc'],res['result']['wm_acc'],res['result']['time']))
        else:
            res_result.append((res['name'],res['result']['test_acc'],res['result']['wm_acc'],res['result']['time']))
    base = 'outputs/cifar10/png/'
    plot_wm_acc(os.path.join(base,'acc.png'),'Accuracy',[d[0] for d in vit_result],[d[1] for d in vit_result],[d[0] for d in res_result],[d[1] for d in res_result],'ViT-16_B','WideResNet','Methods','test_acc')
    plot_wm_acc(os.path.join(base,'wm_acc.png'),'Watermark Accuracy',[d[0] for d in vit_result],[d[2] for d in vit_result],[d[0] for d in res_result],[d[2] for d in res_result],'ViT-16_B','WideResNet','Methods','wm_acc')
    plot_wm_acc(os.path.join(base,'time.png'),'TIME',[d[0] for d in vit_result],[d[3] for d in vit_result],[d[0] for d in res_result],[d[3] for d in res_result],'ViT-16_B','WideResNet','Methods','time [s]')

def plot_wm_acc(filename,main_title,x1,y1,x2,y2,title1,title2,xlabel,ylabel,lim=(0.0,1.0)):
    print('plot')
    fig = plt.figure(figsize=(16,9))
    # plt.title(main_title)
    ax = fig.add_subplot(121)
    p = ax.bar(x1,y1,color=(0.2, 0.4, 0.6, 0.6))
    ax.set_title(title1)
    ax.set_xlabel(xlabel)
    labels = ax.get_xticklabels()
    plt.setp(labels, rotation=30, fontsize=10);
    ax.set_ylabel(ylabel)
    ax.set_ylim(lim[0],lim[1])
    ax.bar_label(p, label_type='center')
    # ax.legend()
    ax = fig.add_subplot(122)
    p = ax.bar(x2,y2,color=(0.2, 0.4, 0.6, 0.6))
    ax.set_title(title2)
    ax.set_xlabel(xlabel)
    labels = ax.get_xticklabels()
    plt.setp(labels, rotation=30, fontsize=10);
    ax.set_ylabel(ylabel)
    ax.set_ylim(lim[0],lim[1])
    ax.bar_label(p, label_type='center')
    # ax.legend()
    # plt.show()
    plt.savefig(filename)

if __name__ == "__main__":
    path = './outputs/cifar10/json/result_00004.json'
    load_result_from_json(path)
