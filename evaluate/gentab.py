from glob import glob
def get_mae(trainds, testds, net):
    path = 'results_train_on_%s/results_*_to_%s/summary_*_test_of_net%s_*' %(trainds, testds, net)
    #print(path)
    fp = glob(path)[0]
    with open(fp) as f:
        l = next(f)
        l = next(f)
        l = next(f)
        assert(l.strip().startswith('Mean absolute error'))
        l = l.strip().split(':')
        val = float(l[1].strip())
    return val

nets={
    'vgg16':'VGG',
    'densenet121bc':'DenseNet',
    'senet50':'SENet',
    'mobilenetv3small':'MobileNet-Small',
    'mobilenetv3large':'MobileNet-Large',
}
training_sets={
    'imdbwiki_cleaned':'\\scriptsize{IMDB-Wiki}',
    'vggface2':'VMAGE',
}
test_sets={
    'imdbwikitest':'\\scriptsize{IMDB-Wiki}',
    'vggface2test':'VMAGE',
}

data = []

print('\\multirow{2}{*}{\\bf{Method}} & \\multirow{%d}{*}{\\bf{Training}} & \\multicolumn{2}{c|}{\\bf{MAE}} \\ '%len(test_sets))
print(' & & ', '&'.join('\\bf{%s}'%s for s in test_sets.values()), '\\ \hline')
for net,net_label in nets.items():
    datarow1 = []
    for train, train_label in training_sets.items():
        datarow = []
        print(net_label, '&', train_label, end='')
        for test, test_label in test_sets.items():
            datarow.append(mae)
            mae = get_mae(train, test, net)
            print(' & %.2f'%mae, end='')
        datarow1.append(datarow)
        print()
    data.append(datarow1)
data = np.array(data)
#G:\My Drive\phd\Sperimentazioni age MIVIA\results_train_on_imdbwiki_cleaned\results_imdbwiki_to_vggface2test\summary_vggface2_test_of_netdensenet121bc_datasetimdbwiki_age_pretrainingimagenet_preprocessingvggface2_augmentationdefault_batch32_lr0.0001_0.2_30_training-epochs90_momentum_sel_gpu1_20200717_114434.txt

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def doplot(data, series_labels, series_styles, title, fname, ylim, draw_line=False, figw=5, figh=4.4):
    assert( series_labels is None or len(series_labels)==len(series_styles))

    colors = {
        'VGG': 'crimson',
        'SENet': 'royalblue',
        'DenseNet': 'darkolivegreen',
        'MobileNet': 'chocolate'
    }

    def get_format(name):
        for k,v in colors.items():
            if k in name:
                color=v
        return color

    def mk_legend(ax):
        from matplotlib.patches import Patch
        custom_lines = []
        meanings = []
        for l,s in zip(series_labels, series_styles):
            custom_lines.append(Patch(edgecolor=s[0], facecolor=s[1], hatch=s[2]))
            meanings.append(l)
        ax.legend(custom_lines, meanings, loc='upper left', framealpha=0.3)
    SERIES_WIDTH=0.8
    w= SERIES_WIDTH / len(series_styles)
    
    labels = ['VGG', 'SENet', 'DenseNet', 'MobileNet']
    pos = np.array([0,1,2,3]) 

    fig, ax = plt.subplots()
    #ax.set_title(title)
    plt.ylabel(title)
    for i,d in enumerate(data[:, 0]):
        color= get_format(labels[i])
        for j,s in enumerate(series_styles):
            barw = w-0.3 if len(series_styles)==1 else w-0.04
            plt.bar(w*j+pos[i], data[i,j], width=barw, 
                edgecolor='w' if s[0]=='w' else color,
                linewidth = 0 if s[0]=='w' else None,
                facecolor='w' if s[1]=='w' else color,
                hatch=s[2])
    ax.set_xticklabels(labels)
    ax.set_xticks(pos-w/2 + SERIES_WIDTH/2)
    if series_labels is not None:
        pass#mk_legend(ax)
    plt.ylim(ylim)
    if draw_line:
        ynew = 1
        ax.axhline(ynew, color='gray', linewidth=0.5)
        yt = ax.get_yticks()
        yt = np.append(yt,ynew)
        ax.set_yticks(yt)
    if figw<6:
        plt.setp( ax.xaxis.get_majorticklabels(), rotation=45, ha='right', rotation_mode='anchor' ) 
    fig.set_size_inches(figw, figh)
    plt.subplots_adjust(top=0.98, bottom=0.145, left=0.28*2/figw, right=1-(0.03*2/figw))
    plt.savefig(fname[:-4]+'.'+OUTEXT)
    #plt.show()


REL_YLIM = [0.65, 1.25]
ERR_YLIM = [12, 18]