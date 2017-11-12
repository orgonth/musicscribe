"""
Plot training/validation loss and F1-score per category.
TODO: generalize the plotter for different situations than 4 models...
"""

import matplotlib.pyplot as plt
from glob import glob
import re

def plot_training_loss(dirname, model):
    """
    Plot training and validation loss during training
    :param dirname: name of directory containing the logs (..._training.log)
    :param model: either onset or note
    """
    f, ax = plt.subplots(2, 2, sharex=True, sharey=True)

    sets = ['set_1',
            'set_2',
            'set_3',
            'set_4'
            ]

    idx = 0
    for set in sets:
        res = {}
        with open(f'{dirname}/best_{model}_{set}_training.log') as fin:
            headers = fin.readline().strip().split()
            for h in headers:
                res[h] = []
            for line in fin:
                i = 0
                for v in line.strip().split():
                    res[headers[i]].append(float(v))
                    i += 1

        u = idx//2
        v = idx%2
        ax[u][v].plot(res['epoch'], res['loss'], label='training')
        ax[u][v].plot(res['epoch'], res['val_loss'], label='validation')
        handles, labels = ax[u][v].get_legend_handles_labels()
        ax[u][v].legend(handles, labels)
        ax[u][v].set_title('model_{}'.format(set[-1]))

        if u==1:
            ax[u][v].set_xlabel('epoch')
        if v==0:
            ax[u][v].set_ylabel('loss')

        idx += 1
    plt.show()

def plot_category_score(dirname):
    """
    Plot F1-score per category
    :param dirname: directory containing the logs (..._predictions.log)
    """
    data = {}
    categories = []

    for f in sorted(glob(dirname+'/*_predictions.log')):
        model_set = re.search('set_[0-9]',f).group()

        with open(f) as fin:
            res = {}
            for line in fin:
                if line[:5]=='TOTAL':
                    set = re.search('set_[0-9]',line).group()
                    res[set] = {}
                if line[:5]=='carac':
                    l = line.strip().split('\t')
                    category = l[0][6:]
                    if category not in categories:
                        categories.append(category)
                    if category not in res[set]:
                        res[set][category] = {'x': [], 'f1': []}
                    f1 = l[5]
                    if f1 != 'nan':
                        res[set][category]['x'].append(float(l[1]))
                        res[set][category]['f1'].append(float(f1))

        data[model_set] = res

    for cat in categories:
        f, ax = plt.subplots(2, 2, sharex=True, sharey=True)
        idx = 0
        for model in data:
            u = idx // 2
            v = idx % 2
            idx += 1
            for set in data[model]:
                ax[u][v].plot(data[model][set][cat]['x'], data[model][set][cat]['f1'], label=set)

            if u==1 and v==1:
                handles, labels = ax[u][v].get_legend_handles_labels()
                ax[u][v].legend(handles, labels)

            ax[u][v].set_title('model_{}'.format(model[-1]))

            if cat=='silence_notes':
                ax[u][v].set_xlim(0,50)

            if u==1:
                ax[u][v].set_xlabel(cat)
            if v==0:
                ax[u][v].set_ylabel('F1-score')

    plt.show()

if __name__=='__main__':
    plot_training_loss('best_models', 'onset')
    plot_training_loss('best_models', 'note')
    plot_category_score('best_models/categories_scores_onset')
    plot_category_score('best_models/categories_scores_note')
