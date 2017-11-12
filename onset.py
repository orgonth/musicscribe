"""
Train and test the onset detector CNN
"""

import numpy as np
import database
import generator
import eda

import os
from glob import glob

import keras
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

def build_model():
    """Builds the onset detector CNN model"""
    model = Sequential()

    model.add(Conv2D(filters=16,
                     kernel_size=(7,3),
                     strides=(2,1),
                     padding='same', activation='relu',
                     kernel_initializer='uniform',
                     input_shape=(15,80,3)))

    model.add(MaxPooling2D(pool_size=(1,3), padding='same'))
    model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(1,3), padding='same'))
    #model.add(Conv2D(filters=40, kernel_size=(2,2), padding='same', activation='relu'))
   # model.add(MaxPooling2D(pool_size=(3,3), padding='same'))

    model.add(Dropout(0.2))
    model.add(Flatten())
    #model.add(Dropout(0.2))
    model.add(Dense(256, activation='sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(lr=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

def train_model(model, db, output=None, epochs=300):
    """
    Train an onset model on a database
    :param model: model to train
    :param db: database to train on
    :param output: output filename of the best trained model
    :param epochs: number of iterations
    :return: train history
    """
    if output==None:
        dbname = db.name()
        output = dbname[:dbname.rindex('.')] + '_onset.hdf5'

    checkpointer = ModelCheckpoint(filepath=output, 
                                   verbose=1, save_best_only=True)

    train_group = ['train','mel']
    xgroup = db.get_subgroup(['train','mel'])
    ygroup = db.get_subgroup(['train','onset_labels'])

    step = min(10000, db.get_total_points(train_group))
    
    frac_val = 0.2
    frac = 1.-frac_val

    nb_steps, tmp = generator.get_nb_steps(xgroup, step, frac, dict_arrays=True)
    nb_steps_val, tmp = generator.get_nb_steps(xgroup, step, frac_val, shift='end', dict_arrays=True)

    print('step: ',step)
    print('nb_steps: ',nb_steps, nb_steps_val)
    
    hist = model.fit_generator(generator.generator( (xgroup, ygroup),
                             nb=step,
                             frac=frac,
                             dict_arrays=True),
                        
                        steps_per_epoch= 50,#4,#max(4,nb_steps),
                        max_queue_size=1,
        
                        validation_data= generator.generator( (xgroup, ygroup),
                                                              nb=step,
                                                              frac=frac_val,
                                                              shift='end',
                                                              dict_arrays=True),     
                        validation_steps= nb_steps_val,
                        
                        epochs=epochs,
                        callbacks=[checkpointer],
                        verbose=2
                        )

    return hist.history

def compute_tfpn_categories(categories, predictions, labels, res):
    """
    Compute True/False Positives/Negatives per category
    :param categories: list of categories
    :param predictions:
    :param labels:
    :param res: counter of true/false positive/negative that will be increased
    :return: res
    """
    x = predictions
    y = labels
    for i in range(len(predictions)):
        nbn_i = categories[i]
        if nbn_i not in res:
            res[nbn_i] = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'total': 0}
        if x[i] == y[i]:
            if x[i] == 1:
                res[nbn_i]['tp'] += 1
            else:
                res[nbn_i]['tn'] += 1
        else:
            if x[i] == 1:
                res[nbn_i]['fp'] += 1
            else:
                res[nbn_i]['fn'] += 1
        res[nbn_i]['total'] += 1
    return res

def compute_scores_categories(tfpn):
    """
    Compute accuracy, precision, recall, and f1 score per category
    :param tfpn: the true/false positive/negative counter (dict)
    :return: accuracy, precision, recall, f1, total number of samples
    """
    accu = (tfpn['tp']+tfpn['tn'])/tfpn['total']
    prec = np.divide(tfpn['tp'], (tfpn['tp']+tfpn['fp']))
    reca = np.divide(tfpn['tp'], (tfpn['tp']+tfpn['fn']))
    f1 = np.divide(2*prec*reca, (prec+reca))
    return accu, prec, reca, f1, tfpn['total']

def compute_scores(model, set_name, subset, log, categories=False):
    """
    Do predictions on a dataset and compute scores
    :param model: model to test
    :param set_name: string in the form of set_1 (must be in data folder)
    :param subset: string in the form of 'test', 'train' (must be in the set_name folder)
    :param log: opened log file to write results into
    :param categories: set to True to compute scores per category
    :return: None
    """
    filenames = glob('data/{}/{}/*.mid'.format(set_name, subset))
    basenames = []
    for f in filenames:
        basenames.append(os.path.splitext(os.path.basename(f))[0])

    db = database.DatabaseReader(f'data_{set_name}.hdf5')

    print("compute scores for : {} / {}".format(set_name, subset))
    dgroup = db.get_subgroup([subset, 'mel'])
    lgroup = db.get_subgroup([subset, 'onset_labels'])
    cgroup = db.get_subgroup([subset, 'onset_caracs'])

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    nb_total = 0

    carac_nb_notes = {}
    carac_volume = {}
    carac_silence_notes = {}

    res = {}
    for name in dgroup:
        test_preds = 1. * (model.predict(dgroup[name]) >= 0.5)
        y_test = lgroup[name]

        x = test_preds[:,0]
        y = y_test[:]
        tp = np.sum(x*y)
        tn = np.sum((1-x)*(1-y))
        fp = np.sum(x) - tp
        fn = np.sum(1-x) - tn
        
        true_positive += tp
        true_negative += tn
        false_positive += fp
        false_negative += fn
        nb_total += len(y)

        #

        c = cgroup[name]
        compute_tfpn_categories(c[:, 0], x, y, carac_nb_notes)
        compute_tfpn_categories(c[:, 1], x, y, carac_volume)
        compute_tfpn_categories(c[:, 4], x, y, carac_silence_notes)

        score = (accuracy_score(y_test, test_preds),
                 precision_score(y_test, test_preds),
                 recall_score(y_test, test_preds),
                 f1_score(y_test, test_preds)
                 )

        for f, b in zip(filenames, basenames):
            if b in name:
                break

        stats = eda.analyze(f)
        stats_simp = [len(y_test)]

        for s in stats:
            stats_simp.append(s[0][s[1].argmax()])

        res[name] = (score, stats_simp)

        print(f'--- {name} ---')
        print("accuracy    {:.4f}".format(score[0]))
        print("precision   {:.4f}".format(score[1]))
        print("recall      {:.4f}".format(score[2]))
        print("F1-score    {:.4f}".format(score[3]))

        log.write(name)
        for r in score:
            log.write('\t{:.4f}'.format(r))
        for s in stats_simp:
            log.write('\t{}'.format(s))
        log.write('\n')
        log.flush()

    ###

    log.write(f'TOTAL_{set_name}_{subset}')

    print(true_positive,true_negative,false_positive,false_negative,nb_total)
    
    accu = (true_positive+true_negative)/nb_total
    prec = np.divide(true_positive, (true_positive+false_positive))
    reca = np.divide(true_positive, (true_positive+false_negative))
    f1 = np.divide(2*prec*reca, (prec+reca))

    for r in (accu, prec, reca, f1):
        log.write('\t{:.4f}'.format(r))
    for s in stats_simp:
            log.write('\t-')
    log.write('\n')
    log.flush()

    ###

    if categories:
        for carac_s in ('carac_nb_notes', 'carac_volume', 'carac_silence_notes'):
            carac = eval(carac_s)
            for k in sorted(carac):
                log.write(f'{carac_s}\t{k}')
                res = compute_scores_categories(carac[k])
                for r in res:
                    log.write(f'\t{r}')
                log.write('\n')
        log.flush()

def train_sets(sets, epochs):
    """
    Train several models on datasets (each model is trained on one dataset)
    :param sets: list of datasets
    :param epochs: number of iterations
    :return: None
    """
    for s in sets:
        print(f'-------- Training on {s} --------')
        db = database.DatabaseReader(f'data_{s}.hdf5')
        model = build_model()
        # model.summary()
        hist = train_model(model, db, output=f'best_onset_{s}.hdf5', epochs=epochs)
        with open(f'best_onset_{s}_training.log','w') as log:
            log.write('epoch\t')
            for k in hist:
                log.write(k+'\t')
                nb = len(hist[k])
            log.write('\n')
            for i in range(nb):
                log.write(f'{i+1}\t')
                for k in hist:
                    log.write(f'{hist[k][i]}\t')
                log.write('\n')

def test_sets(sets, doTrain=False, categories=False):
    """
    Test models on datasets. Each model is tested on all datasets.
    :param sets: list of datasets
    :param doTrain: set to True to also test on training set
    :param categories: set to True to also compute scores per category
    :return: None (writes log files)
    """
    for s in sets:
        with open(f'best_onset_{s}_predictions.log', "w") as log:
            print(f'-------- Predictions for model trained on {s} --------')
            log.write('song\tacc\tprec\trecall\tf1\tnb_total\tstroke\tnote\tvolume\tnb\ttempo\n')
            model = keras.models.load_model(f'best_onset_{s}.hdf5')
            for s2 in sets:
                print(f' -> computing predictions on {s2}')
                if doTrain:
                    compute_scores(model, s2, 'train', log, categories)
                compute_scores(model, s2, 'test', log, categories)
                
if __name__=='__main__':

    sets = ['set_1',
            'set_2',
            'set_3',
            'set_4'
            ]

   # train_sets(sets, epochs=100)
    test_sets(sets, doTrain=True, categories=False)
   # test_sets(sets, doTrain=False, categories=True)
