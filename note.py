"""
Train and test the key identification CNN
"""

import database
import generator
import eda

from glob import glob
import os

import numpy as np

import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, add
from keras.layers import Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

def build_model():
    """Builds the key identification CNN model"""
    # o = 0.0388

    inputs = Input(shape=(15,155,1))

    nb_filters = 32

    # 15 x 155 x 1

    x = Conv2D(filters=nb_filters, kernel_size=(3,3), padding='same', activation='relu')(inputs)
    x = Conv2D(filters=2*nb_filters, kernel_size=(3,3), padding='same', activation='relu')(x)
    x = Dropout(0.2)(x)
    pool = MaxPooling2D(pool_size=(1,2), padding='same')(x)

    # 15 x 78 x 32

    x = Conv2D(filters=nb_filters, kernel_size=(3,3), padding='same', activation='relu')(pool)
    x = Conv2D(filters=2*nb_filters, kernel_size=(3,3), padding='same', activation='relu')(x)
    x = Dropout(0.2)(x)
    x = add([pool,x])
    pool = MaxPooling2D(pool_size=(1,2), padding='same')(x)

    # 15 x 39 x 32

    x = Conv2D(filters=nb_filters, kernel_size=(3,3), padding='same', activation='relu')(pool)
    x = Conv2D(filters=2*nb_filters, kernel_size=(3,3), padding='same', activation='relu')(x)
    x = Dropout(0.2)(x)
    x = add([pool,x])
    pool = MaxPooling2D(pool_size=(1,2), padding='same')(x)

    # 15 x 20 x 32

    x = Conv2D(filters=nb_filters, kernel_size=(3,3), padding='same', activation='relu')(pool)
    x = Conv2D(filters=2*nb_filters, kernel_size=(3,3), padding='same', activation='relu')(x)
    x = Dropout(0.2)(x)
    x = add([pool,x])
    pool = MaxPooling2D(pool_size=(2,2), padding='same')(x)

    # 8 x 10 x 32

    x = Conv2D(filters=nb_filters, kernel_size=(3,3), padding='same', activation='relu')(pool)
    x = Conv2D(filters=2*nb_filters, kernel_size=(3,3), padding='same', activation='relu')(x)
    x = Dropout(0.2)(x)
    x = add([pool,x])
    #pool = MaxPooling2D(pool_size=(2,2), padding='same')(x)
    pool = x

    # 8 x 10 x 64

    x = Conv2D(filters=2*nb_filters, kernel_size=(3,3), padding='same', activation='relu')(pool)
    x = add([pool,x])
    x = Conv2D(filters=4*nb_filters, kernel_size=(3,3), padding='same', activation='relu')(x)
    x = Dropout(0.2)(x)
    pool = MaxPooling2D(pool_size=(2,2), padding='same')(x)

    # 4 x 5 x 128

    x = Conv2D(filters=4*nb_filters, kernel_size=(2,2), padding='same', activation='relu')(pool)
    x = add([pool,x])
    x = Conv2D(filters=8*nb_filters, kernel_size=(2,2), padding='same', activation='relu')(x)
    x = Dropout(0.2)(x)
    pool = MaxPooling2D(pool_size=(2,2), padding='same')(x)

    # 2 x 3 x 256
        
    x = Dropout(0.2)(pool)
    x = Flatten()(x)
##    x = Dense(512, activation='relu')(x)
##    x = Dropout(0.2)(x)
##    x = Dense(256, activation='relu')(x)
##    x = Dropout(0.2)(x)
    predictions = Dense(88, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=predictions)

    model.compile(optimizer=Adam(),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model
    

def train_model(model, db, output=None, epochs=300):
    """
    Train a key identification model on a database
    :param model: model to train
    :param db: database to train on
    :param output: output filename of the best trained model
    :param epochs: number of iterations
    :return: train history
    """
    if output==None:
        dbname = db.name()
        output = dbname[:dbname.rindex('.')] + '_note.hdf5'
    
    #model.summary()

    checkpointer = ModelCheckpoint(filepath=output, 
                                   verbose=1, save_best_only=True)

    train_group = ['train','cqt']
    xgroup = db.get_subgroup(['train','cqt'])
    ygroup = db.get_subgroup(['train','note_labels'])

    step = min(500, db.get_total_points(train_group))
    
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
                        
                        steps_per_epoch= 100,#max(4,nb_steps),
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

    db = database.DatabaseReader('data_{}.hdf5'.format(set_name))

    print("compute scores for : {} / {}".format(set_name, subset))
    dgroup = db.get_subgroup([subset, 'cqt'])
    lgroup = db.get_subgroup([subset, 'note_labels'])

    res = {}

    true_positive = np.zeros(88)
    true_negative = np.zeros(88)
    false_positive = np.zeros(88)
    false_negative = np.zeros(88)
    nb_class = np.zeros(88)
    
    nb_total = 0

    tp_total = 0
    tn_total = 0
    fp_total = 0
    fn_total = 0
    
    for name in dgroup:
        test_preds = 1. * (model.predict(dgroup[name]) >= 0.5)
        y_test = lgroup[name]

        x = test_preds
        y = y_test[:]
        z = y + 2*x

        # foreach note (88 arrays):
        true_positive += np.sum(z==3, axis=0)
        true_negative += np.sum(z==0, axis=0)
        false_positive += np.sum(z==2, axis=0)
        false_negative += np.sum(z==1, axis=0)
        nb_class += np.sum(y, axis=0)
        
        nb_total += x.shape[0]

        tp_total += np.sum(true_positive)
        tn_total += np.sum(true_negative)
        fp_total += np.sum(false_positive)
        fn_total += np.sum(false_negative)

        score = (accuracy_score(y_test, test_preds),
                 precision_score(y_test, test_preds, average='weighted'),
                 recall_score(y_test, test_preds, average='weighted'),
                 f1_score(y_test, test_preds, average='weighted'))

        for f, b in zip(filenames, basenames):
            if b in name:
                break

        stats = eda.analyze(f)
        stats_simp = []

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

    log.write(f'TOTAL_{set_name}_{subset}')

    # score per class
    accu = (true_positive+true_negative)/nb_total
    prec = np.divide(true_positive, (true_positive+false_positive))
    reca = np.divide(true_positive, (true_positive+false_negative))
    f1 = np.divide(2*prec*reca, (prec+reca))

    # macro score (unused)
    macro_f1 = np.nanmean(f1)

    # micro score (unused)
    micro_prec = tp_total/(tp_total+fp_total)
    micro_reca = tp_total/(tp_total+fn_total)
    micro_f1 = 2*micro_prec*micro_reca/(micro_reca+micro_prec)

    # weighted score
    w = nb_class/np.sum(nb_class)
    w_accu = np.nansum(accu*w)
    w_prec = np.nansum(prec*w)
    w_reca = np.nansum(reca*w)
    w_f1 = np.nansum(f1*w)

    for r in (w_accu, w_prec, w_reca, w_f1):
        log.write('\t{:.4f}'.format(r))
    for s in stats_simp:
            log.write('\t-')
    log.write('\n')

    # category score
    if categories:
        for i in range(len(accu)):
            log.write(f'carac_note\t{i}\t{accu[i]}\t{prec[i]}\t{reca[i]}\t{f1[i]}\n')

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
        #model.summary()
        hist = train_model(model, db, output=f'best_note_{s}.hdf5', epochs=epochs)
        with open(f'best_note_{s}_training.log','w') as log:
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
        with open(f'best_note_{s}_predictions.log', "w") as log:
            print(f'-------- Predictions for model trained on {s} --------')
            log.write('song\tacc\tprec\trecall\tF1\tstroke\tnote\tvolume\tnb\ttempo\n')
            model = keras.models.load_model(f'best_note_{s}.hdf5')
            for s2 in sets:
                print(f' -> computing predictions on {s2}')
                if doTrain:
                    compute_scores(model, s2, 'train', log, categories)
                compute_scores(model, s2, 'test', log, categories)
                
if __name__ == '__main__':

    sets = ['set_1',
            'set_2',
            'set_3',
            'set_4'
            ]

  #  train_sets(sets, epochs=50)
    test_sets(sets, doTrain=True, categories=False)
   # test_sets(sets, doTrain=False, categories=True)
    
