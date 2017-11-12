"""
Plot characteristics of a dataset (from MIDI data)
"""
import midi
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

def analyze(filename, delta_min=0.02):
    notes = midi.MidiFile(filename).getNotes(False)

    note_low = 21    # lowest midi note on a keyboard
    note_high = 108  # highest midi note on a keyboard

    # Analyze note frequency and volume (force)

    stat_note = np.zeros(note_high-note_low)
    pos_note = np.arange(note_high-note_low)

    stat_note_simp = np.zeros(12)
    pos_note_simp = np.arange(12)
    lab_note_simp = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

    stat_vol = np.zeros(128)
    pos_vol = np.arange(128)

    for t,n,v in notes:
        stat_note[n-note_low] += 1
        stat_note_simp[n%12] += 1
        stat_vol[v] += 1

    # Analyze number of simultaneous notes played (single strokes / chords)

    stat_nb_notes = np.zeros(12)
    pos_nbnotes = np.arange(12)

    time = -1
    nb_notes = 1
    for t,n,v in notes:
        if (t-time)<delta_min:
            nb_notes+=1
        else:
            if time>=0:
                stat_nb_notes[nb_notes] += 1
            time = t
            nb_notes = 1

    # Analyze time (silence) between notes, i.e. playing speed or tempo (bpm)

    bpm_step = 25
    bpm_max = int(60/delta_min)
    pos_bpm = np.arange(0, bpm_max+1, bpm_step)
    stat_bpm = np.zeros(len(pos_bpm))

    time = -1
    for t,n,v in notes:
        delta_t = t - time
        if t==time or delta_t<delta_min:
            continue
        elif time >=0:
            bpm = min(round(60./delta_t/bpm_step), bpm_max)
            stat_bpm[bpm] += 1
        time = t

    return [ [pos_note, stat_note, 'Key'],
             [pos_note_simp, stat_note_simp, 'Note', lab_note_simp],
             [pos_vol, stat_vol, 'Volume'],
             [pos_nbnotes, stat_nb_notes, 'Nb of simultaneous notes', pos_nbnotes],
             [pos_bpm, stat_bpm, 'Tempo (bpm)'] ]

def analyze_multi(filenames):
    global_stats = None
    for fname in filenames:
        stats = analyze(fname)
        if global_stats==None:
            global_stats = stats
        else:
            for i in range(len(stats)):
                global_stats[i][1] += stats[i][1]
    return global_stats

def plot(stats, label=None, f=None, ax=None, nb_series=1, serie=0):
    if type(f)==type(None) or type(ax)==type(None):
        f, ax = plt.subplots(len(stats), figsize=(10,12))

    for i in range(len(stats)):
        s = stats[i]
        ax[i].set_xlim(0, s[0][-1])
        ax[i].set_xlabel(s[2])
        ax[i].set_ylabel('Frequency')
        if len(s)>3:
            ax[i].set_xticks(s[0])
            ax[i].set_xticklabels(s[3])
        width = 0.8*(s[0][1]-s[0][0])/nb_series
        ax[i].bar(s[0]+width*serie, s[1]/sum(s[1]), width, edgecolor='black', label=label)
        if label!=None:
            ax[i].legend(loc='upper right')

    f.subplots_adjust(hspace=0.5)

    return f,ax
    
if __name__=='__main__':

    set = 'set_3'

    train_stats = analyze_multi(glob(f'data/{set}/train/*.mid'))
    test_stats = analyze_multi(glob(f'data/{set}/test/*.mid'))

    for subset in ('train', 'test'):
        nb_notes = 0
        for f in glob(f'data/{set}/{subset}/*.mid'):
            nb_notes += len(midi.MidiFile(f).getNotes(False))
        print(f'{set} / {subset} : {nb_notes} notes')

    plt.close('all')
    f,ax = plot(train_stats, label='train', nb_series=2, serie=0)
    f,ax = plot(test_stats, label='test', f=f, ax=ax, nb_series=2, serie=1)
    ax[0].set_title('Set 3')
    # adjust x scales
    ax[1].set_xlim(-1,13)
    ax[-1].set_xlim(0,2200)
    plt.show()
