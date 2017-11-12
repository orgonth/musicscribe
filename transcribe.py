"""
Transcribe an audio file to MIDI
"""

# custom libs
from AudioFile import AudioFile
import preprocess

# general libs
import os
import subprocess

# audio libs
import mido

# math libs
import numpy as np
import keras

_VERBOSE = 1
_TIMIDITY_DIR = 'D:/Logiciels/timidity'

def _vprint(msg, verbosity=1):
    if _VERBOSE>=verbosity:
        print(msg)

def find_exec(exe_name, additional_dirs=[]):
    """Find if an executable is present within the PATH
       :param exe_name: name of the executable to find, or a list of different names it can take
       :param additional_dirs: list of additional directories to look into (with higher priority than PATH)
       :return: path of the executable if found, None otherwise
    """
    if type(exe_name)==str:
        exe_name = (exe_name,)

    exe_dirs = additional_dirs + [os.getcwd()] + os.get_exec_path()
    for d in exe_dirs:
        for x in exe_name:
            if os.path.exists(d+'/'+x):
                return os.path.abspath(d+'/'+x)
    return None

def transcribe_to_midi(filename, onset_model, note_model, output):
    """
    Transcribes and audio file to midi and renders it to wav if timidity is found in the path
    :param filename: path of audio file to transcribe
    :param onset_model: onset detection model to use (hdf5 filename)
    :param note_model: key identification model to use (hdf5 filename)
    :param output: output filename (without extension)
    :return: None
    """
    step = 0.02

    _vprint(f'load audio {filename}...')
    audio = AudioFile(filename, pad=(44100,44100))

    _vprint('computing spectrograms...')
    spectrograms = preprocess.ComputeSpectrograms(audio, step=step)

    _vprint('computing (mel) filtered spectrograms...')
    melgrams = preprocess.ComputeMelLayers(spectrograms, step, audio.Fs, latency=0)
    cnn_window = 15
    tensor_mel = preprocess.BuildTensor(melgrams[:, 2], cnn_window)

    _vprint('onset detection...')
    model = keras.models.load_model(onset_model)
    preds_onset = 1. * (model.predict(tensor_mel) >= 0.2)

    nb_notes = np.sum(preds_onset)

    _vprint(f'{nb_notes} onsets detected')

    _vprint('computing CQT...')
    # TODO: compute only useful ones
    FreqAxisLog, time, cqgram = preprocess.ComputeCqt(audio, 200., 4000., step, latency=0, r=3)
    tensor_cqt = preprocess.BuildTensor([cqgram, ], cnn_window)

    max_len = min(tensor_mel.shape[0], tensor_cqt.shape[0])
    select = [i for i in range(max_len) if preds_onset[i] > 0]
    tensor_cqt_select = np.take(tensor_cqt, select, axis=0)

    _vprint('key identification...')
    model = keras.models.load_model(note_model)
    preds_notes = 1. * (model.predict(tensor_cqt_select) >= 0.5)
    _vprint(f'{np.sum(preds_notes)} keys identified')

    _vprint('midi writing...')
    mid = mido.MidiFile(ticks_per_beat=500)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.Message('program_change', program=1, time=0))

    i = 0
    t = 0  # time in seconds
    t_last = 0
    notes_on = np.zeros(88)
    for w in preds_onset:
        if w[0]:
            if np.sum(preds_notes[i])>0:
                delta_t = int((t - t_last) / 0.001) # delta_time in midi ticks

                for n in notes_on.nonzero()[0]:
                    midi_note = int(n + 21)
                    track.append(mido.Message('note_off', note=midi_note, velocity=0, time=delta_t))
                    notes_on[n] = 0
                    delta_t = 0

                for n in preds_notes[i].nonzero()[0]:
                    midi_note = int(n + 21)
                    track.append(mido.Message('note_on', note=midi_note, velocity=64, time=delta_t))
                    notes_on[n] = 1
                    delta_t = 0
                t_last = t

            i += 1

        t += step

    mid.save(f'{output}.mid')

    timidity = find_exec(['timidity', 'timidity.exe'], additional_dirs=[_TIMIDITY_DIR])

    if timidity is not None:
        _vprint('timidity found, rendering to audio: '+timidity)
        subprocess.run([timidity, f'{output}.mid', '-Ow', '-o', f'{output}.wav', '--output-mono'])

if __name__=='__main__':
    piece = 'chpn_op25_e11'
    filename = f'data/set_2/test/{piece}.mp3'

    # piece = 'chpn_op25_e11_yundi_li'
    # filename = piece+'.mp3'

    onset_model = 'best_models/best_onset_set_4.hdf5'
    note_model = 'best_models/best_note_set_4.hdf5'

    transcribe_to_midi(filename, onset_model, note_model, output=f'transcribed_{piece}')
