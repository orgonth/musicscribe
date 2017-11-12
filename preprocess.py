# General libs
from glob import glob

# Audio libs
from AudioFile import AudioFile
from midi import MidiFile
import cqt
import melbank

# Math libs
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Custom code
import database

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

def ComputeSpectrograms(audio, step, windows=[23,46,93]):
    """
    Compute FFT spectrograms with different window sizes
    :param step: step in seconds
    :param windows: window sizes (array) in milliseconds
    :return: list of spectrograms
    """
    hop = int(step*audio.Fs)
    spectrograms = []
    for w in windows:
        wsize = int(w/1000.*audio.Fs)
        freq, time, specgram = signal.spectrogram(audio.data, audio.Fs,
                            #window='hann',
                            nperseg=wsize,
                            noverlap=wsize-hop)
        spectrograms.append( (freq,time,specgram) )
    return spectrograms

def ComputeMelFilter(nb_bands, f_low, f_high, shift, Fs, frequencies, times, spectrogram, step):
    """
    Apply a Mel-filter to a spectrogram
    :param nb_bands: number of Mel bands
    :param f_low: lowest frequency cutoff
    :param f_high: highest frequency cutoff
    :param shift: time shift (in terms of number of steps)
    :param Fs: audio frame rate
    :param frequencies: list of frequencies of the spectrogram
    :param times: list of timings of the spectrogram
    :param spectrogram: spectrogram
    :param step: step (in seconds)
    :return: bands frequencies, timings, and filtered spectrogram
    """
    melmat, (melfreq, fftfreq) = melbank.compute_melmat(nb_bands, f_low, f_high,
                                                        num_fft_bands=len(frequencies),
                                                        sample_rate=Fs)
    # apply filter
    m = np.zeros( (len(times),nb_bands) )
    i=0
    for fft_spectra in spectrogram.transpose():
        mel_spectra = melmat.dot(fft_spectra)
        m[i] = mel_spectra
        i+=1

    # take the log
    m[m==0]=np.nan
    m = np.log(m.transpose())

    # centre around mean and normalization of variance
    m = (m - np.nanmean(m,1,keepdims=True))/np.nanstd(m,1,keepdims=True)
    np.nan_to_num(m, copy=False)
    ashift = ( max(0, -shift) , max(0, shift) )
    m = np.pad(m, ((0,0),ashift), 'constant')
    return (melfreq, times+step*shift, m)

def ComputeMelLayers(spectrograms, step, Fs, latency):
    """
    Apply a Mel-filter to several spectrograms
    :param spectrograms: list of spectrograms
    :param step: step size in seconds
    :param Fs: audio frame rate
    :param latency: latency in terms of number of steps
    :return: list of (bands frequencies, timings, filtered spectrogram)
    """
    melgrams = []

    if step==0.01:
        melshift = np.array([-8, -10, -14])
    elif step==0.02:
        melshift = np.array([-8, -9, -11])
    else:
        raise ValueError('unknown step value {}'.format(step))

    melshift += latency

    nb_bands = 80
    i = 0
    for (f, t, spec) in spectrograms:
        melfreq, times, melgram = ComputeMelFilter(nb_bands,
                                                   f_low=27.5, f_high=8000,
                                                   shift=melshift[i],
                                                   Fs=Fs,
                                                   frequencies=f,
                                                   times=t,
                                                   spectrogram=spec,
                                                   step=step)
        melgrams.append([melfreq, times, melgram])
        i += 1

    return np.array(melgrams)

def ComputeCqt(audio, f_low, f_high, window, latency, r=3):
    """
    Compute Constant-Q Transform
    :param f_low: low frequency cutoff
    :param f_high: high frequency cutoff
    :param window: analysis window in seconds
    :param latency: latency in terms of number of steps
    :param r: frequency resolution (=levels per note)
    :return: frequencies, timings, and transform
    """
    if window==0.01:
        shift = -20
    elif window==0.02:
        shift = -14
    else:
        raise ValueError(f'unkown step value {window}')
    shift += latency

    cqobj = cqt.CqGram(audio, f_low, f_high, r)
    cqobj.compute_kernels(dynamic=False)
    cqgram = cqobj.compute_cqt_full(window, log=True)
    ashift = ( max(0, -shift) , max(0, shift) )
    cqgram = np.pad(cqgram, ((0,0),ashift), 'constant')
    time = np.arange(0, audio.duration+abs(window*shift), window)[:cqgram.shape[1]]
    time += window*shift/2

    FreqAxis = cqobj.get_freq_axis()
    return (FreqAxis, time, cqgram)

def BuildTensor(grams, window, normalize=False):
    """
    Build a TensorFlow-ready array from spectral transforms
    :param grams: list of spectral transforms
    :param window: CNN window width
    :param normalize: set to True to normalize transforms
    :return: a tensor

        /!\ A lot of potential errors are NOT checked. For e.g. all grams are supposed to:
            - start at the same time (but not necesseraly end at the same time)
            - have the same frequency axis
    """
    if normalize:
        for i in range(len(grams)):
            gmin = np.min(grams[i])
            gmax = np.max(grams[i])
            grams[i] = (grams[i]-gmin)/(gmax-gmin)

    # First get the data length restriction by time
    min_nb_times = np.inf
    for g in grams:
        min_nb_times = min(min_nb_times, g.shape[1])
    
    max_len = min_nb_times+1 - window
    if max_len<=0:
        raise ValueError('error building tensor : no data left. shape={} / window={})'.format(grams.shape[1], window))
    tensor = np.zeros( (max_len, window, grams[0].shape[0], len(grams)) )

    for i in range(max_len):
        channel = 0
        for g in grams:
            tensor[i,:,:,channel] = g[:,i:i+window].transpose()  # extract a cnn_window x frequencies matrix
            channel += 1

    return tensor

def preprocess(filename, timidity, latency, truncate, pad=1, get_raw=False):
    """
    Preprocess an audio file ands its MIDI counterpart. Computes transforms and labels.
    :param filename: audio filename
    :param timidity: set to True if the files was rendered with timidity
    :param latency: in seconds
    :param truncate: in seconds (0 for no truncation)
    :param pad: in seconds, will be added at the start and end before spectral transforms
    :param get_raw: set to True to return raw computed spectrograms (e.g. for visualization)
    :return:
    """
    filename_midi = filename.rsplit('.')[0] + '.mid'
    
    dname = filename.replace('/','_').replace('\\','_')

    # Load files
    ipad = int(pad*44100)
    audio_pad = (ipad, ipad) # add one blank second at the beginning and at the end
    if truncate>0:
        audio = AudioFile(filename, truncate=int(truncate*44100), pad=audio_pad)
    else:
        audio = AudioFile(filename, pad=audio_pad)
    mid = MidiFile(filename_midi)

    step = 0.02 # seconds
    latency = int(round(latency/step,0))
   
    # Compute spectrograms
    spectrograms = ComputeSpectrograms(audio, step=step)

    # Compute filtered spectrograms
    melgrams = ComputeMelLayers(spectrograms, step, audio.Fs, latency)

    # Build the input tensor
    cnn_window = 15
    tensor_mel = BuildTensor( melgrams[:,2], cnn_window )

    # Compute CQT
    FreqAxisLog, time, cqgram = ComputeCqt(audio, 200., 4000., step, latency, r=3)
    tensor_cqt = BuildTensor( [cqgram,], cnn_window )

    # Global data length
    max_len = min(tensor_mel.shape[0], tensor_cqt.shape[0])
    
    # Compute output labels
    notes = mid.getNotes(timidity)
    notes_onset = np.array(notes)[:,0]                # get only the note timing
    notes_value = np.array(notes, dtype=np.int)[:,1]  # get only the key value

    onset_labels = np.zeros(max_len)
    onset_caracs = np.zeros( (max_len, 5) )
    onset_caracs[:, 2] = np.arange(max_len)

    note_low = 21    # lowest midi note on a keyboard
    note_high = 108  # highest midi note on a keyboard

    notes_labels = np.zeros( (max_len, note_high-note_low+1) )
    notes_caracs = np.zeros( (max_len, note_high-note_low+1) )

    for i in range(len(notes_onset)):
        t_win = int(np.floor( (notes_onset[i]+audio_pad[0]/audio.Fs)/step ))
        if t_win>=len(onset_labels):
            break
        if t_win>=0:
            onset_labels[t_win] = 1
            onset_caracs[t_win][0] += 1     # nb_notes
            onset_caracs[t_win][1] = max(onset_caracs[t_win][1], notes[i][2])   # max volume
            if t_win+1<len(onset_labels):
                onset_caracs[t_win+1:,2] -= onset_caracs[t_win+1][2]    # nb of blank windows since the last onset

            n = notes_value[i]-note_low
            notes_labels[t_win][n] = 1
            notes_caracs[t_win][n] = notes[i][2]    # volume

    counter = 0
    for i in range(len(onset_labels)-1,-1,-1):
        onset_caracs[i][3] = counter
        if onset_labels[i] == 1:
            counter = 0
        else:
            counter += 1
    onset_caracs[:,4] = np.minimum( onset_caracs[:,2], onset_caracs[:,3] )

    # Extract useful CQT
    select = [i for i in range(max_len) if onset_labels[i]>0]
    tensor_cqt_select = np.take(tensor_cqt, select, axis=0)
    notes_labels_select = np.take(notes_labels, select, axis=0)
    notes_caracs_select = np.take(notes_caracs, select, axis=0)

    if not get_raw:
        return ( tensor_mel[:max_len,...],
                 tensor_cqt_select,
                 onset_labels,
                 onset_caracs,
                 notes_labels_select,
                 notes_caracs_select,
                 dname )
    else:
        return ( melgrams, tensor_mel, onset_labels,
                 cqgram, tensor_cqt, time, FreqAxisLog,
                 max_len, step)

def write_to_db(tensor_mel, tensor_cqt, onset_labels, onset_caracs, notes_labels, notes_caracs, dname, dbname, groupname):
    """
    Writes computed data and labels to a database
    Parameters are the output of preprocess function, except:
    :param dbname: filename of the database
    :param groupname: groupname (e.g. 'train', 'test'...)
    :return: None
    """
    db = database.DatabaseWriter(dbname)
    db.create([groupname, 'mel'], dname, tensor_mel)
    db.create([groupname, 'cqt'], dname, tensor_cqt)

    db.create([groupname, 'onset_labels'], dname, onset_labels)
    db.create([groupname, 'onset_caracs'], dname, onset_caracs)
    db.create([groupname, 'note_labels'], dname, notes_labels)
    db.create([groupname, 'notes_caracs'], dname, notes_caracs)
    db.close()

def preprocess_datasets(nb_threads=1, truncate=0):
    """
    Preprocess all four default datasets found in data folder
    :param nb_threads: number of threads to use
    :param truncate: in seconds, set to 0 to keep it all (default)
    :return: None
    """
    import multi
    tm = multi.ThreadManager()
    comp_group = tm.add_group()
    write_group = tm.add_group(1,0)
    sets = [('set_1', True, 0),
            ('set_2', False, 0.06),
            ('set_3', True, 0),
            ('set_4', False, 0.06)
            ]
    for s,timidity,latency in sets:
        print('preprocessing {}...'.format(s))
        dbname = 'data_{}.hdf5'.format(s)
        for f in glob('data/{}/train/*.mp3'.format(s)):
            comp_group.add_job(preprocess, (f, timidity, latency, truncate),
                               callback=lambda a,b,c,d,e,f,g,h=dbname,i='train': write_to_db(a,b,c,d,e,f,g,h,i),
                               callback_group=write_group)
        for f in glob('data/{}/test/*.mp3'.format(s)):
            comp_group.add_job(preprocess, (f, timidity, latency, truncate),
                               callback=lambda a,b,c,d,e,f,g,h=dbname,i='test': write_to_db(a,b,c,d,e,f,g,h,i),
                               callback_group=write_group)
    tm.run(nb_threads)

def visual_check(filename, timidity, latency, truncate):
    """
    Compute and plot spectral transforms for an audio file
    :param filename: audio file
    :param timidity: set to True if the files was rendered with timidity
    :param latency: in seconds
    :param truncate: in seconds
    :return: None
    """
    (melgrams, tensor_mel, onset_labels,
     cqgram, tensor_cqt, time, FreqAxisLog,
     max_len, step) = preprocess(filename, '', 'train', timidity, latency, truncate, pad=0, get_raw=True)

    # print some statistics

    print('Mel min/max: {} / {}'.format(np.min(tensor_mel), np.max(tensor_mel)))
    print('CQT min/max: {} / {}'.format(np.min(tensor_cqt), np.max(tensor_cqt)))

    # plot melgrams + labels

    f, axarr = plt.subplots(len(melgrams), sharex=True)
    t_labels = np.arange(0, max_len * step, step)
    i = 0
    for (melfreq, t, m) in melgrams:
        axarr[i].plot(t_labels, onset_labels * max(melfreq) / 2.)
        axarr[i].pcolormesh(t - latency * step, melfreq, m[:, :len(t)])
        i += 1

    # plot the first tensor entry with a positive label

    idx = np.nonzero(onset_labels)[0][1]

    feat = tensor_mel[idx]

    f, axarr = plt.subplots(1, len(melgrams), sharey=True)
    for i in range(len(melgrams)):
        m = feat[:, :, i].transpose()
        axarr[i].pcolormesh(m)

    # plot cqt + labels

    f, axarr = plt.subplots(1, figsize=(10, 2.5))
    axarr.pcolormesh(time, FreqAxisLog, cqgram, vmin=-20.0, vmax=-5.)
    axarr.set_xlabel('Time [s]')
    axarr.set_xlim(0,10)
    axarr.set_ylabel('Frequency [Hz]')
    axarr.set_yscale('log')
    axarr.set_title('CQT')
    f.subplots_adjust(bottom=0.2)


    # plot cqt tensor entry

    f, axarr = plt.subplots(1, figsize=(2,5))
    feat = tensor_cqt[idx]
    m = feat[:, :, 0].transpose()
    axarr.pcolormesh(m, vmin=-20., vmax=-5.)
    f.subplots_adjust(left=0.2, bottom=0.2)

    plt.show()

if __name__=='__main__':
#    visual_check('data/set_1/test/beethoven_opus22_2.mp3', True, 0, 10)
    preprocess_datasets(truncate=0)
