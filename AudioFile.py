"""
----------
Audio File
----------
version:    0.1
author:     Tommy Carozzani, 2017

Simple wrapper on top of pydub
Requires pydub and ffmpeg
"""

import os
import numpy as np
from pydub import AudioSegment

class AudioFile:
    """Class holding raw audio data"""
    def __init__(self, filename, truncate=None, pad=(0,0)):
        """Arguments:
           filename -- name of audio file on disk
           truncate -- number of samples to keep (default = None = keep all)
           pad -- number of blank samples to add at the beginning and at the end (default = (0,0))
        """
        self.filename = filename
        self.load(truncate, pad)
        self.normalize()

    def load(self, truncate, pad):
        """Loads data
           truncate -- number of samples to keep (default = None = keep all)
           pad -- number of blank samples to add at the beginning and at the end (default = (0,0))
        """
        if not os.path.isfile(self.filename):
            raise ValueError('cannot load {} (file doesnt exist)'.format(self.filename))
        
        self.audio = AudioSegment.from_file(self.filename)
        self.audio = self.audio.set_channels(1)
        self.Fs = self.audio.frame_rate
        self.data = np.array(self.audio.get_array_of_samples())[:truncate]
        self.data = np.pad(self.data, pad, 'constant')
        self.len = len(self.data)
        self.duration = self.len/self.Fs

    def normalize(self):
        """Normalize the signal so that max = 1
        """
        minmax = max(np.max(self.data),-np.min(self.data))
        if minmax:
            self.data = self.data/minmax
