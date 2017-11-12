"""
--------------------
Constant-Q Transform
--------------------
version:    0.1
author:     Tommy Carozzani, 2017

Computes the constant-Q spectral transform (CQT) of a signal.
"""

from AudioFile import AudioFile
import numpy as np
import time

class CqGram:
    """Main class to compute CQT"""
    def __init__(self, audio, f_low, f_high, r):
        """Arguments:
           audio -- an AudioFile type
           f_low -- lowest frequency to analyze
           f_high -- highest frequency to analyze
           r -- number of frequency per note (standard=3)
        """
        self.r=r
        self.audio=audio
        self.Fs=audio.Fs
        self.f_low=f_low
        self.f_high=f_high
        self.f_range = None

        self.nb_filters = int(np.log2(f_high/f_low)*12*r)

    def _compute_f_range_constant(self):
        """Get the extracted frequencies as defined in the standart CQT method"""
        self.f_range=np.zeros(self.nb_filters+1)
        for k in range(self.nb_filters):
            self.f_range[k] = self.f_low*2**(k/(12*self.r))  # lower frequency of the filter
        self.f_range[-1]=self.f_high
        return self.f_range

    def _compute_f_range_dynamic(self):
        """Get the custom extracted frequencies (higher quality in the audible range)"""
        self.f_range=np.zeros(self.nb_filters+1)

        self.f_range[0] = self.f_low
        for k in range(1,self.nb_filters):
            s = np.exp(0.76*k/self.nb_filters)-0.5
            #s = 0.5*np.exp(1.9*k/self.nb_filters)-0.5
            self.f_range[k] = self.f_range[k-1]*2**(s/(12*self.r))

        self.f_range[-1]=self.f_high
        return self.f_range

    def get_freq_axis(self):
        """Get the list of frequencies
           The kernels must have been computed before calling this function (with compute_kernels or compute_cqt_...)
        """
        return self.f_range[:-1]
        
    def compute_kernels(self, dynamic=False):
        """Compute the kernels (i.e. filters)
        
           Arguments:
           dynamic -- set to True to use custom varying quality (default=False)
        """
        if dynamic:
            self._compute_f_range_dynamic()
        else:
            self._compute_f_range_constant()

        for k in range(self.nb_filters):
            delta_f = self.f_range[k+1]-self.f_range[k]
            nb_points = int(self.Fs/delta_f)

            if k==0:
                self.frame_len = nb_points
                self.kernel = np.zeros( (self.nb_filters, nb_points), dtype=np.complex)
                self.limits = np.zeros( (self.nb_filters, 2), dtype=np.int)
                
            func = np.hanning(nb_points) * np.exp(-2j*np.pi*self.f_range[k]*np.arange(nb_points)/self.Fs) / nb_points
            
            d = int((self.frame_len - nb_points)/2.)
            self.kernel[k] = np.pad(func,
                                    ( d,
                                      self.frame_len-(nb_points+d)),
                                    'constant')
            self.limits[k] = [d, d+nb_points-1]

    def compute_cqt_full(self, step, log=True):
        """Compute the CQTs for the whole signal
        
           Arguments:
           step -- time in seconds between CQTs
           log -- set to True (default) to return the logarithm of the CQT
           
           Returns:
           An array of CQTs
        """
        #start = time.time()
        
        step = int(step*self.Fs)
        dimy = int(np.ceil((self.audio.len-self.frame_len+1)/step))
        gram = np.zeros( (self.nb_filters, dimy) )

        i=0
        idx = 0
        
        while i+self.frame_len-1<self.audio.len:
            for k in range(self.nb_filters):
                gram[k, idx] = np.abs(np.sum( self.audio.data[i+self.limits[k,0]:
                                                              i+self.limits[k,1]+1] \
                                              * \
                                              self.kernel[k,
                                                          self.limits[k,0]:
                                                          self.limits[k,1]+1] \
                                              ))**2
            i += step
            idx += 1
            
        if log:
            gram[gram==0]=np.nan
            gram = np.log(gram)
            np.nan_to_num(gram, copy=False)

        #dt = time.time()-start
        #print('done in {:.4} s'.format(dt))
    
        return gram

    def compute_cqt_select(self, times, log=True):
        """Compute the CQTs for selected times in the signal
        
           Arguments:
           times -- list of times where to compute CQT
           log -- set to True (default) to return the logarithm of the CQT
           
           Returns:
           An array of CQTs
        """
        gram = np.zeros( (self.nb_filters, len(times)) )

        broke=False
        idx=0

        for t in times:
            i = int(t*self.Fs)
            if i+self.frame_len>self.audio.len:
                broke=True
                break

            idx += 1
            
            for k in range(self.nb_filters):
                gram[k, idx] = np.abs(np.sum( self.audio.data[i+self.limits[k,0]:
                                                              i+self.limits[k,1]+1] \
                                              * \
                                              self.kernel[k,
                                                          self.limits[k,0]:
                                                          self.limits[k,1]+1] \
                                              ))**2

        if broke:
            gram.resize( (self.nb_filters, idx) )
            
        if log:
            gram[gram==0]=np.nan
            gram = np.log(gram)
            np.nan_to_num(gram, copy=False)

        return gram
