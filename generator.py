"""
--------------
Data generator
--------------
version:    0.3
author:     Tommy Carozzani, 2017

Provides functions to cycle through a database file and generate (yield) data.
"""

import numpy as np

def _get_i0(nb, frac, shift):
    """Convenience function to know where to start in the data given the shift argument"""
    delta = int(frac*nb)
    delta_skip = nb - delta
    
    if shift=='start':
        i0 = 0
    elif shift=='end':
        i0 = delta_skip
    else:
        try:
            ds = int(shift)
            assert ds<nb
            i0 = ds
        except:
            raise ValueError('unknown value for argument shift: {}'.format(shift))
    return i0,delta,delta_skip
        
def generator(data, nb, frac, steps=-1, consistent_loops=True, shift='start', dict_arrays=False):
    """Select samples from a list of data arrays

    Keyword arguments:
    files -- list of numpy arrays representing the data
    nb -- number of data samples to read at each step (-1 for all)
    frac -- fraction of data to select and send to output (yield)
    steps -- number of generation steps to perform (default -1 for infinite)
    consistent_loops -- True for cyclic data generation (default True). If False, the
        selected data may be different at each loop over files. If True, selected data will be
        always the same, but one step may generate less data than requested (if the total
        number of samples in files is not a multiple of nb).
    shift -- initial shift from the start of data. Allowed values:
        * 'start' -- no shift, equivalent to 0 (default)
        * 'end' -- full shift, i.e. the selected data will be taken at the end of the nb samples
        * integer -- explicit shift in number of samples
    dict_arrays -- Set to true if data arrays have non-int keys, like a dict or h5 files (default False)

    Return/yield:
    An array of size nb*frac, possibly less if consistent_loops=True

    Example:
    To generate 80% training data and 20% testing data, one may call:
    generator(data, 10000, 0.8)
    generator(data, 10000, 0.2, shift='end')

    See also:
    get_nb_steps() -- computes the number of required steps to go through all the data
    """

    debug=False

    nb_groups = len(data)
    nb_arrays = len(data[0])
    
    f = 0
    data_len = 0
    for fil in data[0]:
        if dict_arrays:
            data_len+=len(data[0][fil])
        else:
            data_len+=len(fil)
    if nb<0:
        nb = data_len

    if debug:
        print('data_len: ',data_len)

    i0, delta, delta_skip = _get_i0(nb, frac, shift)

    iterators = []
    data_in = []
    data_out = []
    for group in data:
        it = iter(group)
        if dict_arrays:
            din = group[next(it)]
        else:
            din = next(it)
        iterators.append(it)
        data_in.append(din)
        data_out.append(0)
        
    i_min = i0

    s = 0
    #while steps<0 or s<steps:
    while True:
        if debug:
            print('step {}'.format(s))
            
        for k in range(nb_groups):
            data_out[k] = np.zeros((delta,)+data_in[k].shape[1:])
            
        o_min = 0
        written = 0

        loop=True
        while loop and written<delta:
            while i_min>=len(data_in[0]):
                if debug:
                    print('change fin ',i_min, len(data_in[0]))
                    
                i_min -= len(data_in[0])
                f+=1
                
                if f>=nb_arrays:
                    f=0
                    for k in range(nb_groups):
                        iterators[k] = iter(data[k])

                for k in range(nb_groups):
                    if dict_arrays:
                        data_in[k] = data[k][next(iterators[k])]
                    else:
                        data_in[k] = next(iterators[k])
                #fin=files[f]
                    
                if s==0 and f==0:
                    print("Warning: duplicate data")
                if f==0 and consistent_loops:
                    i_min=i0
                    if written>0:
                        #print('break')
                        for k in range(nb_groups):
                            data_out[k].resize((written,)+data_in[k].shape[1:])
                        loop=False
                        break
            if not loop:
                break
            
            delta_loc = min(delta-written, len(data_in[0])-i_min)
            i_max = i_min+delta_loc
            o_max = o_min+delta_loc

            if debug:
                print(f,i_min,i_max,o_min,o_max)

            for k in range(nb_groups):
                data_out[k][o_min:o_max] = data_in[k][i_min:i_max]

            i_min = i_max
            o_min = o_max
            written += delta_loc
        if loop:
            i_min += delta_skip

        s += 1
        yield data_out

def get_nb_steps(files, nb, frac, consistent_loops=True, shift='start', dict_arrays=False):
    """Computes the number of required steps to go through all the data

    Keyword arguments:
    see definition of generator()

    Return:
    res -- number of steps required to go through all the data
    unique -- number of steps to get unique (distinct) data. May be one less than res if consistent_loops=False
    """
    l = 0
    for f in files:
        if dict_arrays:
            l+=len(files[f])
        else:
            l+=len(f)
        
    if nb<0:
        nb=l
    res = l//nb
    unique = res

    i0, delta, delta_skip = _get_i0(nb, frac, shift)
    #print('nb',i0,l,l%nb)
    rest = l%nb-i0
    if rest>0:
        res += 1
        if rest>=delta or consistent_loops:
            unique += 1
    return (res, unique)

