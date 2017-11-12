"""
---------------------
HDF5 database manager
---------------------
version:    0.2
author:     Tommy Carozzani, 2017

Provides classes to read and write from HDF5 files
"""

import h5py as h5
import numpy as np

class DatabaseReader(h5.File):
    """Base class for reading from an HDF% file"""
    def __init__(self, filename):
        h5.File.__init__(self, filename, 'r')
    
    def name(self):
        """Get the filename"""
        return self.filename

    def get_first(self, group):
        """Get the first item in a group.
           May raise an error if the group doesn't exist.
        
           Arguments:
           group -- name of the group and its hierarchy in the form of a list e.g.: [grandparent_name, parent_name, group_name]
        """
        g = self.get_subgroup(group)
        for d in g:
            return g[d]

    def get_subgroup(self, group):
        """Get a group base on its name and on its hierarchy name.
           May raise an error if the group doesn't exist.
        
           Arguments:
           group -- name of the group and its hierarchy in the form of a list e.g.: [grandparent_name, parent_name, group_name]
        """
        res = self
        for g in group:
            res = res[g]
        return res

    def get_total_points(self, group):
        """Get the total number of datapoints in a group.
           May raise an error if the group doesn't exist.
        
           Arguments:
           group -- name of the group and its hierarchy in the form of a list e.g.: [grandparent_name, parent_name, group_name]
        """
        dgroup = self.get_subgroup(group)
        res = 0
        for d in dgroup:
            res += dgroup[d].shape[0]
        return res

class DatabaseWriter(DatabaseReader):
    """Base class for writing to an HDF% file"""
    def __init__(self, filename, chunk_size=1000):
        """Arguments:
           filename -- name of hdf5 file
           chunk_size -- data chunk size on disk
        """
        h5.File.__init__(self, filename,'a')
        self.chunk_size = chunk_size

    def set_subgroup(self, groups):
        """Get a group base on its name and on its hierarchy name.
           If the group / hierarchy doesn't exist, it will be created.
           May raise an error if there is already another item (not group) with the same name.
        
           Arguments:
           group -- name of the group and its hierarchy in the form of a list e.g.: [grandparent_name, parent_name, group_name]
        """
        container = self
        for g in groups:
            sub = container.get(g)
            if sub==None:
                sub = container.create_group(g)
            elif type(sub)!=h5.Group:
                raise ValueError('cannot create group {} in file {} (name conflict)'.format(self.file.filename, g))
            container = sub
        return container
            
    def add(self, group, name, data):
        """Add some data to an existing dataset in a group.
           If the group / hierarchy doesn't exist, it will be created.
        
           Arguments:
           group -- name of the group and its hierarchy in the form of a list e.g.: [grandparent_name, parent_name, group_name]
           name -- dataset name
           data -- data to be written
        """
        container = self.set_subgroup(group)

        d = container.get(name)
        if d==None:
            d = container.create_dataset(name,
                                         data=data,
                                         maxshape=(None,)+data.shape[1:],
                                         chunks=(self.chunk_size,)+data.shape[1:],
                                         compression='lzf')
        elif type(d)!=h5.Dataset:
            raise ValueError('name conflit for data named {} in file {}'.format(name,self.file.filename))
        else:
            prev_len = d.shape[0]
            d.resize(prev_len+data.shape[0], 0)
            d[prev_len:] = data

    def create(self, group, name, data):
        """Add a dataset to a group.
           If the group / hierarchy doesn't exist, it will be created.
           If there was already data with the same name, it will be removed (depending on HDF5 version, this may or may not diminish file size)
        
           Arguments:
           group -- name of the group and its hierarchy in the form of a list e.g.: [grandparent_name, parent_name, group_name]
           name -- dataset name
           data -- data to be written
        """
        container = self.set_subgroup(group)

        d = container.get(name)
        if d!=None:
            del container[name]
        d = container.create_dataset(name,
                                     data=data,
                                     maxshape=(None,)+data.shape[1:],
                                     chunks=(self.chunk_size,)+data.shape[1:],
                                     compression='lzf')