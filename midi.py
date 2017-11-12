"""
---------
Midi File
---------
version:    0.1
author:     Tommy Carozzani, 2017

Simple wrapper on top of mido
"""

import mido
import os

class MidiFile(mido.MidiFile):
    """Main class holding midi data"""
    def __init__(self, filename):
        if not os.path.isfile(filename):
            raise ValueError('cannot load {} (file doesnt exist)'.format(filename))
            
        mido.MidiFile.__init__(self, filename)
    
    def getNotes(self, timidity):
        """Get notes values, volume and timing
        
           Arguments:
           timidity -- set to True to remove the initial blank (silence), which is the default behavior of Timidity
           
           Returns:
           A list of notes in the form of (time, note, volume)
        """
        notes = []
        t = 0
        for msg in self:
            t += msg.time
            #print(msg,t)
            if msg.type=='note_on' and msg.velocity>0:
                if timidity and len(notes)==0:
                    t=0
                notes.append( (t, msg.note, msg.velocity) )
        return notes
