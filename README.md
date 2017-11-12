# musicscribe
Helper for music transcription, which  allows to convert an audio file to a MIDI file.

The current version supports piano pieces only. It is based on two CNNs, one for note onset detection, and another for key identification. These CNNs were trained on sinthesized music and therefore have limited performances on real recorded music. It may still give acceptable results on some pieces, so try it out !

Main files:
* transcribe.py -- the principal script that converts an audio file to MIDI
* onset.py -- train and test the onset detection CNN
* note.py -- train and test the key identification CNN

