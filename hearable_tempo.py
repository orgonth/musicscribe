import mido
from mido import MidiFile, MidiTrack, Message

mid = MidiFile(ticks_per_beat=500)
track = MidiTrack()
mid.tracks.append(track)

track.append(Message('program_change', program=1, time=0))

print(mid.ticks_per_beat)

# delta_t in ticks
# 500 ticks per beat
# 120 beats per minute
# -> 1 beat = 0.5s
# -> 1 tick = .001s

# TEST 1 - chord

for i in range(0,30,5):
    delta_t = i
    
    # add 1s silence
    track.append(Message('note_on', note=60, velocity=0, time=500))
    track.append(Message('note_off', note=60, velocity=0, time=500))

    notes = (60,64,67,72)
    for k in notes:
        track.append(Message('note_on', note=k, velocity=64, time=delta_t))
    for k in notes:
        track.append(Message('note_off', note=k, velocity=127, time=delta_t+100))

# TEST 2 - note repetition

track.append(Message('note_on', note=60, velocity=0, time=1000))
track.append(Message('note_off', note=60, velocity=0, time=1000))
    
for i in range(0,100,10):
    delta_t = i
    
    # add 1s silence
    track.append(Message('note_on', note=60, velocity=0, time=500))
    track.append(Message('note_off', note=60, velocity=0, time=500))

    notes = (60,60,60,60)
    for k in notes:
        track.append(Message('note_on', note=k, velocity=64, time=delta_t))
        track.append(Message('note_off', note=k, velocity=127, time=max(1,delta_t)))

# TEST 3 - trill

track.append(Message('note_on', note=60, velocity=0, time=1000))
track.append(Message('note_off', note=60, velocity=0, time=1000))
    
for i in range(0,100,10):
    delta_t = i
    
    # add 1s silence
    track.append(Message('note_on', note=60, velocity=0, time=500))
    track.append(Message('note_off', note=60, velocity=0, time=500))

    notes = (60,61,60,61,60,61,60,61,60,61,60)
    for k in notes:
        track.append(Message('note_on', note=k, velocity=64, time=delta_t))
        track.append(Message('note_off', note=k, velocity=127, time=max(1,delta_t)))
                     
mid.save('hearable_tempo.mid')
