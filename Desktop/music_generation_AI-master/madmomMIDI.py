from madmom.utils.midi import MIDIFile
import numpy as np

file = "datasets/1/bach_846 Format 0.mid"
# generate a MIDIFile object from a midi file.
midifile = MIDIFile.from_file(file)

print(type(midifile))
# get all the notes from the MIDIFile. Structure: (onset time, pitch, duration, velocity, channel)
# 	velocity ~ volume; channel ~ instrument.
notes = midifile.notes(unit='s')

#snotes[:, (0,2)] *= midifile.SECONDS_PER_TICK

#shift all the notes up an octave
notes[:,1] -= 12

#clip to midi pitch range: [0, 127]
notes[:,1] = np.clip(notes[:,1], 0, 127)

print(type(notes))
print(notes[:10])

new_midifile = MIDIFile.from_notes(notes)

new_midifile.write("upOctave.mid")
