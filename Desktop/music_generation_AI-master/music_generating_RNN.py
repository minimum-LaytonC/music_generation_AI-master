import numpy as np
from madmom.utils.midi import MIDIFile

import keras
from keras.models import Sequential
from keras.layers import LSTM, Activation

import random
import time, os


num_songs = 500
num_timesteps = 1000
note_vec_size = 5

num_epochs = 100
checkpoint_distance = 10

architecture = [50,50,50]

dataset = "1"
layers = ""
for layer in architecture:
	layers += str(layer) + "x"
activation = 'relu'

run_name = "ds:_" + dataset + "__arch:_" + layers + "__act:_" + activation

print(run_name)


################################################# Import music files

music_files = []

directory =  "datasets/" + dataset + "/"

for filename in os.listdir(directory):
	# generate a MIDIFile object from a midi file
	music_files.append(MIDIFile.from_file(directory + filename).notes())
	#note structure: (onset time, pitch, duration, velocity, channel)

music_files = np.array(music_files)

#print(music_files[0,:10])
import sys
#sys.exit(0)
#################################################


model = Sequential()
first = True
for layer in architecture:
	if(first):
		model.add(LSTM(50, batch_input_shape = (1, 1, note_vec_size), stateful=True, return_sequences=True))#, init=init),
		model.add(Activation(activation))
		first = False
	else:
		model.add(LSTM(50, stateful=True, return_sequences=True))
		model.add(Activation(activation))
model.add(LSTM(note_vec_size, stateful=True, return_sequences=True))


model.compile('adam', loss='mse')

#save network architecture
model.save("models/" + run_name)
#create checkpoint folder if it doesn't already exist
checkpoint_path = "model_checkpoints/" + run_name
if not os.path.exists(checkpoint_path):
	os.makedirs(checkpoint_path)

loss_list = []

# training loop
for epoch in range(num_epochs):					# epochs through all songs
	loss_this_epoch = 0
	print("\n\nEPOCH:\t"+str(epoch)+" / " + str(num_epochs) + "\n\n")
	for i in range(music_files.shape[0]):		# songs
		for note in range(music_files.shape[1]-1):	# timesteps
			print("\nnote:\t"+str(note)+" / "+str(music_files.shape[1]-1))

			loss_this_epoch += model.train_on_batch(np.reshape(music_files[i, note],(1,1,5)), np.reshape(music_files[i, note+1], (1,1,5)))

		model.reset_states()
	print(loss_this_epoch)
	loss_list.append(loss_this_epoch)

	if(epoch % checkpoint_distance == 0):

		model.save_weights(checkpoint_path + "/" + str(epoch) + "_epochs_weights")

		generated_song = []

		generated_song.append( model.predict_on_batch( np.reshape(music_files[0,0],(1,1,5)) ) )#np.random.randint(2, size=num_notes) ) )

		for i in range(len(music_files[0])):
			generated_song.append(model.predict_on_batch(np.reshape(generated_song[-1], (1,1,5))))

		#convert to np array and round pitch, velocity, and channel to integer values, then clip them to the appropriate range
		generated_song = np.array(generated_song)

		
		generated_song = generated_song[:,0,0,:] #manual reshaping
		
		generated_song[:, (1,3,4)] = np.round(generated_song[:,(1,3,4)])
		generated_song[:, (1,3,4)] = np.clip(generated_song[:,(1,3,4)], 0, 127)
		generated_song = np.nan_to_num(generated_song)

		print(generated_song[:10])

		new_midifile = MIDIFile.from_notes(generated_song)

		new_midifile.write("generated_song_attempts/" + run_name + str(epoch) + ".mid")

np.savetxt(("models/" + run_name + "__loss_trend"), loss_list)

#  T O  D O
#				determine which data to feed in
#				tweak neural net
#				get and save error rates

#				set up reinforcement learning