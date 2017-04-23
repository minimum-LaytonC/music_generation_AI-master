import numpy as np
from madmom.utils.midi import MIDIFile

import torch
import torch.nn as nn 
from torch.autograd import Variable
import torch.optim as optim

import random
import time, os


num_songs = 500
num_timesteps = 1000
note_vec_size = 5

num_epochs = 10000
checkpoint_distance = 10

hidden_size = 100
num_layers = 2

architecture = [hidden_size for _ in range(num_layers)]

# architecture 

dataset = "1"
layers = ""
for layer in architecture:
	layers += str(layer) + "x"
activation = 'uhh'

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


# scale and center data
avg = np.mean(music_files, axis=1)
std = np.std(music_files, axis=1)

music_files = (music_files-avg)/std



#print(music_files[0,:10])
#import sys
#sys.exit(0)
#################################################

# PyTorch RNN model.

class RNN(nn.Module):
	def __init__(self, input_size, arch, output_size):
		super(RNN, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers

		self.lstm_arch = []
		for i in range(len(arch)):
			if(i==0):
				self.lstm_arch.append(nn.LSTM(input_size, arch[i], 1, batch_first=True))
				first_layer=False
			else:
				self.lstm_arch.append(nn.LSTM(arch[i-1], arch[i], 1, batch_first=True))

		self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
		self.fc = nn.Linear(arch[-1], output_size)
		
	def forward(self, x, hidden):
	    # Forward propagate RNN
	    hn = []
	    cn = []
	    out = x
	    h = hidden[0]
	    c = hidden[1]
	    for i in range(len(self.lstm_arch)):
	    	out, (hi, ci) = self.lstm_arch[i](out, h[i], c[i])
	    	hn.append(hi)
	    	cn.append(ci)

	    # Decode hidden state of last time step and layer
	    out = self.fc(out[:, -1, :])  
	    return out, (hn, cn)


model = RNN(5, architecture, 5).cuda()



criterion = nn.MSELoss().cuda()

optimizer = optim.Adam(model.parameters())



#	T O  D O 	save network architecture


#create checkpoint folder if it doesn't already exist
checkpoint_path = "model_checkpoints/" + run_name
if not os.path.exists(checkpoint_path):
	os.makedirs(checkpoint_path)

checkpoint_path = "generated_song_attempts/" + run_name
if not os.path.exists(checkpoint_path):
	os.makedirs(checkpoint_path)

loss_list = []



# training loop
for epoch in range(num_epochs):					# epochs through all songs

	# reset hidden states between each song -- clear memory for fresh slate.
	h = [Variable(torch.zeros(1, 1, layer).cuda(), requires_grad=True) for layer in architecture]
	c = [Variable(torch.zeros(1, 1, layer).cuda(), requires_grad=True) for layer in architecture]

	loss_this_epoch = 0
	print("\n\nEPOCH:\t"+str(epoch)+" / " + str(num_epochs) + "\n\n")
	for i in range(music_files.shape[0]):		# songs
		for j in range(music_files.shape[1]-1): # notes
			optimizer.zero_grad()

			input = Variable(torch.from_numpy(music_files[i, j]), requires_grad=False).view(1,1,5).float()
			input = input.cuda()

			target = Variable(torch.from_numpy(music_files[i, j+1]), requires_grad=False).view(1,1,5).float()
			target = target.cuda()
			hidden = [h,c]
			output, (h,c) = model(input, hidden)
			h = [Variable(hi.data) for hi in h]
			c = [Variable(ci.data) for ci in c]
			# print("\n\n\nTHIS SHIT RIGHT HERE")
			# print(output)
			# print(type(output.data.numpy()))
			# print("\n\n\n")

			loss = criterion(output, target)

			if(j % 50 == 0 and j != 0):
				print("note " + str(j) + " / " + str(music_files.shape[1]-1) + "\navg loss:\t" + str(loss_this_epoch/j))


			loss.backward()

			optimizer.step()

			loss_this_epoch += loss.data

		
			
	print(loss_this_epoch)
	loss_list.append(loss_this_epoch)

	if(epoch % checkpoint_distance == 0):

		h = Variable(torch.zeros(num_layers, 1, hidden_size).cuda(), requires_grad=False, volatile=True)
		c = Variable(torch.zeros(num_layers, 1, hidden_size).cuda(), requires_grad=False, volatile=True)

		print("checkpoint at epoch " + str(epoch))

		torch.save(model.state_dict(), str("model_checkpoints/" + run_name + "/" + str(epoch) + "_epochs"))

		print("saved")

		#	T O  D O 	save weights to: (checkpoint_path + "/" + str(epoch) + "_epochs_weights")

		#	T O  D O 	generate song 
		generated_song = []

		#for now, starts with first note of the song
		note = Variable(torch.from_numpy(music_files[0, 0]).cuda(), requires_grad=False, volatile=True).view(1,1,5).float()
		note, (h,c) = model(note, h,c)
		note = note.view(1,1,5)
		

		generated_song.append(note.cpu().data.numpy())#np.random.randint(2, size=num_notes) ) )

		for i in range(999):
			note, (h,c) = model(note, h, c)

			h = Variable(h.data)
			c = Variable(c.data)
			

			note = note.view(1,1,5)
			generated_song.append(note.cpu().data.numpy())

		#convert to np array and round pitch, velocity, and channel to integer values, then clip them to the appropriate range
		generated_song = np.array(generated_song)

		
		generated_song = generated_song[:,0,0,:] #manual reshaping

		generated_song = (generated_song*std)+avg
		
		#	restructuring generated song for the proper midi format
		generated_song[:, (1,3,4)] = np.round(generated_song[:,(1,3,4)])
		generated_song[:, (1,3,4)] = np.clip(generated_song[:,(1,3,4)], 0, 127)
		generated_song = np.nan_to_num(generated_song)

		print(generated_song[:10])

		new_midifile = MIDIFile.from_notes(generated_song)

		new_midifile.write("generated_song_attempts/" + run_name + "/" + str(epoch) + ".mid")

np.savetxt(("models/" + run_name + "__loss_trend"), loss_list)



#  	T O  D O
#				determine which data to feed in
#				tweak neural net
#				get and save error rates

#				set up reinforcement learning