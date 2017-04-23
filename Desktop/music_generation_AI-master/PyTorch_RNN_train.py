import numpy as np
from madmom.utils.midi import MIDIFile

import torch
import torch.nn as nn 
from torch.autograd import Variable
import torch.optim as optim

import random
import time, os, math

################################################# Parameters
num_epochs = 10000
checkpoint_distance = 5

hidden_size = 200
num_layers = 3

architecture = [hidden_size for _ in range(num_layers)]
num_songs = 3000
dataset = "classical"
dataset_name = dataset
if(num_songs>1):
	 dataset_name += str(num_songs)
layers = ""
for layer in architecture:
	layers += str(layer) + "x"
truncate_length = 10

run_name = "ds:_" + dataset_name + "__arch:_" + layers + "__trunc:_" + str(truncate_length)

print(run_name)
##################################################



################################################## Import music files
music_files = []

directory =  "datasets/" + dataset + "/"
count = 1
for filename in os.listdir(directory):
	if count > num_songs:
		break;
		
	print("\nfile" + str(count)+"\n"+filename+"\n")
	count+=1
	
	print(MIDIFile.from_file(directory + filename).notes().shape)
	# generate a MIDIFile object from a midi file
	music_files.append(MIDIFile.from_file(directory + filename).notes())
	#note structure: (onset time, pitch, duration, velocity, channel)


music_files = np.array(music_files)
print(music_files.shape)
#################################################



################################################# scale and center data
#avg = [np.mean(music_files[i], axis=0) for i in range(len(music_files))]
#std = [np.std(music_files[i], axis=0) for i in range(len(music_files))]

#music_files = [(music_files[i]-avg[i])/(std[i]+0.0000001) for i in range(len(music_files))]

print("len(music_files):")
print(len(music_files))
print("music_files[0].shape")
print(music_files[0].shape)

#AVG = np.mean([avg[i]*music_files[i].shape[0] for i in range(len(music_files))])
#STD = math.sqrt(np.sum(np.square(std)))
#################################################



################################################# PyTorch RNN model.
class RNN(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers, output_size):
		super(RNN, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
		self.fc = nn.Linear(hidden_size, output_size)
		
		self.a = 0

	def forward(self, x, h, c):
	    # Forward propagate RNN
	    self.a += 1
	    #print(self.a)
	    out, (hn, cn) = self.lstm(x, (h, c))  
	    
	    

	    # Decode hidden state of last time step
	    out = self.fc(out[:, -1, :])  
	    return out, (hn,cn)

	# def reset_states(self):
	# 	self.h = Variable(torch.zeros(self.num_layers, 1, self.hidden_size).cuda())
	# 	self.c = Variable(torch.zeros(self.num_layers, 1, self.hidden_size).cuda())

model = RNN(5, hidden_size, num_layers, 5).cuda()

criterion = nn.MSELoss().cuda()

optimizer = optim.Adam(model.parameters())
##################################################




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
	

	counter = 0

	loss_this_epoch = 0
	print("\n\nEPOCH:\t"+str(epoch)+" / " + str(num_epochs) + "\n\n")
	for i in range(len(music_files)):		# songs

		h = Variable(torch.zeros(num_layers, 1, hidden_size).cuda(), requires_grad=True)
		c = Variable(torch.zeros(num_layers, 1, hidden_size).cuda(), requires_grad=True)

		print("song " + str(i+1) + " / " + str(range(len(music_files))))
		for j in range(music_files[i].shape[0]-1): # notes

			optimizer.zero_grad()

			input = Variable(torch.from_numpy(music_files[i][j]), requires_grad=False).view(1,1,5).float()
			input = input.cuda()

			target = Variable(torch.from_numpy(music_files[i][j+1]), requires_grad=False).view(1,1,5).float()
			target = target.cuda()

			output, (h,c) = model(input, h, c)

			#detatch memory from graph to truncate backpropagation
			if(j%truncate_length==0):
				h = Variable(h.data)
				c = Variable(c.data)


			loss = criterion(output, target)

			if(j % 50 == 0 and j != 0):
				print("note " + str(j) + " / " + str(music_files[i].shape[0]-1) + "\navg loss:\t" + str(loss_this_epoch/j))


			loss.backward(retain_variables=True)

			optimizer.step()

			loss_this_epoch += loss.data
			

		if((i % checkpoint_distance == 0 and i != 0) or (epoch % checkpoint_distance == 0 and epoch != 0 and i == 0)):

			h = Variable(torch.zeros(num_layers, 1, hidden_size).cuda(), requires_grad=False, volatile=True)
			c = Variable(torch.zeros(num_layers, 1, hidden_size).cuda(), requires_grad=False, volatile=True)

			print("checkpoint at epoch " + str(epoch))

			#	T O  D O 	save weights to: (checkpoint_path + "/" + str(epoch) + "_epochs_weights")
			#checkpoint_file = open(str("model_checkpoints/" + run_name + "/" + str(epoch) + "_epochs"), "wb")
			torch.save(model.state_dict(), str("model_checkpoints/" + run_name + "/" + str(i) + "_songs"))
			print("saved")
			#	T O  D O 	generate song 
			generated_song = []

			#for now, starts with first note of the song
			note = Variable(torch.from_numpy(music_files[0][0]).cuda(), requires_grad=False, volatile=True).view(1,1,5).float()
			note, (h,c) = model(note, h,c)
			note = note.view(1,1,5)
			

			generated_song.append(note.cpu().data.numpy())#np.random.randint(2, size=num_notes) ) )

			for _ in range(999):
				note, (h,c) = model(note, h, c)

				h = Variable(h.data)
				c = Variable(c.data)
				

				note = note.view(1,1,5)
				generated_song.append(note.cpu().data.numpy())

			#convert to np array and round pitch, velocity, and channel to integer values, then clip them to the appropriate range
			generated_song = np.array(generated_song)

			
			generated_song = generated_song[:,0,0,:] #manual reshaping

			#generated_song = (generated_song*STD)+AVG
			
			#	restructuring generated song for the proper midi format
			generated_song[:, (1,3,4)] = np.round(generated_song[:,(1,3,4)])
			generated_song[:, (1,3,4)] = np.clip(generated_song[:,(1,3,4)], 0, 127)
			generated_song = np.nan_to_num(generated_song)

			print(generated_song[:10])

			print(generated_song.shape)

			new_midifile = MIDIFile.from_notes(generated_song)

			new_midifile.write("generated_song_attempts/" + run_name + "/" + str(i) + "i" + str(epoch) + "e" + ".mid")

			np.savetxt("models/" + run_name + "__loss_trend", loss_list)

	print("epoch loss:\t"+str(loss_this_epoch.cpu().numpy()))
	loss_list.append(loss_this_epoch.cpu().numpy())




#  	T O  D O
#				determine which data to feed in
#				tweak neural net

#				set up reinforcement learning