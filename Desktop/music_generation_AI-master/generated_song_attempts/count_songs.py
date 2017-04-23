import os

count = 0
for directory in os.listdir("."):
	if not directory=="count_songs.py":
		for song in os.listdir(directory):
			count += 1

print(count)