import numpy as np

class NTM:

	def __init__(input_shape, output_shape, num_heads=10, mem_shape=(128,32)):
		#initialize memory block as all zeroes
		self.memory_block = np.zeros(shape=mem_shape, dtype=float)

		self.read_heads = [read_head(mem_shape, i, num_heads) for i in range(num_heads)]
		
		self.write_heads = [read_head(mem_shape, i, num_heads) for i in range(num_heads)]



class read_head:

	def __init__(mem_shape, id, num_heads):

	def read(weight_vec, memory_block):
		#multiply the weight vector across the memory locations and sum them up to get the read vector.
		# for 2d memory, weight_vec.shape[0] == memory_block.shape[0], read_vec.shape[0] == memory_block.shape[1]
		read_vec = weight_vec.dot(memory_block)

		return read_vec


class write_head:

	def __init__(mem_shape, id, num_heads):

	def write(weight_vec, erase_vec, write_vec, memory_block):
		
		#creates a block the same shape as memory_block, where each cell tells how much to erase from that cell in memory_block
		percent_of_each_cell_to_erase = np.outer(weight_vec,erase_vec) 

		erase_mask = np.ones(memory_block.shape) - percent_of_each_cell_to_erase

		memory_block = memory_block * erase_mask

		#since the memory location of memory_block is passed in, this operation modifies the actual memory block, so nothing need be returned.