
from Queue import PriorityQueue
from collections import defaultdict
from math import sqrt

class Solver:
	def __init__(self):
		pass

	def knn(self,data,test_data,kcount):
		k_queue = PriorityQueue()
		for item in data:
			total_sum = 0
			for i,j in zip(test_data,item[2]):
				total_sum += sqrt(abs(i**2-j**2))
			label = (item[0],item[1])
			k_queue.put((total_sum,label))
			
	
		rotate = defaultdict(int)
		kcount = int(kcount)
		for count in range(kcount):
			(cost,label_tuple) = k_queue.get()
			rotate[label_tuple[1]] += 1
		
		
		highest = 1
		degree = ""
		for key in rotate:	
			if rotate[key] > highest:
				degree = key 
				highest = rotate[key]


		return degree

	def nnet(self,data,hidden_count):
		pass

	def best(self,data,para):
		pass
