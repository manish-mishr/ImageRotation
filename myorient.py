
import math, sys, operator, time
import numpy as np

def main():
	argv = sys.argv
	train_filename = str(argv[1])
	test_filename = str(argv[2])
	algo = str(argv[3])
	param = int(argv[4])

	train_file = open(train_filename)
	test_file = open(test_filename)

	train = {}
	test = {}

	trainDim = 0

	for line in train_file:
		split = line.split(' ')
		train[split[0]+split[1]] = [int(i) for i in split[1:]]
		if trainDim == 0: trainDim = len(train[split[0]+split[1]]) - 1

	for line in test_file:
		split = line.split(' ')
		test[split[0]+split[1]] = map(int,split[1:])

	if algo == 'knn':
		classifier = knn(train,param)
		res = classifier.test(test)
		for row in res:
			print row
	elif algo == 'nnet':
		network = neuralNet(train, trainDim, param)
		network.train()
		for te in test:
			print "true ", test[te][0], "pred ", network.predict(test[te][1:])

		

class neuralNet:

	def __init__(self, trainset, train_dim, numHidden, numLayers=1, numOutput=4):
		self.numPasses = 500
		self.reg = 0.01
		self.epsilon = 0.01
		self.trainset = trainset
		self.train_dim = train_dim
		self.numHidden = 192
		self.numLayers = numLayers
		self.numOutput = numOutput
		self.model = {}

	def predict(self, x):

		print("reached predicted")
		s = self
		x = np.array(x)
		inpa = np.insert(x,0,1)
		a2 = np.zeros(s.numHidden)
		for j in range(s.numHidden):
			z1 = inpa.dot(s.model['w1'][j]) 
			a2[j] = s.sigmoid(z1)
		
		a2 = np.insert(a2,0,1)
		a3 = np.zeros(s.numOutput)
		for k in range(s.numOutput):
			z2 = a2.dot(s.model['w2'][k])
			a3[k] = s.sigmoid(z2)
		
		return np.argmax(a3)

	def train(self):
		s = self
		inp = [s.trainset[tr][1:] for tr in s.trainset]
		out = [s.trainset[tr][0]/90 for tr in s.trainset]


		#initialize all of our hidden layers (each has numHidden neurons)




		
		np.random.seed(int(time.time()))
		W1 = np.random.randn(s.numHidden,s.train_dim+1) / np.sqrt(s.train_dim)
		W2 = np.random.randn(s.numOutput,s.numHidden+1) / np.sqrt(s.numHidden)
		
		delta2 = np.zeros((s.numOutput,s.numHidden+1))
		delta1 = np.zeros((s.numHidden,s.train_dim+1))

		

		for i in range(5000):
			print "current count:   ", str(i)
				# Forward propagation
			temp1 = np.array(inp[i])
			curr_arr1 = np.insert(temp1,0,1)
			a2_temp = np.zeros(s.numHidden)
			for j in range(s.numHidden):
				z1 = curr_arr1.dot(W1[j])
				a2_temp[j] = s.sigmoid(z1)


			a2_arr = np.array(a2_temp)
			curr_arr2 = np.insert(a2_arr,0,1)
			a3_temp = np.zeros(s.numOutput)
			for k in range(s.numOutput):
				z2 = curr_arr2.dot(W2[k])
				a3_temp[k] = s.sigmoid(z2)
				

			a3_arr = np.array(a3_temp)
			
			result_vec = s.ret_vec(out,i)	
			# Backpropagation

			lambda3 = a3_arr - result_vec
			temp_lambda = np.dot(W2.T,lambda3)
			mul = curr_arr2 * (1-curr_arr2)
			lambda2 = mul*temp_lambda
			
			for row in range(s.numOutput):
				for col in range(s.numHidden+1):
					delta2[row,col] = delta2[row,col] + curr_arr2[col]*lambda3[row] 
			
			for row in range(s.numHidden):
				for col in range(s.train_dim+1):
					delta1[row,col] = delta1[row,col] + curr_arr1[col]*lambda2[row] 

		print("Completed training instance")

		for row in range(s.numHidden):
			for col in range(s.train_dim+1):
				reg_add = s.reg * W1[row,col] 
				W1[row,col] += (delta1[row,col]/5000)
				if col != 0:
					W1[row,col] += reg_add

		for row in range(s.numOutput):
			for col in range(s.numHidden+1):
				reg_add = s.reg * W2[row,col] 
				W2[row,col] += (delta2[row,col]/5000)
				if col != 0:
					W2[row,col] += reg_add

		print("Reached dictionary")
		s.model = {'w1': W1, 'w2': W2}

				


	def sigmoid(self,x):
		return 1.0/(1+np.exp(-x))

	def ret_vec(self,lst,ind):
		result = [0]*4
		ind = lst[ind]
		result[ind] = 1
		return np.array(result)






class knn:

	def __init__(self,train, k):
		self.train = train
		self.k = k

	def test(self,test):
		prediction = {}
		outfile = open("knn_output.txt", "w")
		testCount = len(test)
		correct = 0
		for entry in test:
			neighbors = self.getNeighbors(test[entry])
			prediction[(entry, test[entry][0])] = self.mostLikely(neighbors)
		cmatrix = [[0 for i in range(4)] for j in range(4)]
		for (name, trueclass) in prediction:
			predclass = prediction[(name, trueclass)]
			# print predclass, trueclass
			if predclass == trueclass:
				correct += 1
			cmatrix[trueclass/90][predclass/90] += 1
			outfile.write(name.split('g')[0] + "g " + str(predclass) + "\n")

		outfile.close()
		print "Accuracy: " + str((correct*1.0)/testCount)
		return cmatrix


	def getNeighbors(self,testRow):
		distances = []
		for tr in self.train:
			distances.append((self.train[tr][0], self.dist(self.train[tr],testRow)))
		distances.sort(key=operator.itemgetter(1))
		return distances[:self.k]

	def mostLikely(self,neigh):
		count = {}
		for n in neigh:
			if not n[0] in count:
				count[n[0]] = 0
			count[n[0]] += 1
		sortedVotes = sorted(count.iteritems(), key=operator.itemgetter(1), reverse=True)
		return sortedVotes[0][0]

	#Euclidian distance
	def dist(self,a,b):
		distance = 0
		for i in range(1, len(a)):
			distance += pow(a[i] - b[i], 2)
		return math.sqrt(distance)




if __name__ == "__main__":
	main()