
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
		self.numPasses = 10
		self.reg = 0.01
		self.epsilon = 0.01
		self.trainset = trainset
		self.train_dim = train_dim
		self.numHidden = numHidden
		self.numLayers = numLayers
		self.numOutput = numOutput
		self.model = {}

	def predict(self, x):
		s = self
		inpa = np.array(x)
		z1 = inpa.dot(self.model['w1']) + s.model['b1']
		a1 = np.tanh(z1)
		z2 = a1.dot(s.model['w2']) + s.model['b2']
		exp_scores = np.exp(z2)
		probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
		return np.argmax(probs, axis=1)

	def train(self):
		s = self

		for tr in s.trainset:
			inp.append(s.trainset[tr][1:])
			lst = [0] * 4
			lst[s.trainset[tr][0]/90] = 1
			out.append(lst)



		#initialize all of our hidden layers (each has numHidden neurons)
		np.random.seed(int(time.time()))
		W1 = np.random.randn(s.train_dim, s.numHidden) / np.sqrt(s.train_dim)
		b1 = np.zeros((1, s.numHidden))
		
		W2 = np.random.randn(s.numHidden, s.numOutput) / np.sqrt(s.numHidden)
		b2 = np.zeros((1, s.numOutput))
		

		'''
		numWeights = s.train_dim
		last = 0
		for i in range(s.numLayers):
			s.model['w'+str(i)] = np.random.randn(numWeights, s.numHidden) / np.sqrt(numWeights)
			s.model['b'+str(i)] = np.zeros((1, s.numHidden))
			numWeights = s.numHidden
			last = i
		last += 1
		s.model['w'+str(last)] = np.random.randn(s.numHidden, s.numOutput) / np.sqrt(s.numHidden)
		s.model['b'+str(last)] = np.zeros((1, s.numOutput))
		'''
		inpa = np.array(inp)
		outa = np.array(out)
		for i in range(s.numPasses):
				# Forward propagation

				z1 = inpa.dot(W1) + b1
				a1 = s.sigmoid(z1)
				z2 = a1.dot(W2) + b2
				a2 = np.sigmoid(z2)
				probs = a2

				# Backpropagation
				delta3 = np.multiply(-(outa-probs),s.sigmoidprime(z2))
				# delta3[range(len(inpa)), outa] -= 1
				dW2 = (a1.T).dot(delta3)
				db2 = np.sum(delta3, axis=0, keepdims=True)
				delta2 = delta3.dot(W2.T) * s.sigmoidprime(z1)
				dW1 = np.dot(inpa.T,delta2)
				db1 = np.sum(delta2, axis=0)

				# Add regularization terms (b1 and b2 don't have regularization terms)
				dW2 += s.reg * W2
				dW1 += s.reg * W1

				# Gradient descent parameter update
				W1 += -s.epsilon * dW1
				b1 += -s.epsilon * db1
				W2 += -s.epsilon * dW2
				b2 += -s.epsilon * db2

				s.model = { 'w1': W1, 'b1': b1, 'w2': W2, 'b2': b2}
				'''
				#adapted for multiple layers from http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/
				#currently has a size mismatch in back propogation step line d = d.dot(s.model['w'+str(l)]) * (1 - np.power(a[l],2))



				#forward propogation	
				a = [np.array(inp)]
				z = None
				for l in range(s.numLayers):
					w = 'w'+str(i)
					b = 'b'+str(i)
					z = a[l].dot(s.model[w]) + s.model[b]
					a.append(np.tanh(z))
				exp = np.exp(z)
				prob = exp / np.sum(exp, axis=1, keepdims=True)
				# print prob

				#back propogation
				delta = {}
				d = prob
				for l in range(s.numLayers, -1, -1):
					d[range(len(inp)),out] -= 1
					delta['w'+str(l)] = (a[l].T).dot(d)
					delta['b'+str(l)] = np.sum(d, axis=0, keepdims=True)
					print l
					d = d.dot(s.model['w'+str(l)]) * (1 - np.power(a[l],2))

				for key in delta:
					if 'w' in key:
						delta[key] += s.reg * s.model[key]

				for key in model:
					s.model[key] += -s.epsilon * delta[key]
				'''
	def sigmoid(self,z):
		return 1/(1+np.exp(-z))

	def sigmoidprime(self,z):
		return np.exp(-z)/((1+np.exp(-z))**2)



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