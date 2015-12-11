
import math, sys, operator

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

	for line in train_file:
		split = line.split(' ')
		train[split[0]+split[1]] = [int(i) for i in split[1:]]

	for line in test_file:
		split = line.split(' ')
		test[split[0]+split[1]] = map(int,split[1:])

	if algo == 'knn':
		classifier = knn(train,param)
		res = classifier.test(test)
		print res
	elif algo == 'nnet':
		network = neuralnet(param)
		network.train(train)
		

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
			print predclass, trueclass
			if predclass == trueclass:
				correct += 1
			cmatrix[trueclass/90][predclass/90] += 1
			outfile.write(name.split('g')[0] + "g " + str(predclass) + "\n")

		outfile.close()
		print "Accuracy: " + (correct*1.0)/testCount
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

	
	def dist(self,a,b):
		distance = 0
		for i in range(1, len(a)):
			distance += pow(a[i] - b[i], 2)
		return math.sqrt(distance)




if __name__ == "__main__":
	main()