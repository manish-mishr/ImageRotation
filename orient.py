import sys
from solver import Solver


# Read the training data and test data from the file provided.
def read_data(fname):
	exemplars = []
	with open (fname,'r') as file:
		total_data = file.readlines()
		for data in total_data:
			data = data.split()
			data_list = [int(num) for num in data[2:]]
			exemplars += [(data[0],data[1],data_list),]

	file.close()
	return exemplars


if len(sys.argv) != 5:
    print "Usage: python orient.py training_file test_file algorithm count"
    sys.exit()

(train_file,test_file,algo,count) = sys.argv[1:]

data = read_data(train_file)

test_data = read_data(test_file)

# min_test = test_data[800:820]

# test_data = min_test

matrix = [[0]*4]*4

def accuracy(test_data,method):
	file = open("result.txt",'a')
	correct = 0
	wrong = 0
	for tdata in test_data:
		degree = method(data,tdata[2],count)
		mystr = ""
		mystr += "actual  data:  "
		mystr += tdata[1]
		mystr += "   and predicted data:   "
		mystr += degree
		mystr += "\n"
		file.write(mystr)
		if degree == tdata[1]:
			correct += 1
		else:
			wrong += 1
		row = int(tdata[1])/90
		col = int(degree)/90
		matrix[row][col] += 1
	acc = float(correct)/(wrong+correct)
	file.close()
	return (acc*100)

solver = Solver()


if algo.lower() == 'knn':
 	print "acc. %:  ", accuracy(test_data,solver.knn)
elif algo.lower() == 'nnet':
 	print "acc. %:  ", accuracy(test_data,solver.nnet)
else:
 	print "acc. %:  ", accuracy(test_data,solver.best)




