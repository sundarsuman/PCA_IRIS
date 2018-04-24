import numpy as np
import csv
import re
import matplotlib.pyplot as plt

iris_measurements = {}
iris_labels = {}
with open('Iris.csv','r') as f:
	reader = list(csv.reader(f))

reader.pop(0)


for entity in reader:
	for i in range(len(entity)):
		entity[0] = float(entity[0])
		entity[1] = float(entity[1])
		entity[2] = float(entity[2])
		entity[3] = float(entity[3])
		entity[4] = float(entity[4])
	iris_measurements[entity[0]] = entity[1:-1]
	iris_labels[entity[0]] = entity[-1]	

X, labels = [],[]
for values in iris_measurements.values():
	X.append(values)

for values in iris_labels.values():
	labels.append(values)

X = np.asarray(X, dtype = np.float32)
labels = np.asarray(labels)

U, sigma, V = np.linalg.svd(X, full_matrices = True)

S = np.zeros(X.shape)
np.fill_diagonal(S,sigma)

X_r = np.dot(np.dot(U,S),V[:,:3])
# print X_r

with open('Iris_PCA_3D.txt','w') as f_out:
	for item in X_r:
		for i in item:
			f_out.write(str(i)+' ')
		f_out.write('\n') 

# plt.scatter(X_r[:,0],X_r[:,1])
# plt.show()

labels_guessed = []
with open('IRIS_IDandClusters.txt') as fin:
	for string in fin:
		[a,b] = string.strip().split()
		labels_guessed.append(int(b))

label = iris_labels.values()
label_original = []
for x in label:
	if(x == 'Iris-setosa'):
		label_original.append(2)
	if(x == 'Iris-versicolor'):
		label_original.append(1)
	if(x == 'Iris-virginica'):
		label_original.append(0)

# print label_original[0]
# print labels_guessed[0]

accuracy = 0.0
for i in range(len(label_original)):
	if (labels_guessed[i] == label_original[i]):
		accuracy = accuracy + 1


print float(accuracy/len(label_original))*100