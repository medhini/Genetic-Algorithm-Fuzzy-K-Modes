import scipy
from scipy import spatial
import pandas
import numpy
import math
import random
from collections import Counter

data=pandas.read_csv('data.csv').as_matrix()[:,:-1]
data=data.astype(int)
print data[7]

chromosome = [[] for i in range(10)]

for i in range (10):
	chromosome[i].append(data[i])
	chromosome[i].append(data[i*4+1])
	chromosome[i].append(data[i*4+2])
	chromosome[i].append(data[i*4+3])

print chromosome
c=0
m=2
labels = numpy.random.random_integers(0,3 ,(len(chromosome),len(data)))

membership_new=numpy.tile(0.0, (len(chromosome[0]), data.shape[0]))

for c in range(len(chromosome)):
	for i, centre in enumerate(chromosome[c]):
		for j, point in enumerate(data):
			su=0.0
			k=0
			while k < len(chromosome[c]):
				#print "centre[i] for i="+str(i)+" is "
				#print centre[i]
				#print "centre[k] for k="+str(k)+" is "
				#print centre[k]
				#print "data[i] for i="+str(i)+" is "
				#print data[i]
				su+=math.pow((((scipy.spatial.distance.hamming(data[j],centre[i])))/((scipy.spatial.distance.hamming(data[j],centre[k])))),(2/(m-1)))
				#print "su is "+str(su)
				k+=1

			#print "su is " + str(su) + "with j="+str(j)
			#print "1/su is"+str(1/su)
			membership_new[i][j]=1/float(su)

	print "matrix is"
	print membership_new
	for i in range(len(data)):
		labels[c][i]=list(membership_new[:,i]).index(max(membership_new[:,i]))
	
	print labels




vote=[]
print labels
for i in range(len(data)):
	modes=Counter(labels[:,i])
	print modes.most_common(1)[0][1]
	if (modes.most_common(1)[0][1] >=5):
		vote.append(i)

print vote
	




