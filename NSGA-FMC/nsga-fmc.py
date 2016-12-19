import numpy as np
import urllib
import math
import random

from sklearn import preprocessing
from collections import defaultdict
from fractions import Fraction

X_Features = []
def dissimilarityMeasure(X, Y): 
	""" Simple matching disimilarity measure """
	return np.sum(X!=Y, axis = 0)
	
def calculateCentroids(membership_mat, X_Features, alpha):

	n_points, n_attributes = X_Features.shape
	n_clusters = membership_mat.shape[1]

	WTemp = np.power(membership_mat, alpha)
	centroids = np.zeros((n_clusters,n_attributes))

	for z in xrange(n_clusters):
		for x in xrange(n_attributes):
			freq = defaultdict(int)
			for y in xrange(n_points):
				freq[X_Features[y][x]] += WTemp[y][z]

			centroids[z][x] = max(freq, key = freq.get)
	
	centroids = centroids.astype(int)

	return centroids

def separation(centroids, membership_mat):

	sep = 0.0
	k = len(centroids)
	for x in xrange(k):
		for y in xrange(x, k, 1):
			sep += np.power(membership_mat[i][k], alpha)*dissimilarityMeasure(centroids[x], centroids[y])
	return

"""Compactness or CostFunction"""
def costFunction(membership_mat, n_clusters, n_points, alpha, centroids, X_Features):
	
	cost_function = 0.0
	
	for k in xrange(n_clusters):
		temp = 0.0
		denom = 0.0
		for i in xrange(n_points):
			temp += np.power(membership_mat[i][k], alpha)*dissimilarityMeasure(X_Features[i], centroids[k])
			denom += np.power(membership_mat[i][k], alpha)
		
		temp = temp/denom
		cost_function += temp

	return cost_function

def comparatorCompactness(membership_mat_1, membership_mat_2):

	alpha = 1.2
	n_points = 47
	n_clusters = 4

	np.reshape(membership_mat_1, (-1, n_clusters))
	np.reshape(membership_mat_2, (-1, n_clusters))
	
	centroids_1 = calculateCentroids(membership_mat_1, X_Features, alpha)
	centroids_2 = calculateCentroids(membership_mat_2, X_Features, alpha)

	compactness_1 = costFunction(membership_mat_1, n_clusters, n_points, alpha, centroids_1, X_Features)
	compactness_2 = costFunction(membership_mat_2, n_clusters, n_points, alpha, centroids_2, X_Features)

	return compactness_2 - compactness_1

def comparatorSeparation():

	return

def updateMatrix(centroids, X_Features, n_points, n_clusters, n_attributes, alpha):

	exp = 1/(float(alpha - 1))
	for x in xrange(n_clusters):
		centroid = centroids[x]
		for y in xrange(n_points):
			
			hammingDist = dissimilarityMeasure(centroid, X_Features[y])
			numerator = np.power(hammingDist, exp)
			denom = 0.0
			flag = 0
			
			for z in xrange(n_clusters):
				if (centroids[z] == X_Features[y]).all() and (centroids[z] == centroid).all():
					membership_mat[y][x] = 1
					flag = 1
					break
				elif (centroids[z] == X_Features[y]).all():
					membership_mat[y][x] = 0
					flag = 1
					break

				denom += np.power(dissimilarityMeasure(centroids[z], X_Features[y]), exp)
		
			if flag == 0:
				membership_mat[y][x] = 1/(float(numerator)/float(denom))	 	 
			
	for row in range(len(membership_mat)):
			membership_mat[row] = membership_mat[row]/sum(membership_mat[row])

	cost_function = costFunction(membership_mat, n_clusters, n_points, alpha, centroids, X_Features)
	return membership_mat, cost_function

def fuzzyKModes(membership_mat, X_Features, alpha, max_epochs):
	
	n_points, n_clusters = membership_mat.shape
	n_attributes = X_Features.shape[1]

	centroids = np.zeros((n_clusters,n_attributes))
	epochs = 0
	oldCostFunction = 0.0
	costFunction = 0.0

	while(epochs < max_epochs):
		centroids = calculateCentroids(membership_mat, X_Features, alpha)
		membership_mat, costFunction = updateMatrix(centroids, X_Features, n_points, n_clusters, n_attributes, alpha)

		if((oldCostFunction - costFunction)*(oldCostFunction - costFunction) < 0.3):
			break
		epochs += 1
	
	return membership_mat, costFunction

def Selection(chromosomes, n, k):

	"""Rank Based Fitness Assignment"""
	
	#Sort chromosomes for rank based evaluation
	chromosomes = chromosomes[chromosomes[:,n*k].argsort()]
	newChromosomes = np.zeros((n, n*k + 1))

	beta = 0.1
	fitness = np.zeros(n)
	cumProbability = np.zeros(n)

	for i in xrange(n - 1, 0, -1):
		fitness[i] = beta*(pow((1 - beta), i))

	"""Roulette Wheel Selection"""

	#Cumulative Probability
	for i in xrange(n):
		if i > 1:
			cumProbability[i] = cumProbability[i-1] 
		cumProbability[i] += fitness[i]
	
	#Random number to pick chromosome
	for i in xrange(n):
		pick = random.uniform(0,1)

		if pick < cumProbability[0]:
			newChromosomes[i] = chromosomes[0]
		else :	
			for j in xrange(n - 1):
				if cumProbability[j] < pick and pick < cumProbability[j + 1]:
					newChromosomes[i] = chromosomes[j + 1]
		
		newChromosomes[i][n*k] = 0.0
	
	return newChromosomes

def CrossOver(chromosomes, n, k, X_Features, alpha):

	newChromosomes = np.zeros((n, n * k + 1))
	
	for i in xrange(n):
		membership_mat = np.reshape(chromosomes[i][0:n*k], (-1, k))
		new_membership_met, cost_function = fuzzyKModes(membership_mat, X_Features, alpha, 1)    #Quick termination, 1 step fuzzy kmodes
		newChromosomes[i][0 : n * k] = new_membership_met.ravel()
		newChromosomes[i][n * k] = cost_function

	return newChromosomes

def Mutation(chromosomes, n_points, n_clusters):

	P = 0.001
	for i in xrange(n_points):
		chromosome = chromosomes[i][0 : n * k]
		chromosome = np.reshape(chromosome, (-1, n_clusters))

		for j in xrange(n_points):
			pick = random.uniform(0,1)
			if pick <= P:
				gene = np.random.rand(k)
				gene = gene/sum(gene)
				chromosome[j] = gene

		chromosomes[i][0 : n * k] = chromosome.ravel()

	return chromosomes

def crowdingDistanceAssignment(chromosomes, n_clusters, n_points):

	distance = np.zeros(n_points)

	sorted(chromosomes, cmp = comparatorCompactness)
	distance[0] = distance[n_points-1] = 10000007

	fMax = costFunction(np.reshape(chromosomes[0][0 : n_clusters * n_points] , (-1, n_clusters)))
	fMin = costFunction(np.reshape(chromosomes[n_points - 1][0 : n_clusters * n_points], (-1, n_clusters)))

	denom = fMax - fMin
	for x in xrange(1,n_points - 1):
		distance[x] += (costFunction(np.reshape(chromosomes[x + 1], (-1, n_clusters))) - np.reshape(chromosomes[x - 1], (-1, n_clusters)))/denom
	
	# sorted(chromosomes, cmp = comparatorCompactness(n_clusters, n_points, alpha, X_Features))
	
	return distance	

if __name__ == "__main__":
	dataset = 'soybean.csv'

	# load the CSV file as a numpy matrix

	soyData = np.genfromtxt(dataset, delimiter=',', dtype = 'str')
	X_Features = soyData[:, 0:35].astype(int)
	YLabels = preprocessing.LabelEncoder().fit_transform(soyData[:, 35])  #Convert label names to numbers

	k = 4
	n = len(X_Features)
	n_attributes = X_Features.shape[1]
	alpha = 1.2
	max_epochs = 100
	g_max = 15

	populationSize = n
	chromosomes = np.zeros((n, n * k + 1))

	"""Initialize Population"""
	for i in xrange(populationSize):
		
		membership_mat = np.random.rand(n, k)

		for row in range(len(membership_mat)):
			membership_mat[row] = membership_mat[row]/sum(membership_mat[row])

		chromosomes[i][0 : n * k] = membership_mat.ravel()

		centroids = calculateCentroids(membership_mat, X_Features, alpha)
		chromosomes[i][n*k] = costFunction(membership_mat, k, n, alpha, centroids, X_Features)   #Last column represents the cost function of this chromosome
			
	"""Genetic Algorithm K Modes"""
	for x in xrange(g_max):

		population_after_selection = Selection(chromosomes, n, k)
		population_after_crossover = CrossOver(population_after_selection, n, k, X_Features, alpha)
		chromosomes = Mutation(population_after_crossover, n, k)

		for i in xrange(populationSize):
			membership_mat = np.reshape(chromosomes[i][0:n*k], (-1, k))
			centroids = calculateCentroids(membership_mat, X_Features, alpha)
			chromosomes[i][n*k] = costFunction(membership_mat, k, n, alpha, centroids, X_Features)   #Last column represents the cost function of this chromosome
		#Elitism non dominated sorting


	distance = crowdingDistanceAssignment(chromosomes, k, n)

	print distance
	"""Best of the child chromosomes"""
	min_value = 0
	offspring = chromosomes[0]

	for i in xrange(populationSize):
		if min_value == 0:
			min_value = chromosomes[i][n*k]

		elif chromosomes[i][n*k] < min_value:
			min_value = chromosomes[i][n*k]
			offspring = chromosomes[i]

	print "Final Surviving chromosomes : ", chromosomes
	print "Final chosen chromosome : ", offspring
	print "Compactness : ", min_value

	






	 

