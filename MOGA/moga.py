from deap import base, creator
from deap import tools
import scipy
from scipy import spatial
import pandas
import numpy
import math

no_clusters=4
def evaluate(individual):
	m=2
	data=pandas.read_csv('data.csv').as_matrix()[:,:-1]
	data=data.astype(int)
	categories=[["		april","may","june","july","august","september","october","?"],
	["	normal","lt-normal","?"],
	["		lt-norm","norm","gt-norm","?"],
	["		lt-norm","norm","gt-norm","?"],
	["		yes","no","?"],
	["	diff-lst-year","same-lst-yr","same-lst-two-yrs","same-lst-sev-yrs","?"],
	["	scattered","low-areas","upper-areas","whole-field","?"],
	["	minor","pot-severe","severe","?"],
	["	none","fungicide","other","?"],
	["	90-100%","80-89%","lt-80%","?"],
	["	norm","abnorm","?"],
	["		norm","abnorm"],
	["	absent","yellow-halos","no-yellow-halos","?"],
	["	w-s-marg","no-w-s-marg","dna","?"],
	["	lt-1/8","gt-1/8","dna","?"],
	["	absent","present","?"],
	["	absent","present","?"],
	["	absent","upper-surf","lower-surf","?"],
	["		norm","abnorm","?"],
	["    	yes","no","?"],
	["	absent","below-soil","above-soil","above-sec-nde","?"],
	["	dna","brown","dk-brown-blk","tan","?"],
	["	absent","present","?"],
	["	absent","firm-and-dry","watery","?"],
	["	absent","present","?"],
	["	none","brown","black","?"],
	["	absent","present","?"],
	["	norm","diseased","few-present","dna","?"],
	["	absent","colored","brown-w/blk-specks","distort","dna","?"],
	["		norm","abnorm","?"],
	["	absent","present","?"],
	["	absent","present","?"],
	["	norm","lt-norm","?"],
	["	absent","present","?"],
	["		norm","rotted","galls-cysts","?"],
	]
	print data

	membership_new=numpy.tile(0.0, (no_clusters, data.shape[0]))
	attrisum = categories

	for i in range (len(attrisum)):
		for j in range (len(attrisum[i])):
			attrisum[i][j]=0

	for i, centre in enumerate(individual):
		for j, point in enumerate(data):
			su=0.0
			k=0
			while k < no_clusters:
				print "centre[i] for i="+str(i)+" is "
				print centre[i]
				print "centre[k] for k="+str(k)+" is "
				print centre[k]
				print "data[i] for i="+str(i)+" is "
				print data[i]
				su+=math.pow((((scipy.spatial.distance.hamming(data[j],centre[i])))/((scipy.spatial.distance.hamming(data[j],centre[k])))),(2/(m-1)))
				print "su is "+str(su)
				k+=1

			print "su is " + str(su) + "with j="+str(j)
			print "1/su is"+str(1/su)
			membership_new[i][j]=1/float(su)

			
	print "membership_new is"
	print membership_new
	print attrisum


	###		UPDATING CLUSTER CENTRES BY SUMMING MEMBERSHIP VALUES OVER ATTRIBUTES	###
	for centre in range (len(individual)):

		for atrcount in range (len(attrisum)):
			for i in range (len(data)):
				attrisum[atrcount][data[i][atrcount]] += membership_new[centre][i]

			individual[centre][atrcount] = attrisum[atrcount].index(max(attrisum[atrcount]))

		
		print attrisum
		print individual[centre]

	for i, centre in enumerate(individual):
		for j, point in enumerate(data):
			su=0.0
			k=0
			while k < no_clusters:
				print "centre[i] for i="+str(i)+" is "
				print centre[i]
				print "centre[k] for k="+str(k)+" is "
				print centre[k]
				print "data[i] for i="+str(i)+" is "
				print data[i]
				su+=math.pow((((scipy.spatial.distance.hamming(data[j],centre[i])))/((scipy.spatial.distance.hamming(data[j],centre[k])))),(2/(m-1)))
				print "su is "+str(su)
				k+=1

			print "su is " + str(su) + "with j="+str(j)
			print "1/su is"+str(1/su)
			membership_new[i][j]=1/float(su)

	print individual
	print "membership_new is"
	print membership_new

	# Calculating pi 
	pi=0.0
	for j,indi in enumerate(individual):
		s=0.0
		num=0.0
		#print numpy.shape(num)
		for i , da in enumerate(data):
			s+=(math.pow(membership_new[j][i],m))
			#print "membership[j][i] is " + str(membership[j][i])
			#print "data[i] is " + str(data[i])
			#print "(math.pow(membership[j][i],m))*data[i] is " +str((math.pow(membership[j][i],m))*data[i])
			num+=(math.pow(membership_new[j][i],m))*scipy.spatial.distance.hamming(data[i],individual[j])
			#print "num 1st is "
			#print num 
			print " s is for i= "+str(i)+" is "+str(s)
			print "num is "
			print num 
		pi+=num/s

	membership_clusters=numpy.tile(0.0, (no_clusters, no_clusters))

	for i, centre in enumerate(individual):
		for j, point in enumerate(individual):
			su=0.0
			k=0
			if i!=j:
				while k < no_clusters:
					if j!=k:
						#print "centre[i] for i="+str(i)+" is "
						#print centre[i]
						#print "centre[k] for k="+str(k)+" is "
						#print centre[k]
						#print "data[i] for i="+str(i)+" is "
						#print data[i]
						su+=math.pow((((scipy.spatial.distance.hamming(individual[j],individual[i])))/((scipy.spatial.distance.hamming(data[j],centre[k])))),(2/(m-1)))
						print "su is "+str(su)
						print "i is" +str(i)
						print "j is"+str(j)
						print "k is"+str(k)
					k+=1

				if su!=0.0:
					print "su is " + str(su) + "with j="+str(j)
					print "1/su is"+str(1/su)
					membership_clusters[i][j]=1/float(su)
				
	# Calculating sep
	sep=0.0
	for i, c1 in enumerate(individual):
		sum2=0.0
		for j, c2 in enumerate(individual):
			if j!=i:
				sum2+=(math.pow(membership_clusters[i][j],m)*scipy.spatial.distance.hamming(individual[i],individual[j]))
		sep+=sum2
	return pi,1/sep



data=pandas.read_csv('data.csv').as_matrix()[:,:-1]
data=data.astype(int)

popu=[[data[1], data[10], data[24], data[26]],[data[2], data[12], data[26], data[30]]]

CXPB, MUTPB, NGEN = 0.5, 0.2, 40

creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()
toolbox.register("individual", creator.Individual,
                 n=data.shape[1])
toolbox.register("population",  list, popu, toolbox.individual)
pop = toolbox.population()

toolbox.register("evaluate", evaluate)
fitnesses = map(toolbox.evaluate, pop)
for ind, fit in zip(pop, fitnesses):
	ind.fitness.values = fit

toolbox.register("select", tools.selTournament, tournsize=2)

for g in range(NGEN):
	
	offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
	offspring = map(toolbox.clone, offspring)

	for child1, child2 in zip(offspring[::2], offspring[1::2]):
		if random.random() < CXPB:
			toolbox.mate(child1, child2)
			del child1.fitness.values
			del child2.fitness.values

	pop[:] = offspring

