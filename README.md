# EGA-FMC: Enhanced Genetic Algorithm based Fuzzy K - Modes Clustering for Categorical Data
We implemented a fuzzy K-Modes clustering algorithm using Genetic Algorithm for categorical data. The algorithm is an extension to the work,"[A genetic fuzzy k-Modes algorithm for clustering categorical data](http://dl.acm.org/citation.cfm?id=1465302)". 

Outline of the proposed algorithm:                                                                                                         
1. Intitialize the chromosomes. Each chromosome is a membership matrix representing a possible solution to the clustering problem.         
2. Evaluating the fitness of the chromosomes.                                                                                               
3. Choosing the best parent chromosome.                                                                                                     
4. Multi-objective rank-based assignment and Roulette Wheel Selection, Crossover and Mutation operations.                                   
5. Elitism Operation: Replacing the worst child with the best parent.                                                                       
6. Repeat until termination.                                                                                                               
7. Clustering solution is the choromosome with the best value for Arithment Rand Index, Inter-cluster separation, and Intre-cluster distance.                                                                                                                                   

The method outperforms the state-of-the-art algorithms in terms of Arithment Rand Index, Inter-cluster separation, Intre-cluster distance and Computation Time. 
