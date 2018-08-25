# -*- coding: utf-8 -*-
#Aryana Collins Jackson
#Assignment 2

import numpy as np
# print out entire arrays, not truncated versions
np.set_printoptions(threshold=np.nan)
import matplotlib.pyplot as plt 

#    First I create a function to create individuals. An array is created using 
#    the number of cities. The array is then shuffled randomly.

def createRandomIndividual(numCities):

    individual = np.array(range(0,numCities))
    np.random.shuffle(individual)       
    #print("Individual:", individual)

    return individual
# ----------------------------------------------------------------------------

#    Like the function above, I now create the population, which contains 250.
#    It takes in the already set population size and the number of cities as
#    parameters. For every individual in the population, the individual is set
#    as the individual already created in createRandomIndividual() and it's
#    appended to the list. The list is then turned into an array.
    
def createPopulation(popsize, numCities):

    pop = list()

    for i in range(0, popsize):
        individual = createRandomIndividual(numCities)
        pop.append(individual)
    
    popArray = np.array(pop)
    
    return popArray
# ----------------------------------------------------------------------------

#    I calculate the fitness of each individual. The total distance between 
#    cities is used. That is set up as a list. The function iterates through 
#    the cities in the array individual. If i is 0, it calculates the distance
#    between the first and last city. If not, it calculates the distance 
#    between cities. Those values are added to an arry and summed for the total
#    fitness.

def calculateFitness(individual, distancesMatrix):
#    print("ind",individual)
    #print(individual.shape[0])    

    totalDist = list()
    
    for i in range(individual.shape[0]):
        #print(i)
        if i == 0:
            b = individual[14]
        else:
            b = individual[i-1]
        
        a = individual[i]
#        print("a",a)
#        print("b",b)
        dist = distancesMatrix[a,b]
                
        totalDist.append(dist)
    
    totalDistArray = np.array(totalDist)
    
    total = np.sum(totalDistArray)
    
    return total
# ----------------------------------------------------------------------------

#    I then calculate the fitness of the entire population. A list is created
#    to hold the values, then the function iterates over the array holding all 
#    the individuals. Fitness is calculated from the function above. 

#    FOR NEXT ATTEMPT
#    Need to do different tournament style (maybe 20% instead of 2)
#    If converging too quickly: change tournament style (or switch to roulette) 
#    or change crossover or change mutation
        
def calculatePopulationFitness(populationSize, popArray, distancesMatrix):
       
    popFit = []

    for i in range(0, populationSize):
        #print ("i", i, "shape popArray",popArray[i].shape[0])
        fitness = calculateFitness(popArray[i], distancesMatrix)
        popFit = np.append(popFit, fitness)
    
    #print ("exit calcpopfit")    
    return popFit
# ----------------------------------------------------------------------------
    
#    ------------------ ROUND 1 
#    I randomly select two parents from the array containing the population
#    fitness. This is done by randomly selecting two individuals and then 
#    choosing the one with the lowest fitness (best). The parameter 
#    populationSize is only in there because it's needed for Round 2 (below.)
        
#def randomSelection(populationSize, popArray, distancesMatrix):
#    
#    select1 = popArray[np.random.randint(popArray.shape[0])]
#    select2 = popArray[np.random.randint(popArray.shape[0])]
#    
#    choice1 = calculateFitness(select1, distancesMatrix) 
#    choice2 = calculateFitness(select2, distancesMatrix)   
#
#    if choice1 > choice2:
#        return select2
#    else:
#        return select1
# ----------------------------------------------------------------------------

#    ------------------ ROUND 2
#    Again, I am creating two parents. However, instead of selecting from two
#    individuals, I select from 20% (50).

#def randomSelection(populationSize, popArray, distancesMatrix):
#    
#    twentyp = int(0.2*populationSize)
#    
#    select = popArray[np.random.randint(popArray.shape[0], size = twentyp)]
#    
#    fit1 = calculateFitness(select[0], distancesMatrix)
#    indiv = select[0]
#    
#    for i in range(1,select.shape[0]):
#        
#        fit2 = calculateFitness(select[i], distancesMatrix)
#        
#        if fit1 > fit2:
#            indiv = select[i]
#        
#        fit1 = calculateFitness(indiv, distancesMatrix)
#        
#    return indiv
# ----------------------------------------------------------------------------

#    ------------------ ROUND 3
#    Again, I am creating two parents. However, this time I am selecting from
#    10% of the population.

def randomSelection(populationSize, popArray, distancesMatrix):
    
    tenp = int(0.1*populationSize)
    
    select = popArray[np.random.randint(popArray.shape[0], size = tenp)]
    
    fit1 = calculateFitness(select[0], distancesMatrix)
    indiv = select[0]
    
    for i in range(1,select.shape[0]):
        
        fit2 = calculateFitness(select[i], distancesMatrix)
        
        if fit1 > fit2:
            indiv = select[i]
        
        fit1 = calculateFitness(indiv, distancesMatrix)
        
    return indiv
# ----------------------------------------------------------------------------

#    This function returns the best individual out of a given population array

def bestIndividual(popArray, distancesMatrix):
            
    fit1 = calculateFitness(popArray[0], distancesMatrix)
    indiv = popArray[0]
    
    for i in range(1,popArray.shape[0]):
        
        fit2 = calculateFitness(popArray[i], distancesMatrix)
        
        if fit1 > fit2:
            indiv = popArray[i]
        
        fit1 = calculateFitness(indiv, distancesMatrix)
        
    return indiv
# ----------------------------------------------------------------------------

#    ------------------ CROSSOVER 1
#    A child is chosen by first selecting a random number. The first bit of the 
#    child array is the first cities up to that random number. The rest are the 
#    unused cities from parentB.

def performCrossOver1(parentA, parentB, distancesMatrix):

    num = np.random.randint(len(parentA))
    
    child = parentA[0:num]

    for i in range(parentB.shape[0]):

        if i not in child:
           child = np.append(child,i)

    return child
# ----------------------------------------------------------------------------

#    ------------------ CROSSOVER 2
#    Performing crossover again, but this time, a segment is chosen in the 
#    middle rather than in the beginning or end. Code adapted from
#    https://github.com/DEAP/deap/blob/master/deap/tools/crossover.py

#def performCrossOver1(parentA, parentB, distancesMatrix):
#
#    num = np.random.randint(len(parentA))
#    
#    child1 = parentA[0:num]
#    child2 = parentB[0:num]
#
#    for i in range(parentB.shape[0]):
#
#        if i not in child1:
#           child1 = np.append(child1,i)
#    
#    for i in range(parentB.shape[0]):
#
#        if i not in child2:
#           child2 = np.append(child2,i)
#
#    child1Fit = calculateFitness(child1, distancesMatrix)
#    child2Fit = calculateFitness(child2, distancesMatrix)
#    
#    if child1Fit < child2Fit:
#        return child1
#    else:
#        return child2

# ----------------------------------------------------------------------------
    
#    ------------------ MUTATION 1
#    The mutate function takes two random indices in the array and swaps them.

#def mutate(child):
#
#    a = np.random.randint(len(child))
#    b = np.random.randint(len(child))
#    
#    mutation = np.copy(child)
#    
#    mutation[a], mutation[b] = mutation[b], mutation[a]
#    
#    return mutation
# ----------------------------------------------------------------------------

#    ------------------ MUTATION 2
#    The mutate function takes a random selection and scrambles it. As shown, 
#    it performs much worse than mutation 1

#def mutate(child):
#
#    # select random index
#    a = np.random.randint(len(child))
#    
#    mutation = np.copy(child)
#    
#    # choose the selection 0-random index or random index-14
#    if a > 7:
#        segment = mutation[a:15]
#        segment2 = mutation[0:a]
#        np.random.shuffle(segment)
#        mutation = np.append(segment2,segment)
#    else:
#        segment = mutation[0:a]
#        segment2 = mutation[a:15]
#        np.random.shuffle(segment)
#        np.array(segment)
#        mutation = np.append(segment,segment2)    
#    
#    return mutation
# ----------------------------------------------------------------------------

#    ------------------ MUTATION 3
#    The mutate function takes a random selection and reverses it.

def mutate(child):

    # select random index
    a = np.random.randint(len(child))
    
    mutation = np.copy(child)
    
    length = len(child)/2+1
    if a > length:
        segment = mutation[a:child.shape[0]]
        segment2 = mutation[0:a]
        segmentR = segment[::-1]
        mutation = np.append(segment2,segmentR)
    else:
        segment = mutation[0:a]
        segment2 = mutation[a:child.shape[0]]
        segmentR = segment[::-1]
        mutation = np.append(segmentR,segment2)    
    
    return mutation
# ----------------------------------------------------------------------------

#    Main function with two for loops to iterate through the population and the
#    number of desired generations

def main():
    
    populationSize = 250
    #populationSize = 350
    numberOfGenerations = 50
    #numberOfGenerations = 100
    
    # Read distances matrix from a file
    distancesMatrix = np.genfromtxt("Cities/15City.txt")
    #distancesMatrix = np.genfromtxt("Cities/50City.txt")

    # Calculate the number of cities
    numCities = distancesMatrix.shape[0]

    popArray = createPopulation(populationSize, numCities)
    
    # create new lists for the mean and minimum of each generation's fitness level
    meanFitness = list()
    minFitness = list()
            
    # Iterate for a fixed number of generations
    for num in range(numberOfGenerations):
        
        # calculate fitness for all individuals in the current population
        # return a 1D array containing fitness of each individual
#        popFitArray = calculatePopulationFitness(populationSize, popArray, distancesMatrix)
        
        # create a new population for the current generation
        newPopulation = np.zeros((populationSize, numCities), dtype = int)
        
        #mutationFitnessArray = np.zeros((populationSize,1) dtype = int)
            
        for i in range(populationSize):
            #print(range(populationSize))

            # Random selection to select parents
            parentA = randomSelection(populationSize, popArray, distancesMatrix)
            parentB = randomSelection(populationSize, popArray, distancesMatrix)

            # Cross over parents
            child = performCrossOver1(parentA, parentB, distancesMatrix)
            
            # Mutation 
            mutation = mutate(child)
            
            # Add the child to the new population
            newPopulation[i] = mutation 
            #newPopulation[i] = child

        # Set the new population equal to the current population
        popArray = np.copy(newPopulation) 
        
        # calculate the mean and minimum fitnesses for the population
        mean = np.mean(calculatePopulationFitness(populationSize, popArray, distancesMatrix))
        minimum = np.amin(calculatePopulationFitness(populationSize, popArray, distancesMatrix))   
        
        # append them onto the lists
        meanFitness.append(mean)
        minFitness.append(minimum)
        
    # turn those lists into arrays
    meanFit = np.array(meanFitness)
    minFit = np.array(minFitness)
        
    # plot the array
    plt.plot(meanFit)
    plt.plot(minFit)
    plt.legend(['mean', 'min'], loc = 0)
    plt.ylabel("Fitness")
    plt.xlabel("Generations")
    plt.show()
    
    # get the best individual
    best = bestIndividual(popArray, distancesMatrix)
    bestFitness = calculateFitness(best, distancesMatrix)
    print("best individual is", best, bestFitness)

main()
    
    
    