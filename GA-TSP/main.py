from random import random,sample,randint
import numpy as np
import xml.etree.ElementTree as ET
import copy


def readData(file,dimension):
    tree = ET.parse(file)
    root = tree.getroot()
    graph = root.find('graph')

    distances=np.zeros((dimension,dimension))
    i=-1
    for vertex in graph.getiterator('vertex'):
        i+=1
        for edge in vertex.getiterator('edge'):
                   distances[i][int(edge.text)]=edge.attrib['cost']

    return distances




def generatePopulation(num,dim):
    population=[]
    for i in range(num):
        numbers=sample(range(0,dim),dim)
        population.append(numbers)
    return population


def fitness(chromosome): # calculates fitness of each chromosome of population
    cost = 0
    _, idx = np.unique(chromosome, return_index=True)
    chromosome=chromosome[np.sort(idx)]
    for i in range(len(chromosome)-1):
        cost += distances[chromosome[i]][chromosome[i + 1]]
    cost += distances[len(chromosome) - 1][0]
    return cost


def tournament(remainingPop,tSize):
    randoms=[]
    fitnessScores=[]
    for i in range(tSize):
        randoms.append(randint(0,len(remainingPop)-1))
    for i in range(len(randoms)):
        fitnessScores.append(fitness(remainingPop[randoms[i]]))
    fittestIndex=randoms[fitnessScores.index(min(fitnessScores))]
    remainingPop=np.delete(remainingPop,remainingPop[fittestIndex],axis=0)
    return fittestIndex


def pickFittest(population):
    popFitness=[]
    for chromosome in population:
        popFitness.append(fitness(chromosome))

    fittestPop=[]
    idx=np.argsort(popFitness)
    for i in range(int(len(idx)*(1-pReplace))):
        fittestPop.append(population[idx[i]])
    return fittestPop


def findInPopulation(parent,pop):
    for x in pop:
        if np.array_equal(parent,x):
            return False
    return True



def reproduction(population,crossoverProb,mutationProb,pReplace): # cycle crossover
    oldFitPop = pickFittest(population)
    remainingPop=copy.deepcopy(population)
    newPopulation=[]
    while len(newPopulation)< int(populationSize*pReplace):
        p1Index=tournament(remainingPop,2)
        p2Index=tournament(remainingPop,2)
        parent1=population[p1Index]
        parent2=population[p2Index]
        if crossoverProb>random():
            child1,child2=crossover(parent1,parent2)
            if mutationProb > random():
                child1 = mutate(child1)
                child2 = mutate(child2)
                newPopulation.append(child1)
                newPopulation.append(child2)
        else:
            if findInPopulation(parent1,oldFitPop):
                newPopulation.append(parent1)
            if len(newPopulation)<int(populationSize*pReplace):
                if findInPopulation(parent2,oldFitPop):
                    newPopulation.append(parent2)


    return newPopulation + oldFitPop





def mutate(chromosome):
    idx1=0
    idx2=0
    while idx1==idx2:
        idx1 = randint(0, len(chromosome) - 1)
        idx2 = randint(0, len(chromosome) - 1)

    minIdx=min(idx1,idx2)
    maxIdx=max(idx1,idx2)
    tempList=[chromosome[maxIdx]]
    newChromosome=np.concatenate((chromosome[:minIdx+1] , tempList, chromosome[minIdx+1:maxIdx] , chromosome[maxIdx+1:]),axis=0)

    return newChromosome



def removeFromList(lst, x):
    i = -1
    for key in lst:
        i = i + 1
        if key==x:
            lst =np.concatenate((lst[:i],lst[i + 1:]),axis=0)
            i-=1
    return lst

def crossover(parent1,parent2):
    p1 = copy.deepcopy(parent1)
    p2 = copy.deepcopy(parent2)
    child1 = copy.deepcopy(p1)
    child2 = copy.deepcopy(p2)
    cycle = 0
    while len(p1) > 0:
        stPoint = p1[0]
        i = np.where(parent1==stPoint)
        cycle += 1
        flag = True
        while flag:
            j = parent1[i]
            p1 = removeFromList(p1, parent1[i])
            k = parent2[i]
            if cycle % 2 == 0:
                child1[i] = k
                child2[i] = j
            else:
                child1[i] = j
                child2[i] = k

            i = np.where(parent1==k)

            if k == stPoint:
                flag = False

    return  child1,child2

def stoppingValidation():
    global generations,bestAnswers,bestChroms,meanAnswers,worstAnswers
    if len(generations)>0:
            population= generations[-1]
            popFitness = []
            for chromosome in population:
                popFitness.append(fitness(chromosome))
            fittestScore=min(popFitness)
            bestAnswers.append(fittestScore)
            meanAnswers.append(np.mean(popFitness))
            worstAnswers.append(max(popFitness))
            bestChroms.append(population[popFitness.index(fittestScore)])
            print("BEST ANSWER FITNESS SCORE:" +str(fittestScore))



def printResults(numGen):
    np.set_printoptions(threshold=np.nan)
    print("Generation #" +str(numGen) )
    if len(bestAnswers)>1:
        print("Best Answers Scores:")
        print(bestAnswers[-1])
        print("Mean Answers Scores:")
        print(meanAnswers[-1])
        print("Worst Answers Scores:")
        print(worstAnswers[-1])
        print("Best Chromosme:")
        print(bestChroms[-1])


dim=29
#distances=readData('ali535.xml',dim)
#distances=readData('d2103.xml',dim)
distances=readData('bayg29.xml',dim)
#distances=readData('ftv70.xml',dim)


populationSize=100
pReplace=0.8
pCrossover=0.95
pMutation=0.1
global population
population=generatePopulation(populationSize,dim)
population=np.array(population)
maxIteration=200

generations=[]
bestAnswers=[]
bestChroms=[]
meanAnswers=[]
worstAnswers=[]
for i in range(maxIteration):
    population=reproduction(population, pCrossover, pMutation, pReplace)
    generations.append(population)
    stoppingValidation()
    printResults(len(generations))

