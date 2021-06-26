# This code is part of the tumblr blog post series: Research Progression
# Link to blog post: https://thenerdyexplorer.tumblr.com/post/655047303580811264/research-progression

# The code below creates a synthetic dataset based on an ideal function which can be used as a sample
# to observe the behaviour of an ML model for a simplistic dataset

import numpy as np
import pandas as pd

def saveCSV(DF, inpDiscrete, numberOfClasses, fileName):
	if np.any(inpDiscrete):
		for i in range(len(inpDiscrete)):
			if inpDiscrete[i] == True:
				origColVals = np.asarray(DF[i])
				newColVals = []
				numClasses = numberOfClasses[i]
				origColVals *= numClasses
				origColVals = [int(j) for j in origColVals]
				newColVals = ["Class-{}".format(j+1) for j in origColVals]
				DF[i] = newColVals
			else:
				pass

	DF.to_csv(fileName)


def customFunction(numInputVariables, inputValues, numSamples):
	# Simple weighted sum function used: i.e. y = a1*x1 + a2*x2 + ... an*xn
	# where a1, a2, a3... are randomly sampled weights used for all samples in the dataset
	# x1, x2, x3... are input attributes and y is the target value for a single sample
	weights = (np.random.random_sample((numInputVariables,)) - 0.5) *2
	targetMatrix  = inputValues* weights
	yValues = np.reshape(np.sum(targetMatrix, axis = 1), (numSamples,1))

	# semiFinalDB is the database which is the combination of [x1, x2, x3...xn, y]
	semiFinalDB = np.hstack((inputValues,yValues))

	return semiFinalDB

def createDB(numInputVariables, numSamples, inpDiscrete, numberOfClasses, noise = 0, fileName = "sampleDB.csv"):
	# Variables are normalized float values between 0 to 1
	inputValues = np.random.random_sample((numSamples, numInputVariables))

	# Using a simple weighted sum as the target function 
	# edit the function: "customFunction" to create your own custom function
	semiFinalDB = customFunction(numInputVariables, inputValues, numSamples)

	# introduce noise
	noiseValue = np.random.random_sample((numSamples, numInputVariables + 1))
	noiseValue[:,-1] = 0
	noiseValue *= noise
	semiFinalDB += noiseValue

	# writing into csv
	DF = pd.DataFrame(semiFinalDB)
	  
	# save the dataframe as a csv file
	saveCSV(DF, inpDiscrete, numberOfClasses, fileName)

print("Input variable values are normalised (taken between 0 to 1)!!\n\n")

numInputVariables = int(input("How many input variables do you want? (Enter a natural number)\t"))

noise = input("\n\nDo you want to add noise? y/n\t")
if noise == "y":
	print("\nRandom noise will be added in the range of 0 to upper limit\n")
	noise = float(input("What should be the upper limit of noise? (Enter float value from 0 to 1)\t"))
else:
	noise = 0

classChoice = input("\n\nDo you want to make some input variables discrete (having classes: 1, 2, 3...)? y/n\t")
if classChoice == "y":
	inpDiscrete = []
	numberOfClasses = []
	for i in range(numInputVariables):
		ch = input("\nIs input variable-{} a discrete value attribute? y/n\t".format(i+1))
		if ch == "y":
			inpDiscrete.append(True)
			numClasses = int(input("Enter the number of classes: (Enter a natural number)\t"))
			numberOfClasses.append(numClasses)
		else:
			inpDiscrete.append(False)
			numberOfClasses.append(0)
else:
	inpDiscrete = [False for i in range(numInputVariables)]
	numberOfClasses = [0 for i in range(numInputVariables)]

numSamples = int(input("\n\nHow many samples do you want? (Enter a natural number)\t"))

createDB(numInputVariables, numSamples, inpDiscrete, numberOfClasses, noise)