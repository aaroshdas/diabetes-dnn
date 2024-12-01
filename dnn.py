import numpy as np
import pickle
from random import shuffle




def sigmoid(num):
    return(1/(1+np.exp(-num)))
sigmoid_vectorized = np.vectorize(sigmoid)


def activationFuncDerivative(activationFunc,x):
    return activationFunc(x)*(1-activationFunc(x))


def test_network(inputs, w, b, activationFunc):
    correct = 0
    lowerCorrect = 0
    numOfZeros = 0
    for inp in inputs: #inputs (0,1 stuff like that)
        As ={}
        As[0] = inp[0]
        dots = {}        
        for layer in range(1, len(w)): #get layer
            dots[layer] = (w[layer]@As[layer-1])+b[layer]
            As[layer] = activationFunc(dots[layer])
        if(correct <10):
            print(As[len(w)-1][0, 0])
        # if((As[len(w)-1][0, 0] >= 0.2 and inp[1] == 1) or (As[len(w)-1][0, 0] < 0.2 and inp[1] == 0)):
        #     
        if((As[len(w)-1][0, 0] >= 0.5 and inp[1] == 1) or (As[len(w)-1][0, 0] < 0.5 and inp[1] == 0)):
            correct+=1
            if(inp[1] == 1):
                lowerCorrect+=1
        if(inp[1] == 0):
            numOfZeros +=1
    print(str(numOfZeros*100/len(inputs)) +"% non diabetic")
    print(str(lowerCorrect*100/len(inputs)) + "% (1) diabetic correct")
    return correct/len(inputs)
    
def back_propagation(inputs, w, b, activationFunc, learningRate, epochs):
    for epoch in range(epochs):     
        for ind, inp in enumerate(inputs): #inputs (0,1 stuff like that)
            if(ind%50000 == 0):
                print(ind)
            As ={}
            As[0] = inp[0]
            dots = {}        
            for layer in range(1, len(w)): #get layer
                dots[layer] = (w[layer]@As[layer-1])+b[layer]
                As[layer] = activationFunc(dots[layer])
            deltas= {}
            deltas[len(w)-1] = activationFuncDerivative(activationFunc, dots[len(w)-1]) * (inp[1]-As[len(w)-1])
            
            for layer in range(len(w)-2, 0,-1):
                deltas[layer] = activationFuncDerivative(activationFunc, dots[layer]) *(np.transpose(w[layer+1])@deltas[layer+1])
            for layer in range(1, len(w)):
                b[layer] = b[layer]+learningRate*deltas[layer]
                w[layer] = w[layer]+learningRate*deltas[layer] *np.transpose(As[layer-1])
    return (w,b)



def create_rand_values(dimensions):
    weights= [None]
    biases = [None]
    for i in range(1,len(dimensions)):
        weights.append(2*np.random.rand(dimensions[i],dimensions[i-1]) - 1)
        biases.append(2*np.random.rand(dimensions[i],1)-1)
    return weights, biases


inputSet = []
testSet = []
num = 0
with open("diabetes_dataset_02.csv") as f:
    for line in f:
        if(num !=0):
            splitLine = line.split()
            commaSplitLine = splitLine[0].split(",")
            
            newLine = [int(float(x)) for x in commaSplitLine]
            inpVals = np.zeros((len(commaSplitLine)-1, 1))
            for i in range(1, len(newLine)):
                inpVals[i-1, 0] = int(float(newLine[i]))    
            if(num > 50000):
                inputSet.append(((inpVals, np.array([[newLine[0]]]))))
            else:
                testSet.append(((inpVals, np.array([[newLine[0]]]))))
        num += 1

equalInputSet = []
numOfNonDiabetics = 0
for i in inputSet:
    if(i[1] == 1):
        equalInputSet.append(i)
    elif(i[1]== 0 and numOfNonDiabetics < 155462):
        equalInputSet.append(i)
        numOfNonDiabetics +=1
    
w1,b1 = create_rand_values([21,100, 10, 1])

with open("weights_and_biases.pkl", "rb") as f:
    w1,b1 = pickle.load(f)
    

inputs = inputSet
tests = testSet



print("test set size " + str(len(tests)))
print("input set size " + str(len(inputs)))
print("")
testRes = test_network(tests, w1, b1, sigmoid)
print(str(testRes *100) + "% with " + str(int(len(tests) * testRes)) + " correct")
print("")

for i in range(4):
    w1, b1 = back_propagation(inputs, w1, b1, sigmoid, 0.01, 1)

    with open("weights_and_biases.pkl", "wb") as f:
        pickle.dump((w1,b1), f)
    
    print("run number " + str(i+1))
    testRes = test_network(tests, w1, b1, sigmoid)
    print(str(testRes *100) + "% with " + str(int(len(tests) * testRes)) + " correct")
    print("")
print("run complete")


