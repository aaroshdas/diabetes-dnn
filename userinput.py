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
        print(As[len(w)-1][0, 0])
    return correct/len(inputs)


with open("weights_and_biases.pkl", "rb") as f:
    w1,b1 = pickle.load(f)
    

tests = []
userInp = []
print("Format: Diabetes_binary,HighBP,HighChol,CholCheck,BMI,Smoker,Stroke,HeartDiseaseorAttack,PhysActivity,Fruits,Veggies,HvyAlcoholConsump,AnyHealthcare,NoDocbcCost,GenHlth,MentHlth,PhysHlth,DiffWalk,Sex,Age,Education,Income")
print("")
userInfo = input("User info? ")
userInp=  userInfo.split(",")

print("")
print("inputted user info:")
print(userInp)
input("continue? ")

newLine = [int(float(x)) for x in userInp]
inpVals = np.zeros((len(userInp)-1, 1))
for i in range(1, len(newLine)):
    inpVals[i-1, 0] = int(float(newLine[i]))    
tests.append(((inpVals, np.array([[newLine[0]]]))))

print(userInp)



print("")
testRes = test_network(tests, w1, b1, sigmoid)
print(str(testRes *100) + "% with " + str(int(len(tests) * testRes)) + " correct")
print("")

print("run complete")


