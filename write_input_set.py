import numpy as np
num = 0
inputs=[]
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
                inputs.append(((inpVals, np.array([[newLine[0]]]))))
        num += 1

diabetic =0
for i in inputs:
    if(i[1] == 0):
        diabetic +=1

print(diabetic)
print(len(inputs)-diabetic)
print(diabetic*100/len(inputs))