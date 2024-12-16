dnn.py -  back_propagation function uses sigmoid as activation function on a [21, 100, 10, 1] network to calculate probability of being diabetic based on 
blood pressure, cholestrol, BMI, smoking status, ect...
<br/>

when fully trained, classifies ~88% of inputs correctly
<br/>

issues:
<br/>
~82% of the training data is non-diabetic, and thus, the network is rewarded for classifying all points as non diabetic. I tried solving this by splitting the data set 50-50,
however, there are not enough diabetic data points for the network to be succesful
