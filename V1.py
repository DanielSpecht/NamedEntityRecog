# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense
import keras
import numpy


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

EspTestA = numpy.genfromtxt("esp.testa",dtype='str')
EspTestB = numpy.genfromtxt("esp.testb",dtype='str')
EspTrain = numpy.genfromtxt("esp.train",dtype='str')

NedTestA = numpy.genfromtxt("ned.testa",dtype='str')
NedTestB = numpy.genfromtxt("ned.testb",dtype='str')
NedTrain = numpy.genfromtxt("ned.train",dtype='str')

#output
Y = EspTestA[:,1]
#input
X = EspTestA[:,0:1]
