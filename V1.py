# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.utils import np_utils

import keras
import numpy

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)


def getFileLineArray2D( filePath ):
    return numpy.genfromtxt(filePath,dtype='str')

def getWordArray1D( fileLineArray ):
    return fileLineArray[:,0:1].ravel()

def getClassificationArray1D( fileLineArray ):
    return fileLineArray[:,1].ravel()

def chunks(l, n):
    for i in xrange(0, len(l), n):
        yield l[i:i + n]
        

languages = ["esp","ned"]

arquivos = ["FileTestA","FileTestB","FileTrain"]

#dicionário indexando os arquivos dos idiomas
languageFileDictionary ={
    'ned':{"FileTestA":"ned.testa","FileTestB":"ned.testb","FileTrain":"ned.train"},
    'esp':{"FileTestA":"esp.testa","FileTestB":"esp.testb","FileTrain":"esp.train"}
    }

#neste exemplo obtemos o FileTestA de ned, recebemos apenas o nome do arquivo
print languageFileDictionary['ned']['FileTestA']

#Obtem o array 2d de linhas do arquivo
#print getFileLineArray2D(languageFileDictionary['ned']['FileTestA'])


for language in languages:
    print "-------"+language+"-------"
    
    tokenizer = Tokenizer()

    fileTestA2D = getFileLineArray2D(languageFileDictionary[language]['FileTestA']);
    fileWords1D = getWordArray1D(fileTestA2D)
    fileClassification1D = getClassificationArray1D(fileTestA2D)

    print fileWords1D

    tokenizer.fit_on_texts(fileWords1D) # turn to 1D array
    print('Encontrou %s palavras.' % len(tokenizer.word_index))

    sequences2D = tokenizer.texts_to_sequences(fileWords1D)
    
    print sequences2D[0:10]
    # remove todas as ocorrências de listas vazias que correspondem a pontuações e caracteres ignorados (util?)
    while list() in sequences2D: sequences2D.remove(list()) 
    print sequences2D[0:10]
    #chunking
    print list(chunks(numpy.squeeze(numpy.asarray(sequences2D)),3))[1]

    embedding_matrix = tokenizer.texts_to_matrix(tokenizer.word_index)
    print type(tokenizer.texts_to_matrix(tokenizer.word_index))
