# Create first network with Keras
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.layers import Embedding
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Flatten

from sklearn.metrics import (precision_score, recall_score, f1_score, accuracy_score)

from keras.utils import np_utils

import keras
import numpy


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

LINE_SENTENCE_END = ""
WINDOW_PADDING = 4
SENTENCE_DELIMITER = "***"
MAX_SEQUENCE_LENGTH = 2*WINDOW_PADDING+1

#IGNORABLE_WORDS = ["","-DOCSTART-"]

def getFileLineList( filePath ):
    return open(filePath,'r').read().splitlines()

def getWordList( fileLineArray ):
    words = list()
    for i, s in enumerate(fileLineArray):
        # pega a primeira palavra
        if(len(fileLineArray[i].split())>1):
            words.append(fileLineArray[i].split()[0])
        else:
            words.append(LINE_SENTENCE_END)
    return words

def getClassificationList( fileLineArray ):
    classifications = list()
    for i, s in enumerate(fileLineArray):
        # pega ultima palavra
        if(len(fileLineArray[i].rsplit(' ', 1))>1):
            classifications.append(fileLineArray[i].rsplit(' ', 1)[1])
        else:
            classifications.append(LINE_SENTENCE_END)
    return classifications

def getInputChunks(wordList,clasificationsList):
    chunks = list()
    for i in range(0,len(wordList)):
        chunk = list()
        
        if wordList[i] != LINE_SENTENCE_END:
            chunk.append(wordList[i])
            for leftIndex in range(i-1,i-WINDOW_PADDING-1,-1):
                if leftIndex < 0:
                    chunk.insert(0,SENTENCE_DELIMITER)
                elif wordList[leftIndex]==LINE_SENTENCE_END:
                    for _ in range(leftIndex,i-WINDOW_PADDING-1,-1):
                        chunk.insert(0,SENTENCE_DELIMITER)
                    break
                else:
                    chunk.insert(0,wordList[leftIndex])

            for rightIndex in range(i+1,i+WINDOW_PADDING+1):
                if  rightIndex>len(wordList)-1 :
                    chunk.append(SENTENCE_DELIMITER)
                elif wordList[rightIndex]==LINE_SENTENCE_END:
                    for _ in range(rightIndex,i+WINDOW_PADDING+1):
                        chunk.append(SENTENCE_DELIMITER)
                    break
                else:
                    chunk.append(wordList[rightIndex])
            chunks.append([chunk,clasificationsList[i]])
            
    return chunks

def getInputSequences(chunks,tokenizer):
    
    inputs = numpy.array(chunks)[:,0]
    
    inputSequences = list()
    for i in range( 0,len(inputs)):
        inputSequences.append(numpy.array(tokenizer.texts_to_sequences(inputs[i])).ravel().tolist())

    return numpy.array(inputSequences)

def getTargetSequences(chunks,targetTokenizer):
    targets = numpy.array(chunks)[:,1]
    targetSequences = targetTokenizer.texts_to_sequences(targets)    
    targetSequences = numpy.array(targetSequences).flatten()
    targetSequences = numpy.array(map(lambda x: x-1, targetSequences))
    
    return targetSequences

languages = ["esp","ned"]

#dicionário indexando os arquivos dos idiomas
languageFileDictionary ={
    'ned':{"development":"ned.testa","test":"ned.testb","train":"ned.train"},
    'esp':{"development":"esp.testa","test":"esp.testb","train":"esp.train"}}

for language in languages:
    
    print "-------"+language+"-------"
    print ("Teste com uma janela de tamanho:%d"%(MAX_SEQUENCE_LENGTH))
    
    fileTrain = getFileLineList(languageFileDictionary[language]['train']);
    
    words = getWordList(fileTrain)
    targets = getClassificationList(fileTrain)
    
    inputTokenizer = Tokenizer(filters ="")
    inputTokenizer.fit_on_texts(words)
    inputTokenizer.fit_on_texts([SENTENCE_DELIMITER])
    print('Encontrou %s palavras.' % len(inputTokenizer.word_index))
    
    chunks = getInputChunks(words,targets)
    
    inputSequences = getInputSequences(chunks,inputTokenizer)
    
    targetTokenizer = Tokenizer(filters="")
    targetTokenizer.fit_on_texts(numpy.array(targets))
    print('Encontrou %s classes.' % len(targetTokenizer.word_index))    
    
    targetSequences = getTargetSequences(chunks,targetTokenizer)
    
    model = Sequential()
    model.add(Embedding(len(inputTokenizer.word_index) + 1, 500, input_length=MAX_SEQUENCE_LENGTH))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dense(len(targetTokenizer.word_index), activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    
    #TREINAMENTO
    
    model.fit(inputSequences, targetSequences, nb_epoch=3, batch_size=10)
    
    #TESTE
    
    fileTest = getFileLineList(languageFileDictionary[language]['test']);
    
    wordsTest = getWordList(fileTest)
    targetsTest = getClassificationList(fileTest)
    
    chunksTest = getInputChunks(words,targets)

    targetSequencesTest = getTargetSequences(chunks,targetTokenizer)
    inputSequencesTest = getInputSequences(chunks,inputTokenizer)
    
    #model.fit(inputSequences, targetSequences,validation_data=(inputSequencesTest,inputSequencesTest) ,nb_epoch=1, batch_size=10)
    
    
    print ">>TESTE<<"
    scores = model.evaluate(targetSequencesTest, inputSequencesTest)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        
    
