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
from keras.layers import Conv1D, MaxPooling1D, Embedding
from sklearn.metrics import (precision_score, recall_score, f1_score, accuracy_score)

from keras.utils import np_utils

import keras
import numpy


# fix random seed for reproducibility
#seed = 7
#numpy.random.seed(seed)
CLASSES = ["I-MISC","B-MISC","I-ORG","B-ORG","I-LOC","B-LOC","I-PER","B-PER","O"]
UNKNOWN_WORD = "@@@"
LINE_SENTENCE_END = ""
WINDOW_PADDING = 0
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
    #return map(lambda x: x.lower(), words)
    return map(lambda x:x.lower(), words)
    inputSequences = list()
    for i in range( 0,len(inputs)):
        inputInstance = tokenizer.texts_to_sequences(inputs[i])
        inputInstance = numpy.array(inputInstance).ravel()
        inputSequences.append(inputInstance)
    return numpy.array(inputSequences)

def getTargetSequences(chunks,targetTokenizer):
    targets = numpy.array(chunks)[:,1]
    targetSequences = targetTokenizer.texts_to_sequences(targets)   
    targetSequences = numpy.array(targetSequences).flatten()
    targetSequences = numpy.array(map(lambda x: x-1, targetSequences))   
    return targetSequences

languages = ["ned","esp"]

#dicionário indexando os arquivos dos idiomas
languageFileDictionary ={
    'ned':{"development":"ned.testa","test":"ned.testb","train":"ned.train"},
    'esp':{"development":"esp.testa","test":"esp.testb","train":"esp.train"}}
exception_verbosity="high"

targetTokenizer = Tokenizer(filters="",lower=False)
targetTokenizer.fit_on_texts(CLASSES)

print targetTokenizer.word_index

for language in languages:
   
    print "-------"+language+"-------"
    print ("Teste com uma janela de tamanho:%d"%(MAX_SEQUENCE_LENGTH))
   
    fileTrain = getFileLineList(languageFileDictionary[language]['train']);

    words = getWordList(fileTrain)

    targets = getClassificationList(fileTrain)

    inputTokenizer = Tokenizer(filters ="",lower=True)
    inputTokenizer.fit_on_texts(words)
    inputTokenizer.fit_on_texts([SENTENCE_DELIMITER])
    inputTokenizer.fit_on_texts([UNKNOWN_WORD])
    print('Encontrou %s palavras.' % len(inputTokenizer.word_index))
   
    chunks = getInputChunks(words,targets,inputTokenizer)
   
    inputSequences = getInputSequences(chunks,inputTokenizer)
   
    print('Encontrou %s classes.' % len(targetTokenizer.word_index))   
   
    targetSequences = getTargetSequences(chunks,targetTokenizer)
    targetSequences = np_utils.to_categorical(targetSequences, 9)
   
   
    model = Sequential()
    model.add(Embedding(len(inputTokenizer.word_index) + 1, 500 , input_length=MAX_SEQUENCE_LENGTH))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dense(len(targetTokenizer.word_index), activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['fmeasure','precision','recall'])
   
    #################################
    #DEBUG
    #################################
    fileTest = getFileLineList(languageFileDictionary[language]['test']);
    wordsTest = getWordList(fileTest)
    targetsTest = getClassificationList(fileTest)
    chunksTest = getInputChunks(wordsTest,targetsTest,inputTokenizer)
    inputSequencesTest = getInputSequences(chunksTest,inputTokenizer)
    targetSequencesTest = getTargetSequences(chunksTest,targetTokenizer)
    targetSequencesTest = np_utils.to_categorical(targetSequencesTest, 9)

    if False:
        target = open("DEBUG-INPUT", 'w')
        janelasTreino = numpy.array(chunks)[:,0]
        clasificacoesTreino = numpy.array(chunks)[:,1]
        #Averiguar as estruturas das windows
        for i in range(0,len(janelasTreino)):
            if clasificacoesTreino[i] != 'O':
                target.write(janelasTreino[i][0]+"--"+clasificacoesTreino[i]+"--"+str(inputTokenizer.word_index[janelasTreino[i][0]])+"====>"+"("+str(inputSequences[i][0])+str(targetSequences[i])+ ")"  )
                target.write("\n")

        target.close()

        target = open("DEBUG-OUTPUT", 'w')
        janelasTeste = numpy.array(chunksTest)[:,0]
        clasificacoesTeste= numpy.array(chunksTest)[:,1]
        print clasificacoesTeste[10]
        print janelasTeste[10]

        #Averiguar as estruturas das windows DE TESTE
        for i in range(0,len(janelasTeste)):
            if clasificacoesTeste[i] != 'O':
                target.write("Chunks.Janela : "+ janelasTeste[i][0])
                target.write("\n")
                target.write("Chunks.Classificacao : "+ clasificacoesTeste[i])
                target.write("\n")
                target.write("Código no Tokenizer : "+str(inputTokenizer.word_index[janelasTeste[i][0]]))
                target.write("\n")
                target.write("Input na ANN : "+str(inputSequencesTest[i][0]))
                target.write("\n")
                target.write("Output na ANN : "+str(targetSequencesTest[i]))
                target.write("\n")
                #target.write(janelasTeste[i][0]+"--"+clasificacoesTeste[i]+"--"+str(inputTokenizer.word_index[janelasTeste[i][0]])+"====>"+"("+str(inputSequencesTest[i][0])+str(targetSequencesTest[i])+ ")"  )
                target.write("\n")
        target.close()
        model.summary()

   
   

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = Embedding(len(inputTokenizer.word_index) + 1, 500, input_length=MAX_SEQUENCE_LENGTH)(sequence_input)
    x = Flatten()(embedded_sequences)
    x = Dense(500, activation='relu')(x)
    preds = Dense(len(targetTokenizer.word_index), activation='softmax')(x)
    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['fmeasure','precision','recall'])
    model.summary()

    print "Tipo array input : "+str(type(inputSequences))
    print "Shape array input : "+str(inputSequences.shape)

    print "Tipo array classes : "+str(type(targetSequences))
    print "Shape array classes : "+str(targetSequences.shape)

    model.fit(inputSequences[:len(inputSequences)/3], targetSequences[:len(targetSequences)/3], batch_size=1024, nb_epoch=1, verbose=1, validation_data=(inputSequencesTest , targetSequencesTest), shuffle=True)       
