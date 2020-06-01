from __future__ import print_function
import os
import sys
from statistics import stdev, mean
import pandas as pd
import tensorflow
from matplotlib import pyplot as plt
from matplotlib import pyplot
import numpy as np
from keras import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Conv1D, MaxPooling1D, Embedding, Flatten, Dense, Dropout,GaussianNoise, GlobalMaxPooling1D
from keras.initializers import Constant

BASE_DIR = ''
GLOVE_DIR = os.path.join(BASE_DIR, 'WordEmbedding')
# TEXT_DATA_DIR = os.path.join(BASE_DIR, 'Authors')
TEXT_DATA_DIR = os.path.join(BASE_DIR, 'authors1')
MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.25
########################################################################################################################
#                                                LOAD WORD EMBEDDING                                                   #
########################################################################################################################
# first, build index mapping words in the embeddings set
# to their embedding vector
print('Indexing word vectors.')
embeddings_index = {}
with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'),encoding="utf8") as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, 'f', sep=' ')
        embeddings_index[word] = coefs
print('Found %s word vectors.' % len(embeddings_index))

########################################################################################################################
#                                                 PREPARE DATA SET                                                     #
########################################################################################################################
# second, prepare text samples and their labels
print('Processing text dataset')
texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids
for name in sorted(os.listdir(TEXT_DATA_DIR)):
    path = os.path.join(TEXT_DATA_DIR, name)
    if os.path.isdir(path):
        label_id = len(labels_index)
        labels_index[name] = label_id
        for fname in sorted(os.listdir(path)):
            if fname.isdigit():
                fpath = os.path.join(path, fname)
                args = {} if sys.version_info < (3,) else {'encoding': 'latin-1'}
                with open(fpath, **args) as f:
                    t = f.read()
                    i = t.find('\n\n')  # skip header
                    if 0 < i:
                        t = t[i:]
                    texts.append(t)
                labels.append(label_id)
print('Found %s texts.' % len(texts))
########################################################################################################################
#                                                PREPARE CNN INPUT                                                     #
########################################################################################################################
# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]

print('Preparing embedding matrix.')

# prepare embedding matrix
num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
print(embedding_matrix[1])
np.savetxt("word.csv", embedding_matrix[1], delimiter=",")
###################################
arr2 = np.empty((0,10000), float)
arr4 = np.empty((0,10000), float)
arr6 = np.empty((0,10000), float)
noiseArr = np.empty((0,1), float)

###################################F
########################################################################################################################
#                                                      CNN MODEL                                                       #
########################################################################################################################
# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
noise = 0
for x in range(21):
    print(noise)
    # noiseArr = np.append(arr, np.array(noise), axis=0)
    print('Training model.')
    model = Sequential()
    model.add(Embedding(num_words,EMBEDDING_DIM,embeddings_initializer=Constant(embedding_matrix),input_length=MAX_SEQUENCE_LENGTH, trainable=False))
    model.add(GaussianNoise(noise, input_shape=(MAX_SEQUENCE_LENGTH,)))
    model.add(Conv1D(filters=100, kernel_size=3, padding='valid', activation='relu', input_shape=(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Conv1D(filters=100, kernel_size=4, padding='valid', activation='relu', input_shape=(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Conv1D(filters=100, kernel_size=5, padding='valid', activation='relu', input_shape=(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Flatten())
    model.add(Dense(units=256, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(2, activation='softmax', name='output'))
    model.compile(loss=tensorflow.keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
    model.summary()
    # with open('conv3Filter.csv','a') as f:
    #     np.savetxt(f, np.ravel(conv3Filter[0][0]), delimiter="\t", newline='\t')

    # history = model.fit(x_train, y_train, validation_split=0.25, epochs=10, verbose=2)
    conv2Filter = model.layers[2].get_weights()
    conv4Filter = model.layers[4].get_weights()
    conv6Filter = model.layers[6].get_weights()
    history = model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_val, y_val))

    ########################################################################################################################
    #                                                     SAVE WEIGHTS                                                     #
    ########################################################################################################################
    conv2Filter = model.layers[2].get_weights()
    conv4Filter = model.layers[4].get_weights()
    conv6Filter = model.layers[6].get_weights()
    arr2 = np.append(arr2, np.array([conv2Filter[0][0].flatten()]), axis=0)
    arr4 = np.append(arr4, np.array([conv4Filter[0][0].flatten()]), axis=0)
    arr6 = np.append(arr6, np.array([conv6Filter[0][0].flatten()]), axis=0)
    noise = np.random.normal(0.02, 0.1,1)

    ############################################
arr2T = arr2.transpose()
arr4T = arr4.transpose()
arr6T = arr6.transpose()
covArr2 = np.empty((0,10000), float)
covArr4 = np.empty((0,10000), float)
covArr6 = np.empty((0,10000), float)
for i, x in enumerate(arr2T):
    std2 = stdev(arr2T[i])
    mean2 = mean(arr2T[i])
    abs2 = abs(mean2)
    cov2 = std2/abs2
    covArr2 = np.append(covArr2, cov2)
    arr2 = np.append(arr2, covArr2)
    #####################
    std4 = stdev(arr4T[i])
    mean4 = mean(arr4T[i])
    abs4 = abs(mean4)
    cov4 = std4/abs4
    covArr4 = np.append(covArr4, cov4)
    arr4 = np.append(arr4, covArr4)
    #####################
    std6 = stdev(arr6T[i])
    mean6 = mean(arr6T[i])
    abs6 = abs(mean6)
    cov6 = std6/abs6
    covArr6 = np.append(covArr6, cov6)
    arr6 = np.append(arr6, covArr6)
    #####################
df2 = pd.DataFrame(covArr2, columns=['2'])
df2q = df2.quantile(.75)

df4 = pd.DataFrame(covArr4, columns=['4'])
df4q = df4.quantile(.75)

df6 = pd.DataFrame(covArr6, columns=['6'])
df6q = df6.quantile(.75)
    #####################
plt.plot(covArr2)
plt.plot(covArr4)
plt.plot(covArr6)
plt.title('Covariance')
plt.savefig('a.jpeg', bbox_inches='tight', pad_inches=0)
plt.show()
plt.savefig('1_2_plot.png', dpi=300)

with open('1_2_conv2_quantile.csv','a') as f:
    np.savetxt(f, df2q)
with open('1_2_conv4_quantile.csv','a') as f:
    np.savetxt(f, df4q)
with open('1_2_conv6_quantile.csv','a') as f:
    np.savetxt(f, df4q)
    ############################################
with open('1_2_conv2.csv','a') as f:
    np.savetxt(f, arr2, delimiter=",")
with open('1_2_conv4.csv','a') as f:
    np.savetxt(f, arr4, delimiter=",")
with open('1_2_conv6.csv','a') as f:
    np.savetxt(f, arr6, delimiter=",")
    ############################################

########################################################################################################################
#                                                   RESULTS PLOTTING                                                   #
########################################################################################################################
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

model.save("1_2_Model.h5")
