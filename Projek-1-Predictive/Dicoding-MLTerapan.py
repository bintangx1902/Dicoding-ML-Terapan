#%%
import opendatasets as od
import os, re, tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
import contractions
import seaborn as sns

from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
#%%
# downloading the dataset using opendatasets  
od.download('https://www.kaggle.com/datasets/ramjasmaurya/poem-classification-nlp')
#%%
# reading the data with pd.read_csv, there are only 2 data : train and test

train = pd.read_csv('poem-classification-nlp/Poem_classification - train_data.csv')
test = pd.read_csv('poem-classification-nlp/Poem_classification - test_data.csv')
#%%
# show the upper 5 data from train data
train.head()
#%%
# find the label / genre inside data Poem dataset from train.
class_name = train.Genre.unique()
class_name
#%%
# Check the data shape 
print("Train Shape", train.shape)
print("Test Shape", test.shape)
#%%
# check is there a null data in train dataset
print("Train")
train.isnull().sum()
#%%
# check is there a null data in test dataset
print("Test")
test.isnull().sum()
#%% md
# # Before Drop NaN Column
#%%
# check the label(genre) distribution
train['Genre'].value_counts()
#%%
# Plot the label(genre)
train.groupby('Genre').size().plot(kind='bar') 
#%% md
# # After drop NaN Column
#%%
# drop the NaN or null using .dropna() method
train = train.dropna()
#%%
# Check the new shape after dropping 
train.shape
#%%
# counting the genre distribution after dropping missing value
train['Genre'].value_counts()
#%%
# Plot the label(genre)
train.groupby('Genre').size().plot(kind='bar') 
#%% md
# # Test dataset
#%%
# check the label(genre) distribution
test['Genre'].value_counts()
#%%
# Plot
test.groupby('Genre').size().plot(kind='bar')
#%% md
# ## Sample Text
#%%
# make a random integer in range 0 until 150
n = np.random.randint(0, 150)
n
#%%
# Get sample from the random number
sample_train = train['Poem'][n]
sample_test = test['Poem'][n]
#%%
print("train sample txt:", sample_train)
print("test sample txt:", sample_test)
#%% md
# # Text preprocessing
# - lowercase 
# - remove number and punctuation 
# - remove stopwords 
# - Lemmatize 
#%%
# get the clean text function from utils.py 
from utils import clean_text
#%%
# Apply the preprocessing function from utils
train["clean_text"] = train["Poem"].apply(clean_text)
test["clean_text"] = test["Poem"].apply(clean_text)
#%%
# show first 50 data after preprocessing
train.head(50)
#%%
# use the label encoder form scikit learn to transform text label to numeric based on the alphabet order
le = LabelEncoder()
le.fit(class_name)
#%%
# transform all the labels 
train['label'] = le.transform(train.Genre)
test['label'] = le.transform(test.Genre)
#%%
# Check the label mapping
label_mapping = {index: label for index, label in enumerate(le.classes_)}
label_mapping
#%%
# show the first 5 data in train 
train.head(5)
#%%
# grab the clean_text and label for the feature and label 
x_train = train['clean_text']
x_test = test['clean_text']

y_train = train['label']
y_test = test['label']
#%%
# delete the old train and test, because it is no longer used after Tokenization
del train, test
#%% md
# # Tokenization
#%%
# import the tensorflow library to tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
#%%
# create the tokenizer, num of words is 1k and set the out of vocabulary token
tokenizer = Tokenizer(num_words=1000, oov_token='<OOV>')
tokenizer.fit_on_texts(x_train)

# make the text as sequences
x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)

# the actual vocab size is the length of word index + 1
vocab_size = len(tokenizer.word_index) + 1
word_index = tokenizer.word_index
print(vocab_size)
#%%
# make a padding for the array sequences with maximum length is 200
x_train = pad_sequences(x_train, padding='post', maxlen=200, truncating='post')
x_test = pad_sequences(x_test, padding='post', maxlen=200, truncating='post')
#%%
# cast as numpy array
x_train = np.array(x_train)
x_test = np.array(x_test)
#%%
# make the label as categorical
y_train = to_categorical(y_train)
#%% md
# # Model
#%%
# Building model using Convolutional 1 Dimension with padding = causal.
model = tf.keras.Sequential([ 
    tf.keras.layers.Embedding(vocab_size, 128, input_length=200), 
    tf.keras.layers.Conv1D(64, 5, padding='causal', activation='relu'),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax'),
])

model.summary()
#%%
# plot the model as image
tf.keras.utils.plot_model(model, show_shapes=True)
#%%
tf.config.list_physical_devices()
#%%
# compile the model using Adaptive Algorithm optimizer
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=.0001),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy']
)
#%%
# train the model
H = model.fit(
    x_train, y_train,
    verbose=2,
    epochs=250,
    batch_size=128
)
#%% md
# # Evaluation
#%%
# plot the training loss and accuracy
N = len(H.history['loss']) + 1

fig, ax = plt.subplots(2, 1, figsize=(12, 10))
ax = ax.flatten()

ax[0].plot(np.arange(1, N), H.history["loss"], label="Training loss")
ax[0].title.set_text("Loss")
ax[0].legend()

ax[1].plot(np.arange(1, N), H.history["accuracy"], label="Training accuracy")
ax[1].title.set_text("Accuracy Score")
ax[1].legend()

plt.show()
#%%
# make a prediction from the test data (argmax to find the max value, it means the class with higher accuracy confidence)
predict = model.predict(x_test)
predict_class = np.argmax(predict, axis=1)
predict_class = np.array(predict_class)
predict_class
#%%
from sklearn.metrics import classification_report, confusion_matrix
#%%
# plot the confusion matrix
cm = confusion_matrix(y_test, predict_class)
disp_log = sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
disp_log = disp_log.plot(cmap=plt.cm.Blues,values_format='g')
plt.title("Confusion Matrix", pad= 20, fontsize= 20, fontweight= "bold")
plt.show()
#%%
print(classification_report(y_test, predict_class, target_names=class_name))
#%% md
# # Model
#%%
# Building model using Bidirectional Recurrent Neural Network with LSTM Layer
model_lstm = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 128, input_length=200), 
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax'),
])

model_lstm.summary()

# print the model as image
tf.keras.utils.plot_model(model_lstm, show_shapes=True)
#%%
# compile the model using Adaptive Algorithm optimizer
model_lstm.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=.0001),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy']
)
#%%
# train the model
H = model.fit(
    x_train, y_train,
    verbose=2,
    epochs=250,
    batch_size=128
)
#%% md
# # Evaluation
#%%
# plot the training accuracy and loss
N = len(H.history['loss']) + 1

fig, ax = plt.subplots(2, 1, figsize=(12, 10))
ax = ax.flatten()

ax[0].plot(np.arange(1, N), H.history["loss"], label="Training loss")
ax[0].title.set_text("Loss")
ax[0].legend()

ax[1].plot(np.arange(1, N), H.history["accuracy"], label="Training accuracy")
ax[1].title.set_text("Accuracy Score")
ax[1].legend()

plt.show()
#%%
# make a prediction using the test data
predict = model_lstm.predict(x_test)
predict_class = np.argmax(predict, axis=1)
predict_class = np.array(predict_class)
predict_class
#%%
# Plot confusion matrix
cm = confusion_matrix(y_test, predict_class)
disp_log = sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
disp_log = disp_log.plot(cmap=plt.cm.Blues,values_format='g')
plt.title("Confusion Matrix", pad= 20, fontsize= 20, fontweight= "bold")
plt.show()
#%%
print(classification_report(y_test, predict_class, target_names=class_name))
#%%
