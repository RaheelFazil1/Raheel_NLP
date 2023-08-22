import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense
import matplotlib.pyplot as plt


# csv reviews file was downloaded from the link https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
# Load the CSV data
mydata = pd.read_csv(r'IMDB Dataset.csv')
# mydata = mydata[mydata['sentiment'] != 'positive']
# mydata = mydata[mydata['sentiment'] != 'negative']
# print(mydata.isnull().sum())
# print(mydata.dtypes)
# mydata.info()
# print(mydata.head())
# print(mydata)

# print(mydata['sentiment'])
# Label encoding for sentiment labels
label_encoder = LabelEncoder()
mydata['sentiment'] = label_encoder.fit_transform(mydata['sentiment'])
# print(mydata['sentiment'])

# dict size taking 10k
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(mydata['review'])
# word_ind = tokenizer.word_index
# print(word_ind)
sequences = tokenizer.texts_to_sequences(mydata['review'])
x = pad_sequences(sequences, maxlen=256)
y = mydata['sentiment']
# print(x)

# Splitting the data
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
# print(x_train)
# print('\n\nlinebreak\n\n')
# print(y_train)

# Building the model
vocab_size = 10000
embedded_vector_size = 16
max_length = 256

model = Sequential()
model.add(Embedding(vocab_size, embedded_vector_size, input_length=max_length, name='my_embedding'))
model.add(Flatten())
# model.summary()
# number of neurons are 32
# use Rectified Linear Activation, relu
model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(16, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
# model.summary()

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model.summary()


# Train the model
history = model.fit(x_train, y_train, epochs=8, batch_size=128, validation_data=(x_val, y_val))
# here I found a error in binary_crossentropy/Cast, sting to float, because of data type mismatch
# so I use LabelEncoder() to encode the sentiment column to int

# print(history.history['accuracy'])
# print('\n ------------------------- \n')
# print(history.history)

# Plot the training and validation accuracy/loss curves
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Training and Validation Curves')
# plt.show()
plt.savefig('accuracy_loss_curves.png')

# Save the trained model to a file
model.save('sentiment_analysis_model.h5')







