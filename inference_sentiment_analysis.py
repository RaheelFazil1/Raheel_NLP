import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the previously saved trained model
loaded_model = tf.keras.models.load_model('sentiment_analysis_model.h5')

# Take user input for a movie review
input_text = input("Enter a movie review: ")
print(input_text)

# Preprocess the input text
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts([input_text])
input_sequence = tokenizer.texts_to_sequences([input_text])
input_sequence = pad_sequences(input_sequence, maxlen=256)

# Predict sentiment using the loaded model
predicted_sentiment = loaded_model.predict(input_sequence)[0][0]
if predicted_sentiment >= 0.5:
    sentiment_label = 'Positive'
else:
    sentiment_label = 'Negative'

# Display the result
print("Predicted Sentiment:", sentiment_label)
print("Input Text:", input_text)





