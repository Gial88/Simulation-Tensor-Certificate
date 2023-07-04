# ===================================================================================================
# PROBLEM B4
#
# Build and train a classifier for the BBC-text dataset.
# This is a multiclass classification problem.
# Do not use lambda layers in your model.
#
# The dataset used in this problem is originally published in: http://mlg.ucd.ie/datasets/bbc.html.
#
# Desired accuracy and validation_accuracy > 91%
# ===================================================================================================

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pandas as pd
import numpy as np

class myCallback(tf.keras.callbacks.Callback):
    # Define the correct function signature for on_epoch_end
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.91 and logs.get('val_accuracy') > 0.91:
            print("\nAkurasi Telah mencapai 91%")

            # Stop training once the above condition is met
            self.model.stop_training = True

def solution_B4():
    bbc = pd.read_csv('https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/bbc-text.csv')

    # DO NOT CHANGE THIS CODE
    # Make sure you used all of these parameters or you can not pass this test
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_portion = .8

    # YOUR CODE HERE
    sentences = bbc['text']
    labels = bbc['category']
    # Using "shuffle=False"

    training_sentences, validation_sentences = train_test_split(sentences, train_size=training_portion, shuffle=False)
    training_labels, validation_labels = train_test_split(labels, train_size=training_portion, shuffle=False)#YOUR CODE HERE
    # Fit your tokenizer with training data
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(training_sentences)

    training_sequences = tokenizer.texts_to_sequences(training_sentences)
    training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    validation_sequences = tokenizer.texts_to_sequences(validation_sentences)
    validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    # You can also use Tokenizer to encode your label.
    tokenizer_labels = Tokenizer()
    tokenizer_labels.fit_on_texts(labels)

    training_labels = np.array(tokenizer_labels.texts_to_sequences(training_labels))
    validation_labels = np.array(tokenizer_labels.texts_to_sequences(validation_labels))

    model = tf.keras.Sequential([
        # YOUR CODE HERE.
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dropout(0.4),
        # YOUR CODE HERE. DO not change the last layer or test may fail
        tf.keras.layers.Dense(6, activation='softmax')
    ])

    # Make sure you are using "sparse_categorical_crossentropy" as a loss fuction
    model.compile(optimizer=tf.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(training_padded, training_labels,
             epochs=100,
             validation_data=(validation_padded, validation_labels),
             callbacks=[myCallback()])
    return model

    # The code below is to save your model as a .h5 file.
    # It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_B4()
    model.save("model_B4.h5")
