# This file contains a simple implementation of the Word2vec Skim-Gram model from scratch.
#
# Copyright (c) 2023 Brad Edwards
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

## A brief explanation of the Word2vec Skip-Gram model
# The Word2vec Skip-Gram model is a neural network that learns word embeddings from a corpus of text.
# Word embeddings are vector representations of words that capture the semantic meaning of the words.
# The Word2vec Skip-Gram model learns word embeddings by predicting the context of each word in the corpus.
# The model is trained by feeding it a word and a context word, and then updating the weights of the model
# so that the model is more likely to predict the context word given the input word.
# The model is trained using stochastic gradient descent, and the weights are updated using backpropagation.

import numpy as np
import collections
from gensim.models import Word2Vec
from keras.models import Sequential
from keras.layers import Dense, Embedding, Lambda


# Function to convert a string of text into a list of words
def preprocess_text(text):
    # Convert all words to lower case and split the string into words
    return text.lower().split()


# Function to prepare the dataset for training the model
def prepare_dataset(words, window_size=2):
    # The dataset will be a list of pairs, where each pair consists of a word and its context (the words surrounding it)
    dataset = []
    for i, word in enumerate(words):
        # For each word, we define its context as the words within a window of a certain size around it
        start = max(0, i - window_size)
        end = min(len(words), i + window_size + 1)
        for j in range(start, end):
            # We add each pair (word, context_word) to the dataset
            if i != j:
                dataset.append((word, words[j]))
    return dataset


# Function to build a vocabulary from the input words
def build_vocab(words, vocab_size=10000):
    # We first count the frequency of each word in the input
    word_counts = collections.Counter(words)
    # We then keep the most common words, up to a maximum vocabulary size
    most_common = word_counts.most_common(vocab_size - 1)
    vocab = {k: v for v, (k, _) in enumerate(most_common)}
    # We add a special token for unknown words
    vocab["UNK"] = len(vocab)
    return vocab


# Function to convert a list of words into a list of their corresponding integer IDs
def convert_words_to_int(words, vocab):
    return [vocab[word] if word in vocab else vocab["UNK"] for word in words]


# Function to create the Skip-Gram model
def skip_gram_model(vocab_size, embedding_dim):
    # We initialize the weight matrices with random values
    W1 = np.random.rand(vocab_size, embedding_dim)
    W2 = np.random.rand(embedding_dim, vocab_size)
    return W1, W2


# Function to perform forward propagation
def forward_propagation(word, W1, W2):
    # We calculate the hidden layer values (h) as the dot product of the input word and W1
    h = np.dot(W1.T, word)
    # We then calculate the output layer values (u) as the dot product of h and W2
    u = np.dot(W2.T, h)
    # We apply the softmax function to get the predicted probabilities of each context word
    y_hat = softmax(u)
    return h, u, y_hat


# Function to perform backward propagation
def backward_propagation(word, true_word, h, u, y_hat, W1, W2, learning_rate=0.1):
    # We calculate the error as the difference between the predicted probabilities and the true values
    e = y_hat - true_word
    # We calculate the gradients for W1 and W2
    dW1 = np.outer(word, np.dot(W2, e))
    dW2 = np.outer(h, e)
    # We update the weights using the gradients and the learning rate
    W1 = W1 - learning_rate * dW1
    W2 = W2 - learning_rate * dW2
    return W1, W2


# Function to calculate the softmax of an array
def softmax(x):
    e_x = np.exp(x - np.max(x))
    # Subtract max(x) to improve the stability of the calculation
    return e_x / e_x.sum(axis=0)


# Function to train the model
def train_model(
    dataset, vocab, vocab_size, embedding_dim, learning_rate=0.1, epochs=1000
):
    # We initialize the weight matrices
    W1, W2 = skip_gram_model(vocab_size, embedding_dim)
    # We perform forward and backward propagation for a specified number of epochs
    for epoch in range(epochs):
        loss = 0
        # For each (word, context) pair in the dataset
        for word, context in dataset:
            # We convert the word and its context to one-hot vectors
            word_id = vocab[word]
            context_id = vocab[context]
            word_vector = np.zeros(vocab_size)
            word_vector[word_id] = 1
            context_vector = np.zeros(vocab_size)
            context_vector[context_id] = 1
            # We perform forward propagation
            h, u, y_hat = forward_propagation(word_vector, W1, W2)
            # We calculate the loss
            loss += -np.sum(
                [u[word_id] for word_id in np.where(context_vector)[0]]
            ) + len(np.where(context_vector)[0]) * np.log(np.sum(np.exp(u)))
            # We perform backward propagation
            W1, W2 = backward_propagation(
                word_vector, context_vector, h, u, y_hat, W1, W2, learning_rate
            )
        # We print the loss at each epoch
        print(f"Epoch: {epoch}, Loss: {loss}")
    return W1, W2


def main():
    # Example corpus
    text = "the quick brown fox jumps over the lazy dog"

    # Preprocess the text
    words = preprocess_text(text)

    # Prepare the dataset
    dataset = prepare_dataset(words)

    # Build the vocabulary and get the vocabulary size
    vocab = build_vocab(words)
    vocab_size = len(vocab)

    # Convert words to integers
    words_int = convert_words_to_int(words, vocab)

    # Initialize the model
    embedding_dim = 100  # dimension of the word embeddings
    W1, W2 = skip_gram_model(vocab_size, embedding_dim)

    # Train the model
    W1, W2 = train_model(dataset, vocab, vocab_size, embedding_dim, epochs=100)

    # Print the word embeddings for the words in the vocabulary
    for word, id in vocab.items():
        print(f"Word: {word}")
        print(f"Embedding: {W1[id]}")


if __name__ == "__main__":
    main()


def practical_with_gensim(sentences):
    # Assuming `sentences` is a list of lists of tokens
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=5, workers=4)
    model.save("word2vec.model")
