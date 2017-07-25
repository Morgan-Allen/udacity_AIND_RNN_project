import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras



def window_transform_series(series, window_size):
    """
    Transforms the input series and window-size into a set of input/output
    pairs for use with our RNN model.
    """
    X = [series[n:n + window_size] for n in range(len(series) - window_size)]
    y = series[window_size:]
    X = np.asarray(X)
    y = np.asarray(y)
    y.shape = (len(y), 1)
    return X, y


def build_part1_RNN(window_size):
    """
    Builds an RNN to perform regression on our time series input/output data
    """
    model = Sequential()
    model.add(LSTM(5, input_shape = (window_size, 1)))
    model.add(Dense(1))
    return model


ALL_PUNCTUATION = ('!', ',', '.', ':', ';', '?')
ALL_ASCII       = set(c for c in "abcdefghijklmnopqrstuvwxyz")

def cleaned_text(text):
    """
    Returns the text input with only ascii lowercase and certain punctuation
    characters included.
    """
    filtered = []
    for c in text:
        if c in ALL_PUNCTUATION or c in ALL_ASCII:
            filtered.append(c)
        else:
            filtered.append(' ')
    return "".join(filtered)


def window_transform_text(text, window_size, step_size):
    """
    Transforms the input text and window-size into a set of input/output
    pairs for use with our RNN model
    """
    indices = range(window_size, len(text))[::step_size]
    inputs  = [text[i - window_size:i] for i in indices]
    outputs = [text[i]                 for i in indices]
    return inputs, outputs


def build_part2_RNN(window_size, num_chars):
    """
    Builds the required RNN model: a single LSTM hidden layer with softmax
    activation
    """
    model = Sequential()
    model.add(LSTM(200, input_shape = (window_size, num_chars)))
    model.add(Dense(num_chars, activation = 'softmax'))
    return model


















