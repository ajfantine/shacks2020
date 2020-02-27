#made, with regret, by Alex Fantine

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.utils import to_categorical
import random
from nltk.corpus import words

letters = "abcdefghijklmnopqrstuvwxyz"
letters = [letter for letter in letters]

#basically the hardcoded version of this entire friggin algorithm... that's right,
#i use the hardcoded version to generate data to train the ml version. bite me.
def get_next_letter(word):
    last_letter = word[len(word)-1].lower()
    last_index = letters.index(last_letter)
    next_letter = letters[(last_index + 1)%len(letters)]
    return next_letter

def create_data():
    #import nltk's word corpora
    word_list = words.words()

    #arbitrary data size, 10000 seemed cute
    data_size = 10000
    random.seed(13)
    random.shuffle(word_list)
    #get a smaller randomized list of words
    word_list = word_list[:data_size]

    #make the basic x and y values, in terms of word and next alphabetical letter
    x = []
    y = []
    for word in word_list:
        x.append(word.lower())
        y.append(get_next_letter(word))

    #convert y values into ascii ints
    y = [ord(char) for char in y]

    #max word len = 23
    sorted_words = sorted(word_list, key=len)
    max_word_len = len(sorted_words[-1])
    #find the largest word length, which happened to be 23 chars

    #convert the char sequences to ascii ints
    word_seqs = [[ord(char) for char in word] for word in x]

    #pad the words to all be 23 chars, pre-pad with 0s
    input_sequences = np.array(pad_sequences(word_seqs, maxlen=max_word_len))
    #123 is the ascii code for 'z', need to one hot encode labels
    labels = to_categorical(y, num_classes=123)

    #split 90/10 train/test
    index = int(.9 * data_size)
    train_x = input_sequences[:index]
    test_x = input_sequences[index:]
    train_y = labels[:index]
    test_y = labels[index:]

    return train_x, train_y, test_x, test_y, max_word_len

def create_model(train_x, train_y, max_word_len):
    #123 is the ascii code for 'z'
    num_chars=123

    model = Sequential()

    #number of distinct chars in training set, size of embedding vectors,
    #size of each input sequence (size of the largest word as all words
    #have been normalized to that size)
    #second param was 10 before
    model.add(Embedding(num_chars, 32, input_length = max_word_len))

    model.add(LSTM(100))

    model.add(Dropout(0.1))

    #softmax squashes the values between 0 and 1 but makes sure they add to 1
    #the values are proprtional to the number of values
    model.add(Dense(num_chars, activation='softmax'))

    #calculates with an ouput between 0 and 1
    model.compile(loss='categorical_crossentropy', optimizer = 'adam')
    model.fit(train_x, train_y, epochs=10, verbose=1)

    return model

#return the string rep of the given int (a char)
def get_char(prediction):
    char_index = np.where(prediction == 1)
    #first index accesses array of found locations, second accesses first value
    return chr(char_index[0][0])

#take a list of words and return guesses, i.e. the next char alphabetically
def make_guess(max_word_len, word_list, model):
    words_data = [[ord(char) for char in word] for word in word_list]
    model_input = np.array(pad_sequences(words_data, maxlen=max_word_len))
    predictions = model.predict(model_input)
    for i in range(len(predictions)):
        print("Based on your input", word_list[i], ", the next character alphabetically is:", get_char(predictions[i]))

if __name__ == '__main__':
    #uncomment below lines if you dont have the model saved lol sucker
    #train_x, train_y, test_x, test_y, max_word_len = create_data()
    #model = create_model(train_x, train_y, max_word_len)

    model = load_model('shacks.model')
    max_word_len = 23
    input_raw = ""
    #loop for the sake of shacks demo
    while(input_raw != 'q'):
        input_raw = input("Please enter a word or multiple words separated by spaces: ")
        if input_raw != 'q':
            word_list = input_raw.split(" ")
            word_list = [word for word in word_list if word != ''] #weird edge case
            make_guess(max_word_len, word_list, model)
