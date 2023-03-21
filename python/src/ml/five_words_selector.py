import pandas as pd
import random
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))

def get_filtered_sentence():
    print("inside generate_five_random_words")
    with open('/home/sjonnal3/Hate_Speech_Detection/Applied_Machine_Learning/chase-master/output/google-10000-english.txt') as f:
        contents = f.read()
        word_tokens = word_tokenize(contents)
        filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]

    # random_words= random.sample(filtered_sentence, 15)
    # print(random_words)
    return filtered_sentence


# if __name__ == "__main__":
#     print("Inside five_words_selector")

#     random_words = generate_five_random_words()