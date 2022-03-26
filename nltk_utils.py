import pickle

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet


def remove_stopwords(text, custom_stopwords):
    if custom_stopwords:
        stopwords_list = load_stopwords_custom_object()
    else:
        stopwords_list = stopwords.words('english')
    text_tokens = word_tokenize(text)
    tokens = [word for word in text_tokens if word not in stopwords_list]
    text = ""
    for token in tokens:
        text = text + str(token) + " "
    return text


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return "n"


def lemmatize_triplets(triplets):
    lemmatizer = WordNetLemmatizer()
    for triplet in triplets:
        relation = triplet["relation"]
        tagged_tokenized_relation = pos_tag(word_tokenize(str(relation)))
        new_relation = ""
        for token in tagged_tokenized_relation:
            if new_relation == "":
                new_relation = new_relation + str(lemmatizer.lemmatize(token[0], get_wordnet_pos(token[1])))
            else:
                new_relation = new_relation + " " + str(lemmatizer.lemmatize(token[0], get_wordnet_pos(token[1])))
        triplet["relation"] = new_relation


def lemmatize_triplets_only_verbs(triplets):
    lemmatizer = WordNetLemmatizer()
    for triplet in triplets:
        relation = triplet["relation"]
        tagged_tokenized_relation = pos_tag(word_tokenize(str(relation)))
        new_relation = ""
        for token in tagged_tokenized_relation:
            if new_relation == "":
                if get_wordnet_pos(token[1]) is wordnet.VERB:
                    new_relation = new_relation + str(lemmatizer.lemmatize(token[0], wordnet.VERB))
                else:
                    new_relation = new_relation + str(token[0])
            else:
                if get_wordnet_pos(token[1]) is wordnet.VERB:
                    new_relation = new_relation + " " + str(lemmatizer.lemmatize(token[0], wordnet.VERB))
                else:
                    new_relation = new_relation + " " + str(token[0])
        triplet["relation"] = new_relation


def setup():
    nltk.download()


def print_stopwords():
    for word in stopwords.words('english'):
        print(word)


def create_stopwords_custom_object(filename):
    custom_stopwords = []
    with open(filename, "r") as f:
        custom_stopwords = f.read().splitlines()
    print("Custom stopwords loaded: " + str(len(custom_stopwords)))
    print(custom_stopwords)
    with open("./obj/custom_stopwords.obj", "wb") as f:
        pickle.dump(custom_stopwords, f)


def load_stopwords_custom_object():
    with open("./obj/custom_stopwords.obj", "rb") as f:
        return pickle.load(f)
