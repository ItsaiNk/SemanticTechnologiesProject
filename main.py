import csv
from SPARQL_query import query
from ampligraph_training import train_model, grid_search_hyperparams
from ampligraph_test import test_model
from coreference import coref_resolution
from references import create_ref_elements
from triplets import OpenIEClient
import pickle
from nltk_utils import lemmatize_triplets, lemmatize_triplets_only_verbs, print_stopwords, \
    create_stopwords_custom_object, sentence_tokenize, preprocess_text
from nltk_utils import remove_stopwords
from config import *
from neo4j_utils import *
from ampligraph_predict import *


# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def readfile(filename):
    with open("./" + filename, "r") as f:
        return f.read()


def triplets_to_csv(triplets, filename):
    with open(filename, "w", newline='') as f:
        writer = csv.writer(f)
        for triplet in triplets:
            data = [triplet["subject"], triplet["relation"], triplet["object"]]
            writer.writerow(data)


def create_csvs():
    client = OpenIEClient()
    for i in range(start, stop + 1):
        text = readfile("./plots/hp" + str(i))
        print("\n*********TEXT*********\n")
        print(text)
        print("\n**********************\n")
        text = preprocess_text(text)
        if coreference:
            text = coref_resolution(text)
        if stopwords:
            text = remove_stopwords(text, custom_stopwords)

        triplets = client.extract_triplets(text)

        if lemmatize:
            if only_verbs:
                lemmatize_triplets_only_verbs(triplets)
            else:
                lemmatize_triplets(triplets)

        triplets_to_csv(triplets, csv_folder + "triplets_hp" + str(i) + ".csv")

    # how to manipulate: is a dict with 3 keys:
    # subject => triplet["subject"] return the subject of the triplet
    # relation => triplet["relation"] return the relation (predicate) of the triplet
    # object => triplet["object"] return the object of the triplet
    #
    # each triplet is of class 'dict'
    # var 'triplets' is of class 'list'


def merge_csvs():
    with open(csv_folder + "triplets_hp_merged.csv", "w", newline='') as f_out:
        writer = csv.writer(f_out)
        for i in range(1, 8):
            with open(csv_folder + "triplets_hp" + str(i) + ".csv", "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    writer.writerow(row)


def create_graph_node4j():
    graph = create_graph()
    dict_elements = create_ref_elements()
    with open(csv_folder + "triplets_hp_merged.csv", "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            add_triple(graph, row[0], row[1], row[2], dict_elements)
    for i in range(num_gen_repetions):
        with open(csv_folder + "predicted"+str(i)+".csv", "r", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                add_triple(graph, row[0], row[1], row[2], dict_elements)
    return graph


def main():
    # create_stopwords_custom_object()
    # create_csvs()
    # merge_csvs()
    # grid_search_hyperparams()
    # test_model()
    # create_unseen()
    # predict_unseen()
    # g = create_graph_node4j()
    # 116 subjects, 207 objects, 134 subjects now, 214 objects now
    return


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
