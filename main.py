import csv
from ampligraph_training import train_model
from ampligraph_test import test_model
from coreference import coref_resolution
from triplets import OpenIEClient
import pickle
from nltk_utils import lemmatize_triplets, lemmatize_triplets_only_verbs, print_stopwords, \
    create_stopwords_custom_object, sentence_tokenize, preprocess_text
from nltk_utils import remove_stopwords
from config import *
from neo4j_utils import *
# This is a sample Python script.

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


def ordered_triplets_to_csv(dict, filename):
    with open(filename, "w", newline='') as f:
        writer = csv.writer(f)
        for key in dict.keys():
            list = dict[key]
            for el in list:
                data = [el["subject"], el["relation"], el["object"]]
                writer.writerow(data)


def create_dict_by_element(triplets, element, save=False, filename=None):
    dict_element = {}
    for triplet in triplets:
        if not triplet[element] in dict_element:
            dict_element[triplet[element]] = []
        else:
            pass
        dict_element[triplet[element]].append(triplet)
    if save:
        with open(filename, "wb") as f:
            pickle.dump(dict_element, f)

    return dict_element


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

        if dict_by_element:
            dict_subjects = create_dict_by_element(triplets, "subject", save_files,
                                                   "./obj/dict_subjects_hp" + str(i) + ".obj")
            _ = create_dict_by_element(triplets, "relation", save_files, "./obj/dict_relations_hp" + str(i) + ".obj")
            _ = create_dict_by_element(triplets, "object", save_files, "./obj/dict_objects_hp" + str(i) + ".obj")
            # ordered_triplets_to_csv(dict_subjects, csv_folder + "triplets_hp" + str(i) + "_subjects.csv")

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
    with open(csv_folder+"triplets_hp_merged.csv", "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            add_triple(graph, row[0], row[1], row[2])


def main():
    # create_stopwords_custom_object()
    # create_csvs()
    # merge_csvs()
    # train_model()
    # test_model()
    create_graph_node4j()
    return


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
