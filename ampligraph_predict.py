import numpy as np
import os
from ampligraph.datasets import load_from_csv
from ampligraph.latent_features import restore_model
from scipy.special import expit
import csv
from config import *
from tqdm import tqdm
import random


def create_unseen():
    triplets_hp = load_from_csv(csv_folder, "triplets_hp_merged.csv", sep=",")
    random.seed(42)
    subjects = np.unique(triplets_hp[:, 0]).tolist()
    predicates = np.unique(triplets_hp[:, 1]).tolist()
    objects = np.unique(triplets_hp[:, 2]).tolist()
    triplets_hp = triplets_hp.tolist()
    for i in range(num_gen_repetions):
        with open(csv_folder + "unseen" + str(i) + ".csv", "w", newline="") as f:
            writer = csv.writer(f)
            for _ in tqdm(range(num_gen_unseen)):
                added = False
                while not added:
                    s = random.choice(subjects)
                    p = random.choice(predicates)
                    o = random.choice(objects)
                    if s != o:
                        triple = [s, p, o]
                        if triple not in triplets_hp:
                            writer.writerow(triple)
                            added = True


def predict_unseen():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    model = restore_model('./training_set/model.pkl')
    for i in range(num_gen_repetions):
        with open(csv_folder + "predicted"+str(i)+".csv", "w", newline="") as f_out:
            writer = csv.writer(f_out)
            triplets_unseen = load_from_csv(csv_folder, "unseen"+str(i)+".csv", sep=",")
            scores = model.predict(triplets_unseen)
            probs = expit(scores)
            for j in range(len(probs)):
                if probs[j] >= 0.98:
                    writer.writerow(triplets_unseen[j])


