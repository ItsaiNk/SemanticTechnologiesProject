import numpy as np
import pandas as pd
import pickle
from ampligraph.datasets import load_from_csv
from ampligraph.latent_features import restore_model
from ampligraph.evaluation import evaluate_performance
from scipy.special import expit
import csv
from config import csv_folder

def test_model():
    X = load_from_csv("./csv_lemm_stopwords/", "triplets_hp_merged.csv", sep=",")
    positives_filter = X

    model = restore_model('./test1.pkl')

    with open("X_train", "rb") as f:
        X_train = pickle.load(f)
    with open("X_test", "rb") as f:
        X_test = pickle.load(f)

    X_unseen = X_test
    # X_unseen = load_from_csv(".", "testing.csv", sep=",")
    # print(X_unseen)

    positives_filter = X_train
    unseen_filter = np.array(list({tuple(i) for i in np.vstack((positives_filter, X_unseen))}))

    ranks_unseen = evaluate_performance(
        X_unseen,
        model=model,
        filter_triples=unseen_filter,  # Corruption strategy filter defined above
        corrupt_side='s+o',
        use_default_protocol=False,  # corrupt subj and obj separately while evaluating
        verbose=True
    )

    scores = model.predict(X_unseen)
    probs = expit(scores)
    df = pd.DataFrame(list(zip([' '.join(x) for x in X_unseen],
                               ranks_unseen,
                               np.squeeze(scores),
                               np.squeeze(probs))),
                      columns=['statement', 'rank', 'score', 'prob']).sort_values("score")
    with open("./dataframe.csv", "w", newline="") as f:
        df.to_csv(f)

    with open(csv_folder+"predicted.csv", "w") as f:
        writer = csv.writer(f)
        for i in range(len(probs)):
            if probs[i]>=0.75:
                writer.writerow(X_unseen[i])




