import numpy as np
import pickle
from ampligraph.datasets import load_from_csv
from ampligraph.evaluation import train_test_split_no_unseen
from ampligraph.latent_features import ComplEx
import tensorflow as tf
from ampligraph.latent_features import save_model, restore_model


def train_model():

    X = load_from_csv("./csv_lemm_stopwords/", "triplets_hp_merged.csv", sep=",")

    entities = np.unique(np.concatenate([X[:, 0], X[:, 2]]))
    # print(entities)

    relations = np.unique(X[:, 1])
    # print(relations)

    # 215 is the 20% of the set

    X_train, X_test = train_test_split_no_unseen(X, test_size=215)

    with open("X_train", "wb") as f:
        pickle.dump(X_train, f)
    with open("X_test", "wb") as f:
        pickle.dump(X_test, f)

    print('Train set size: ', X_train.shape)
    print('Test set size: ', X_test.shape)

    model = ComplEx(batches_count=50,
                    seed=0,
                    epochs=200,
                    k=150,
                    eta=5,
                    optimizer='adam',
                    optimizer_params={'lr': 1e-3},
                    loss='multiclass_nll',
                    regularizer='LP',
                    regularizer_params={'p': 3, 'lambda': 1e-5},
                    verbose=True)

    positives_filter = X

    tf.logging.set_verbosity(tf.logging.ERROR)

    model.fit(X_train, early_stopping=False)

    if model.is_fitted:
        print('The model is fit!')
        save_model(model, './test1.pkl')
    else:
        print('The model is not fit! Did you skip a step?')
