import pickle
from ampligraph.datasets import load_from_csv
from ampligraph.evaluation import train_test_split_no_unseen, select_best_model_ranking
from ampligraph.latent_features import ComplEx
import tensorflow as tf
from ampligraph.latent_features import save_model
from config import *
import os
from param_grid_3 import param_grid_3


def train_model():
    if os.path.exists("./training_set/train") and os.path.exists("./training_set/test"):
        with open("./training_set/train", "rb") as f:
            X_train = pickle.load(f)
        with open("./training_set/test", "rb") as f:
            X_test = pickle.load(f)
    else:
        X = load_from_csv(csv_folder, "triplets_hp_merged.csv", sep=",")
        print("Number of relations: " + str(len(X)))
        test_val_split_size = int((len(X) / 100) * 20)
        X_train, X_test = train_test_split_no_unseen(X, test_size=test_val_split_size)
        val_split_size = int((test_val_split_size / 100) * 17)
        X_test, X_val = train_test_split_no_unseen(X_test, test_size=val_split_size)
        with open("./training_set/train", "wb") as f:
            pickle.dump(X_train, f)
        with open("./training_set/test", "wb") as f:
            pickle.dump(X_test, f)
        with open("./training_set/valid", "wb") as f:
            pickle.dump(X_val, f)

    print("Number of element in training set: " + str(len(X_train)))
    print("Number of element in test set: " + str(len(X_test)))

    model = ComplEx(batches_count=50,
                    seed=0,
                    epochs=200,
                    k=150,
                    eta=10,
                    optimizer='adam',
                    optimizer_params={'lr': 0.001},
                    loss='multiclass_nll',
                    regularizer='LP',
                    embedding_model_params={'negative_corruption_entities': 'all'},
                    regularizer_params={'p': 2, 'lambda': 0.0001},
                    verbose=True)

    tf.logging.set_verbosity(tf.logging.ERROR)

    model.fit(X_train, early_stopping=False)

    if model.is_fitted:
        print('The model is fit!')
        save_model(model, './training_set/model.pkl')
    else:
        print('The model is not fit! Did you skip a step?')


def grid_search_hyperparams():
    if os.path.exists("./training_set/train") and os.path.exists("./training_set/test") and os.path.exists(
            "./training_set/valid"):
        with open("./training_set/train", "rb") as f:
            X_train = pickle.load(f)
        with open("./training_set/test", "rb") as f:
            X_test = pickle.load(f)
        with open("./training_set/valid", "rb") as f:
            X_val = pickle.load(f)
    else:
        X = load_from_csv(csv_folder, "triplets_hp_merged.csv", sep=",")
        print("Number of relations: " + str(len(X)))
        test_val_split_size = int((len(X) / 100) * 20)
        X_train, X_test = train_test_split_no_unseen(X, test_size=test_val_split_size)
        val_split_size = int((test_val_split_size / 100) * 17)
        X_test, X_val = train_test_split_no_unseen(X_test, test_size=val_split_size)
        with open("./training_set/train", "wb") as f:
            pickle.dump(X_train, f)
        with open("./training_set/test", "wb") as f:
            pickle.dump(X_test, f)
        with open("./training_set/valid", "wb") as f:
            pickle.dump(X_val, f)

    print("Number of element in training set: " + str(len(X_train)))
    print("Number of element in validation set: " + str(len(X_val)))
    print("Number of element in test set: " + str(len(X_test)))

    model_class = ComplEx
    param_grid = param_grid_3
    # best_model, best_params, best_mrr_train, ranks_test, mrr_test
    best_model = select_best_model_ranking(model_class,
                                           # Class handle of the
                                           # model to be used
                                           # Dataset
                                           X_train,
                                           X_val,
                                           X_test,
                                           # Parameter grid
                                           param_grid,
                                           # Use filtered set for
                                           # eval
                                           use_filter=True,
                                           # corrupt subject and
                                           # objects separately
                                           # during eval
                                           use_default_protocol=True,
                                           # Log all the model
                                           # hyperparams and
                                           # evaluation stats
                                           verbose=False)

    print(best_model)
    save_model(best_model[0], './model_param_3.pkl')
    with open("./training_set/best_model", "wb") as f:
        pickle.dump(best_model[1], f)
