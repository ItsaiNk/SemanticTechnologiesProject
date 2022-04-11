import pickle
from ampligraph.datasets import load_from_csv
from ampligraph.evaluation import train_test_split_no_unseen, select_best_model_ranking
from ampligraph.latent_features import ComplEx
import tensorflow as tf
from ampligraph.latent_features import save_model
from config import *
import os
from param_grid_1 import param_grid_1
from param_grid_2 import param_grid_2


def _load_sets():
    training_set = test_set = validation_set = None
    if os.path.exists("./training_set/train") and os.path.exists("./training_set/test") and os.path.exists(
            "./training_set/valid"):
        with open("./training_set/train", "rb") as f:
            training_set = pickle.load(f)
        with open("./training_set/test", "rb") as f:
            test_set = pickle.load(f)
        with open("./training_set/valid", "rb") as f:
            validation_set = pickle.load(f)
    return training_set, test_set, validation_set


def _create_sets():
    triplets_hp = load_from_csv(csv_folder, "triplets_hp_merged.csv", sep=",")
    print("Number of relations: " + str(len(triplets_hp)))
    test_val_split_size = int((len(triplets_hp) / 100) * 20)
    training_set, test_set = train_test_split_no_unseen(triplets_hp, test_size=test_val_split_size)
    val_split_size = int((test_val_split_size / 100) * 17)
    test_set, validation_set = train_test_split_no_unseen(test_set, test_size=val_split_size)
    with open("./training_set/train", "wb") as f:
        pickle.dump(training_set, f)
    with open("./training_set/test", "wb") as f:
        pickle.dump(test_set, f)
    with open("./training_set/valid", "wb") as f:
        pickle.dump(validation_set, f)
    return training_set, test_set, validation_set


def train_model():
    training_set, _, _ = _load_sets()
    if training_set is None:
        training_set, _, _ = _create_sets()

    print("Number of element in training set: " + str(len(training_set)))

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

    model.fit(training_set, early_stopping=False)

    if model.is_fitted:
        print('The model is fit!')
        save_model(model, './training_set/model.pkl')
    else:
        print('The model is not fit! Did you skip a step?')


def grid_search_hyperparams():
    training_set, test_set, validation_set = _load_sets()
    if training_set is None or test_set is None or validation_set is None:
        training_set, test_set, validation_set = _create_sets()

    print("Number of element in training set: " + str(len(training_set)))
    print("Number of element in validation set: " + str(len(validation_set)))
    print("Number of element in test set: " + str(len(test_set)))

    model_class = ComplEx
    param_grid = param_grid_2
    # best_model, best_params, best_mrr_train, ranks_test, mrr_test
    best_model = select_best_model_ranking(model_class,
                                           # Class handle of the
                                           # model to be used
                                           # Dataset
                                           training_set,
                                           validation_set,
                                           test_set,
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
    save_model(best_model[0], './model_param_2.pkl')
    with open("./training_set/best_model", "wb") as f:
        pickle.dump(best_model[1], f)
