import pickle
from ampligraph.datasets import load_from_csv
from ampligraph.latent_features import restore_model
from ampligraph.evaluation import evaluate_performance
from ampligraph.evaluation import mr_score, mrr_score, hits_at_n_score
from config import csv_folder


def test_model():
    model = restore_model('./training_set/model.pkl')
    with open("./training_set/test", "rb") as f:
        X_test = pickle.load(f)

    positives_filter = load_from_csv(csv_folder, "triplets_hp_merged.csv", sep=",")
    ranks = evaluate_performance(
        X_test,
        model=model,
        filter_triples=positives_filter,  # Corruption strategy filter defined above
        corrupt_side='s+o',
        use_default_protocol=True,  # corrupt subj and obj separately while evaluating
        verbose=True
    )

    mrr = mrr_score(ranks)
    print("MRR: %.2f" % (mrr))

    hits_10 = hits_at_n_score(ranks, n=10)
    print("Hits@10: %.2f" % (hits_10))
    hits_3 = hits_at_n_score(ranks, n=3)
    print("Hits@3: %.2f" % (hits_3))
    hits_1 = hits_at_n_score(ranks, n=1)
    print("Hits@1: %.2f" % (hits_1))
