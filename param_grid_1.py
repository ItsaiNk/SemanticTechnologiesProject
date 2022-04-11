param_grid_1 = {
    "batches_count": [50, 100],
    "seed": 0,
    "epochs": [100],
    "k": [150, 100, 50],
    "eta": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    "loss": ["pairwise", "multiclass_nll", "self_adversarial"],
    # We take care of mapping the params to corresponding classes
    "loss_params": {
        # margin corresponding to both pairwise and adverserial loss
        "margin": [0.5, 20],
        # alpha corresponding to adverserial loss
        "alpha": [0.5]
    },
    "embedding_model_params": {
        # generate corruption using all entities during training
        "negative_corruption_entities": "all"
    },
    "regularizer": [None, "LP"],
    "regularizer_params": {
        "p": [2],
        "lambda": [1e-3, 1e-4, 1e-5]
    },
    "optimizer": ["adam"],
    "optimizer_params": {
        "lr": [0.01, 0.001, 0.0001]
    },
    "verbose": True
}
