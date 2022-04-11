param_grid_2 = {
    "batches_count": [50],
    "seed": 0,
    "epochs": [200],
    "k": [150],
    "eta": [5, 10, 15, 20],
    "loss": ["multiclass_nll"],
    "embedding_model_params": {
        # generate corruption using all entities during training
        "negative_corruption_entities": "all"
    },
    "regularizer": ["LP"],
    "regularizer_params": {
        "p": [2, 3],
        "lambda": [1e-4, 1e-5]
    },
    "optimizer": ["adam"],
    "optimizer_params": {
        "lr": [1e-3, 1e-4]
    },
    "verbose": True
}

# Params selected by grid search:
# {
#     'batches_count': 50,
#      'seed': 0,
#      'epochs': 200,
#      'k': 150,
#      'eta': 10,
#      'loss': 'multiclass_nll',
#      'regularizer': 'LP',
#      'optimizer': 'adam',
#      'verbose': True,
#      'embedding_model_params': {'negative_corruption_entities': 'all'},
#      'regularizer_params': {'p': 2, 'lambda': 0.0001},
#      'optimizer_params': {'lr': 0.001}
# }
