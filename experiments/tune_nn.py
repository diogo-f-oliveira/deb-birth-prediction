"""Example NN hyperparameter search (T08B).

Starts from an experiment JSON and searches the space below with Ray Tune and
HyperOpt; every trial is saved as a normal run under results/tune/. The space
is a starting point: final spaces, budgets and seeds are fixed in T08.

    python -m experiments.tune_nn --formulation boundary --num-samples 50
"""
import argparse

from ray import tune

from src.debbirth.tuning import run_search


def search_space():
    """Fixed entries pass through unchanged; epochs and checkpoint policy come from the base config."""
    return {
        "lr": tune.loguniform(1e-4, 1e-2),
        "weight_decay": tune.loguniform(1e-5, 1e-1),
        "batch_size": tune.choice([64, 128, 256, 512]),
        "dropout": tune.uniform(0.0, 0.3),
        # hidden_dims = [width] * n_layers
        "n_layers": tune.choice([1, 2, 3]),
        "width": tune.choice([16, 32, 64, 128]),
        "progress_every": 1,  # keep every trial's per-epoch validation curve
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--formulation", required=True, choices=("full_par", "normalized", "boundary"))
    parser.add_argument("--config", help="base experiment JSON (default: experiments/nn_<formulation>.json)")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--name", help="search name (default: nn_<formulation>)")
    parser.add_argument("--search-seed", type=int, default=0)
    parser.add_argument("--max-concurrent", type=int)
    parser.add_argument("--cpus-per-trial", type=int, help="default: 1 + the base config's num_workers")
    args = parser.parse_args(argv)
    return run_search(family="nn", base_config=args.config or f"experiments/nn_{args.formulation}.json",
                      space=search_space(), name=args.name or f"nn_{args.formulation}",
                      num_samples=args.num_samples, search_seed=args.search_seed,
                      max_concurrent=args.max_concurrent, cpus_per_trial=args.cpus_per_trial, script=__file__)


if __name__ == "__main__":
    main()
