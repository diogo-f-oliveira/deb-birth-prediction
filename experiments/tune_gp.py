"""Example GP hyperparameter search (T08B).

Starts from an experiment JSON and searches the space below with Ray Tune and
HyperOpt; every trial is saved as a normal run under results/tune/. The space
is a starting point: final spaces, budgets and seeds are fixed in T08.

    python -m experiments.tune_gp --formulation boundary --num-samples 50
"""
import argparse

from ray import tune

from src.debbirth.tuning import run_search


def search_space():
    """Fixed entries pass through unchanged; population and generations come from the base config."""
    return {
        "tournament_fraction": tune.quniform(0.05, 0.5, 0.01),
        "parsimony_coefficient": tune.qloguniform(1e-5, 1e-3, 1e-5),
        # Operator group: crossover = 1 - reproduction - mutation total; shares from (u1, u2).
        "p_reproduction": tune.quniform(0.0, 0.3, 0.01),
        "p_mutation_total": tune.quniform(0.05, 0.4, 0.01),
        "mutation_u1": tune.uniform(0.0, 1.0),
        "mutation_u2": tune.uniform(0.0, 1.0),
        # Named registered sets; this choice replaces the base config's constants.
        "constants": tune.choice(["extended", "extended_c0"]),
        "progress_every": 1,  # keep every trial's per-generation validation curve
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--formulation", required=True, choices=("full_par", "normalized", "boundary"))
    parser.add_argument("--config", help="base experiment JSON (default: experiments/gp_<formulation>.json)")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--name", help="search name (default: gp_<formulation>)")
    parser.add_argument("--search-seed", type=int, default=0)
    parser.add_argument("--max-concurrent", type=int)
    parser.add_argument("--cpus-per-trial", type=int, help="default: the base config's num_workers")
    args = parser.parse_args(argv)
    return run_search(family="gp", base_config=args.config or f"experiments/gp_{args.formulation}.json",
                      space=search_space(), name=args.name or f"gp_{args.formulation}",
                      num_samples=args.num_samples, search_seed=args.search_seed,
                      max_concurrent=args.max_concurrent, cpus_per_trial=args.cpus_per_trial, script=__file__)


if __name__ == "__main__":
    main()
