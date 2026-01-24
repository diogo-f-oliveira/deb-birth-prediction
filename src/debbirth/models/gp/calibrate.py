from dataclasses import asdict
from typing import Dict, Any, Optional

import os
import random
from functools import partial
from datetime import datetime as dt

import numpy as np
from scipy.stats import beta
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from ray import tune, train

os.environ['RAY_AIR_NEW_OUTPUT'] = '0'

from ...data.schema import DatasetSpec
from ...evaluate.metrics import BinaryMetrics
from .config import GPConfig, TrainGPConfig
from .train import train_gp_classifier, save_gp_run
from ...evaluate.predict import evaluate_binary_classifier


def _dirichlet3_from_uniforms(u1, u2, a1=1, a2=1, a3=1, eps=1e-12):
    # Deterministic transform: (u1,u2) -> Dirichlet(a1,a2,a3)

    u1 = min(max(u1, eps), 1 - eps)
    u2 = min(max(u2, eps), 1 - eps)

    v1 = beta.ppf(u1, a1, a2 + a3)
    v2 = beta.ppf(u2, a2, a3)

    p1 = v1
    p2 = (1 - v1) * v2
    p3 = (1 - v1) * (1 - v2)
    return p1, p2, p3


def evaluate_config(config: Dict[str, Any], data_spec: DatasetSpec, random_state: int = 42,
                    report_metrics: bool = False, verbose: int = 0, num_workers=1, **gp_config_params) -> Optional[Dict[str, Any]]:
    """Evaluate a given hyperparameter configuration for genetic programming symbolic classifier."""

    population_size = config.get("pop_size")
    generations = config.get("n_gen")
    tournament_fraction = config.get("tourn_frac")
    tournament_size = max(2, int(population_size * tournament_fraction))
    parsimony_coefficient = config.get("parsi_coef")
    # Genetic operation probabilities

    p_reprod = config.get("p_reprod")
    p_mut_total = config.get("p_mut_total")
    p_crossover = 1 - p_reprod - p_mut_total

    # Calculate mutation probabilities
    p_subtree_mutation, p_hoist_mutation, p_point_mutation = _dirichlet3_from_uniforms(
        u1=config.get("u1_subtree"), u2=config.get("u2_hoist"),
    )
    p_hoist_mutation *= p_mut_total
    p_point_mutation *= p_mut_total
    p_subtree_mutation *= p_mut_total

    # Build GPConfig and TrainGPConfig
    gp_cfg = GPConfig(
        population_size=population_size,
        generations=generations,
        tournament_size=tournament_size,
        parsimony_coefficient=parsimony_coefficient,
        p_crossover=p_crossover,
        p_subtree_mutation=p_subtree_mutation,
        p_hoist_mutation=p_hoist_mutation,
        p_point_mutation=p_point_mutation,
    )

    train_cfg = TrainGPConfig(
        gp=gp_cfg,
        data_spec=data_spec,
        outdir=None,
        # Force verbose to 0 so models themselves are quiet; Tune will report progress.
        verbose=verbose,
        class_weights=config.get("class_weights", None),
        seed=random_state,
        num_workers=num_workers,
    )

    # Run training (this may be slow depending on population/generations).
    output = train_gp_classifier(train_cfg, save_run=False)
    output['train_config'] = train_cfg
    val_metrics = output.get("val_metrics")

    # Report the CV R2 score to Ray Tune
    if report_metrics:
        train.report(asdict(val_metrics))
    else:
        return output


def hyperopt_calibration(search_space: Dict[str, Any],
                         data_spec: DatasetSpec,
                         num_samples: int = 100,
                         metric: str = 'f1_macro',
                         mode: str = 'max',
                         current_best_params=None,
                         run_name=None, tune_dir=None, max_concurrent_trials=None,
                         evaluate_on_test=False, save_best_model=False,
                         random_state: int = 42,
                         gp_config_parms: Optional[Dict[str, Any]] = None):
    if run_name is None:
        time_str = dt.now().strftime('%Y-%m-%d_%H-%M-%S')
        run_name = f"GPSC__{time_str}"
    if tune_dir is None:
        tune_dir = os.path.abspath('results/tune')

    trainable = partial(evaluate_config,
                        data_spec=data_spec,
                        report_metrics=True,
                        verbose=0,
                        random_state=random_state,
                        **gp_config_parms)
    alg = HyperOptSearch(metric=metric, mode=mode, points_to_evaluate=current_best_params,
                         random_state_seed=random_state)

    tuner = tune.Tuner(
        trainable,
        tune_config=tune.TuneConfig(
            search_alg=alg,
            max_concurrent_trials=max_concurrent_trials,
            num_samples=num_samples,
            trial_dirname_creator=lambda t: t.trial_id,
            scheduler=None,
        ),
        run_config=train.RunConfig(
            name=run_name,
            verbose=1,
            progress_reporter=CLIReporter(
                metric=metric, mode=mode,
                metric_columns=['f1_macro', 'precision_macro', 'recall_macro', 'accuracy'],
                parameter_columns=[p for p, s in search_space.items() if isinstance(s, tune.search.sample.Domain)],
                sort_by_metric=True,
                max_report_frequency=60,
            ),
            storage_path=tune_dir,
        ),
        param_space=search_space,
    )
    results = tuner.fit()

    best_result = results.get_best_result(metric=metric, mode=mode)
    print("Best parameters found: ", best_result.config)
    print(f"Best score: {best_result.metrics[metric]:.4f}")

    best_model_train_output = evaluate_config(config=best_result.config, data_spec=data_spec, random_state=random_state,
                                              report_metrics=False, verbose=True, num_workers=-1)


    if save_best_model:
        save_gp_run(
            model=best_model_train_output['model'],
            cfg=best_model_train_output['train_config'],
            val_metrics=best_model_train_output['val_metrics'])

    if evaluate_on_test:
        test_metrics = evaluate_binary_classifier(
            model=best_model_train_output["model"],
            X=best_model_train_output['features']['test'],
            y=best_model_train_output['targets']['test']
        )
        best_model_train_output['test_metrics'] = test_metrics

        if save_best_model:
            test_metrics.save_json(best_model_train_output['train_config'].outdir / "metrics" / "test_metrics.json")

    return best_model_train_output


if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)
    random.seed(seed)

    data_spec = DatasetSpec(
        feature_set="dimensionless"
    )

    # Evaluate a single configuration
    # cfg = {
    #     "population_size": 100,
    #     "generations": 10,
    #     "tournament_fraction": .2,
    #     "parsimony_coefficient": 1e-3,
    #     "p_reprod": .1,
    #     "p_mut_total": .1,
    #     "u1_subtree": .3,
    #     "u2_hoist": .3,
    #     "seed": 42,
    #     "outdir": None,
    # }
    # val_metrics = evaluate_config(config=cfg, data_spec=data_spec, random_state=seed, report_metrics=False,
    #                               verbose=True)
    # print(val_metrics)

    # Hyperparameter optimization
    search_space = {
        "pop_size": tune.qrandint(50, 500, 25),
        "n_gen": tune.qrandint(5, 60, 5),
        "tourn_frac": tune.quniform(0.05, 0.5, 0.01),
        "parsi_coef": tune.qloguniform(1e-4, 1e-1, 1e-4),
        "p_reprod": tune.quniform(0.0, 0.4, 0.01),
        "p_mut_total": tune.quniform(0.0, 0.4, 0.01),
        "u1_subtree": tune.uniform(0.0, 1.0),
        "u2_hoist": tune.uniform(0.0, 1.0),
    }
    from .functions import DEFAULT_FUNCTION_SET, EXTENDED_FUNCTION_SET
    best_output = hyperopt_calibration(
        search_space=search_space,
        data_spec=data_spec,
        num_samples=10,
        metric='f1_macro',
        mode='max',
        # run_name='GPSC_Hyperopt_Test',
        # tune_dir='results/tune',
        # max_concurrent_trials=2,
        evaluate_on_test=True,
        save_best_model=True,
        random_state=seed,
        gp_config_parms={
            'function_set': EXTENDED_FUNCTION_SET,
        }
    )

    print("\nBest program:")
    print(best_output["best_program"])
    print("\nTest metrics:")
    print(best_output["test_metrics"])