"""Per-generation GP validation (T08D).

After each logged generation, the program gplearn would return if the run
stopped there (lowest raw training loss in that generation) is evaluated on the
validation split through GPPredictor inference semantics. The fit-bound
boundary metric is never used on validation rows.
"""
from time import perf_counter

from .boundary import GPBoundaryModel
from .generation_hook import best_of_generation
from .predict import GPPredictor
from ...evaluate.predict import metrics_from_predictions
from ...evaluate.progress import ProgressLogger, bce_from_logits

GP_EXTRA_COLUMNS = ("best_length", "average_length")


def gp_progress_logger(cfg, path=None):
    return ProgressLogger(step_kind="generation", total_steps=cfg.gp.generations, every=cfg.progress_every,
                          path=path, extra_columns=GP_EXTRA_COLUMNS, print_extras=("best_length",))


def gp_validation_callback(logger, split, cfg):
    """Return callback(engine, run_details) for generation_hook, or None when logging is off."""
    if not logger.enabled:
        return None
    spec, boundary = cfg.data_spec, cfg.data_spec.formulation == "boundary"

    def callback(engine, run_details):
        step = run_details["generation"][-1] + 1
        if not logger.due(step):
            return
        program = best_of_generation(engine)
        start = perf_counter()
        if boundary:
            # Reads the program only; GPBoundaryModel.from_engine deep-copies the final one.
            model = GPBoundaryModel(program, tuple(spec.feature_cols), tuple(cfg.gp.constants),
                                    cfg.boundary_temperature)
            predictor = GPPredictor(model, spec)
        else:
            predictor = GPPredictor(engine, spec, program=program)
        try:
            details = predictor.predict_prepared(split)
            metrics = metrics_from_predictions(split.labels, details["probability"], y_pred=details["prediction"])
            val_bce = bce_from_logits(split.labels, details["logit"])
        except (ValueError, FloatingPointError):
            # Nonfinite validation outputs: record the step rather than stopping evolution.
            metrics, val_bce = None, float("nan")
        logger.log(step, metrics=metrics, val_bce=val_bce, val_eval_s=perf_counter() - start,
                   best_length=int(program.length_), average_length=float(run_details["average_length"][-1]))

    return callback
