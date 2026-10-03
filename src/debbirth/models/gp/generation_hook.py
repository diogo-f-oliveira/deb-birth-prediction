"""Per-generation callback for gplearn engines (T08D).

gplearn 0.4.3 has no generation callback. The audited BaseSymbolic.fit calls
self._verbose_reporter(run_details) after recording each generation, while
self._programs[-1] is that generation's population (and once beforehand,
with run_details=None, to print a header). While a callback is installed the
mixin routes those calls to it instead of printing gplearn's table. The
callback only reads the population, so evolution and the final program are
unchanged.
"""
import numpy as np

from .boundary_prototype import check_backend_contract


class GenerationCallbackMixin:
    _generation_callback = None  # installed only for the duration of one fit

    def _verbose_reporter(self, run_details=None):
        callback = self._generation_callback
        if callback is None:
            return super()._verbose_reporter(run_details)
        if run_details is not None:
            callback(self, run_details)


def fit_with_generation_callback(engine, callback, *args, **kwargs):
    """engine.fit(*args, **kwargs), calling callback(engine, run_details) after each generation."""
    if callback is None:
        return engine.fit(*args, **kwargs)
    if not isinstance(engine, GenerationCallbackMixin):
        raise TypeError("The engine does not support generation callbacks.")
    check_backend_contract()  # fit's reporter call and population bookkeeping are audited
    verbose = engine.verbose
    engine._generation_callback = callback
    # fit calls the reporter only when verbose is truthy; 1 keeps joblib quiet.
    engine.verbose = verbose or 1
    try:
        return engine.fit(*args, **kwargs)
    finally:
        engine.verbose = verbose
        del engine._generation_callback  # back to the class default; never pickled


def best_of_generation(engine):
    """The current generation's lowest raw-loss program: gplearn's own final-program rule."""
    if engine._metric.greater_is_better:
        raise ValueError("Expected a minimized loss metric.")
    population = engine._programs[-1]
    # The same np.argmin call as gplearn, including its tie and NaN handling.
    return population[np.argmin([program.raw_fitness_ for program in population])]
