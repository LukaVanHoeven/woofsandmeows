import copy
import logging
import optuna
import os
import time
import yaml

import plotly.graph_objects as go

from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from optuna.visualization import plot_parallel_coordinate
from typing import Any, Callable

import handle_output


TUNABLE_PARAMS = {
    "sample_rate": ("int", False), # (type, log_scale)
    "stride": ("int", False),
    "duration": ("int",False),
    "n_fft": ("int", False),
    "hop_length": ("int", False),
    "n_mels": ("int", False),
    "top_db": ("int", False),
    "batch_size": ("int", False),
    "learning_rate": ("float", True),
    "weight_decay": ("int", True),
}

def _build_search_space(
    trial: optuna.Trial,
    job: dict[str, Any],
)-> dict[str, Any]:
    """
    Derive an Optuna search space from the job config. Parameters with a
    single value are fixed and not tuned. Parameters with two values are
    treated as [low, high] and sampled over that range, using a log 
    scale for learning_rate and weight_decay and a linear int scale for 
    the rest. Parameters with three or more values are treated as a 
    categorical list.

    :param trial: Current Optuna trial.
    :type trial: optuna.Trial
    :param job: Job description from config.
    :type job: dict[str, Any]
    :returns: Flat dict of concrete hyperparameter values for this trial
    :rtype: dict[str, Any]
    """
    params: dict[str, Any] = {}
    for key, (dtype, log) in TUNABLE_PARAMS.items():
        values = job.get(key, [])
        n = len(values)

        if n == 0:
            raise ValueError(f"Job is missing required key '{key}'.")
        elif n == 1:
            # Fixed – not a search dimension.
            params[key] = values[0]
        elif n == 2:
            low, high = values[0], values[1]
            if dtype == "float":
                params[key] = trial.suggest_float(key, low, high, log=log)
            else:
                params[key] = trial.suggest_int(
                    key,
                    int(low),
                    int(high),
                    log=log
                )
        else:
            # Categorical list.
            params[key] = trial.suggest_categorical(key, values)
    
    return params

class OptunaPruningCallback:
    """
    A callback class for hyperband pruning.
    """

    def __init__(self, trial: optuna.Trial, monitor: str = "loss")-> None:
        """
        Initialize the pruning callback.

        :param trial: Current optuna trial.
        :type trial: optuna.Trial
        :param monitor: Which metric the callback class is based on.
        :type monitor: str
        """
        self.trial = trial
        self.monitor = monitor
        self._epoch = 0

    def __call__(self, val_loss: float)-> None:
        """
        Report the validation loss to Optuna and prune the trial if
        needed.

        :param val_loss: Validation loss of the current epoch.
        :type val_loss: float.
        """
        self.trial.report(val_loss, step=self._epoch)
        self._epoch += 1
        if self.trial.should_prune():
            raise optuna.TrialPruned(
                f"Trial pruned at epoch {self._epoch - 1} "
                f"(val_{self.monitor}={val_loss:.4f})."
            )

def tune_job(
    job: dict[str, Any],
    job_id: int,
    build_run_fn: Callable[
        [
            dict[str, Any],
            int,
            logging.Logger,
            optuna.Trial | None
        ], 
        float
    ],
    logger: logging.Logger,
    n_trials: int,
    n_startup_trials: int,
    direction: str,
    study_name: str | None = None,
    storage: str | None = None,
)-> optuna.Study:
    """
    Run an Optuna study for one job.

    :param job: Job description, pulled from config.
    :type job: ditct[str, Any]
    :param job_id: ID of the current job (for logging and directory 
        naming).
    :type job_id: int
    :param build_run_fn: Callable function that runs a trial and returns
        the metric optuna is optimising.
    :type build_run_fn: Callable
    :param logger: Logger to log to.
    :type logger: logging.Logger
    :param n_trials: Total number of trials.
    :type n_trials: int
    :param n_startup_trials: Warm up trials before optimisation starts.
    :type n_startup_trials: int
    :param direction: The direction the optimisation tries to move the
        metric toward ("maximize" or "minimize").
    :type direction: str
    :param study_name: Optional name for the Optuna study.
    :type study_name: str
    :param storage: Optional Optuna storage URL for persistence
        (e.g. "sqlite:///study.db").
    :type storage: str
    :returns: Completed Optuna study.
    :rtype: optuna.Study
    """
    study_name = study_name or f"job_{job_id}_tuning"

    sampler = TPESampler(
        n_startup_trials=n_startup_trials,
        seed=42,
        multivariate=True,
    )
    
    pruner = HyperbandPruner(
        min_resource=1,
        max_resource=job["n_epochs"],
        reduction_factor=3,
    )

    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )

    tuning_output_dir = handle_output.OUTPUT_DIR

    def _log_progress_bar(trial_i, n_trials, start_time, logger):
        trial_progress = int((trial_i + 1) / n_trials * 50)

        elapsed = time.perf_counter() - start_time
        s_per_trial = elapsed / (trial_i + 1)
        estimate = s_per_trial * n_trials

        trail_progress_message = \
            f"\033[36mTrial {trial_i}  {round((trial_i + 1) / n_trials * 100, 2)}%|" \
            f"{'#' * trial_progress}{' ' * (50 - trial_progress)}| " \
            f"{trial_i + 1}/{n_trials} [{int(elapsed // 60)}:" \
            f"{int(elapsed % 60)}<{int(estimate // 60)}:" \
            f"{int(estimate % 60)}, {s_per_trial:.2f}s/it]\033[37m"
        logger.info(trail_progress_message)

    def objective(trial: optuna.Trial) -> float:
        """
        Run a single trial.

        :param trial: Current trial.
        :type trial: optuna.Trial
        :return: Score of the trial to be optimised.
        :rtype: float
        """
        handle_output.OUTPUT_DIR = f"{tuning_output_dir}trial_{trial.number}/"
        os.makedirs(handle_output.OUTPUT_DIR, exist_ok=True)

        sampled = _build_search_space(trial, job)

        run = copy.deepcopy(job)
        run.update(sampled)

        logger.info(f"Trial {trial.number} | params: {sampled}")

        try:
            score = build_run_fn(run, trial.number, logger, trial)
        except optuna.TrialPruned:
            _log_progress_bar(trial.number, n_trials, start_time, logger)
            raise
        except Exception as e:
            logger.error(
                f"Trial {trial.number} failed with {type(e).__name__}: {e}"
            )
            raise optuna.exceptions.TrialPruned() from e

        logger.info(
            f"Trial {trial.number} finished | {direction} metric = {score}"
        )
        _log_progress_bar(trial.number, n_trials, start_time, logger)

        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    start_time = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    handle_output.OUTPUT_DIR = tuning_output_dir

    logger.info(
        f"Best trial: #{study.best_trial.number} | "
        f"value={study.best_trial.value:.4f} | "
        f"params={study.best_trial.params}"
    )

    _save_study_summary(study, tuning_output_dir, logger)
    return study

def _save_study_summary(
    study: optuna.Study,
    output_dir: str,
    logger: logging.Logger,
)-> None:
    """
    Write a trial overview CSV, a best-params YAML and a parallel
    coordinates plot reflecting the parameters used in the study to 
    *output_dir*.

    :param study: The hyperparameter optimization study.
    :type study: optuna.Study
    :param output_dir: Output directory to save files to.
    :type output_dir: str
    :param logger: Logger to log to.
    :type logger: logging.Logger
    """
    df = study.trials_dataframe()
    csv_path = os.path.join(output_dir, "optuna_trials.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Hyperparameter optimisation trials saved to {csv_path}")

    best_path = os.path.join(output_dir, "best_params.yml")
    with open(best_path, "w") as f:
        yaml.dump(
            {
                "best_trial": study.best_trial.number,
                "best_value": study.best_trial.value,
                "best_params": study.best_trial.params,
            },
            f,
        )
    logger.info(f"Best params saved to {best_path}")
    
    param_cols = [
        f"params_{key}"
        for key in TUNABLE_PARAMS
        if f"params_{key}" in df.columns
    ]
    plot_df = df[df["state"] == "COMPLETE"].dropna(
        subset=param_cols + ["value"]
    ).copy()
 
    if plot_df.empty:
        logger.warning("No completed trials to plot in parallel coordinates.")
        return
 
    dimensions = []
    for col in param_cols:
        values = plot_df[col].tolist()
        dimensions.append(
            go.parcoords.Dimension(
                values=values,
                label=col.replace("params_", "").replace("_", " ").title(),
                range=[min(values), max(values)],
                tickformat=".2e" if TUNABLE_PARAMS[
                    col.replace("params_", "")
                ][0] == "float" else "d",
            )
        )
 
    objective_values = plot_df["value"].tolist()
    fig = go.Figure(
        go.Parcoords(
            line=dict(
                color=objective_values,
                colorscale="Plasma_r",
                colorbar=dict(title="Validation Loss", tickformat=".2e"),
                showscale=True,
                cmin=min(objective_values),
                cmax=max(objective_values),
            ),
            dimensions=dimensions,
        )
    )
    
    fig.update_layout(
        margin=dict(l=80, r=80, t=40, b=40),
    )
 
    plot_path = os.path.join(output_dir, "parallel_coordinates.png")
    fig.write_image(plot_path)
    logger.info(f"Parallel coordinates plot saved to {plot_path}")
