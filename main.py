"""
DISCLAIMER: 
This code was previously part of Joris Heemskerk's prior
work for the Computer Vision course, and is being re-used here.
"""

import argparse
import copy
import logging
import numpy as np
import os
import shutil
import torch
import traceback
import yaml

from resnet_bcos import make_resnet18, make_resnet34, make_resnet50
from resnet_18_baseline import BaselineResNet
from bcos.modules.bcosconv2d import BcosConv2d
from bcos.modules.losses import BinaryCrossEntropyLoss
from bcos.optim import LRSchedulerFactory
from functools import partial
from jsonschema import validate, ValidationError
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from typing import Any

import handle_output

# from dog_cat_audio_dataset import CatDogAudioDataset, LABEL_MAP
from esc_audio_dataset import ESCAudioDataset, LABEL_MAP
from create_logger import create_logger
from config.config_validation_template import CONFIG_TEMPLATE
from data import to_dataloaders
from train import train_cross_validation, train, evaluate, METRICS
from utils import format_result
from tune import tune_job, OptunaPruningCallback, TUNABLE_PARAMS
from augment import SpecAugment, AudioAugment
from visualise import \
    visualise_training, \
    visualise_tuning, \
    plot_confusion_matrix

DATASET_MAPPING = {
    "dirs": ["data/ESC-50-master/audio/"],
    "csv_path": "data/ESC-50-master/meta/esc50.csv",
    "train_folds": [1, 2, 3, 4],
    "test_folds": [5]
}


def _process_job(
    job: dict[str, Any], 
    job_id: int, 
    logger: logging.Logger
)-> None:
    """
    This function executes the jobs according to their description.

    If the job contains a `tune: true`, a Bayesian optimisation with
    hyperband pruning study is run instead of the following sequential
    search.

    For each tunable parameter in job description, spin up several jobs,
    each with different values for that parameter. For all others, the
    first item in the list will be used as to prevent highly 
    computationally expensive grid searches. 

    :param job: Job description, pulled from config.
    :type job: dict[str, Any]
    :param job_id: ID of the current job (for logging).
    :type job_id: int
    :param logger: Logger to log to.
    :type logger: logging.Logger
    """
    ############ Change output dir to specific job folder. #############
    handle_output.OUTPUT_DIR = f"{handle_output.OUTPUT_DIR}job_{job_id}/" if \
        job_id == 0 else "/".join((
                handle_output.OUTPUT_DIR.split("/")[:-2] if 
                    "run" not in handle_output.OUTPUT_DIR else 
                    handle_output.OUTPUT_DIR.split("/")[:-3]
        )) + f"/job_{job_id}/"
    os.makedirs(handle_output.OUTPUT_DIR, exist_ok=True)
    job_output_dir = handle_output.OUTPUT_DIR

    # Hyperparameter tuning via bayesian optimization and hyperband
    # pruning.
    if job.get("tune", False):
        logger.info(
            f"Running job {job_id}] with bayesian optimisation and hyperband"
            "pruning."
        )
        tune_job(
            job=job,
            job_id=job_id,
            build_run_fn=lambda run, trial_number, logger, trial: _process_run(
                run=run,
                run_id=trial_number,
                logger=logger,
                pruning_callback=OptunaPruningCallback(trial, monitor="loss")
            )[run['model']][2],
            logger=logger,
            n_trials=job["n_trials"],
            n_startup_trials=job["n_startup_trials"],
            direction="minimize",
        )
        return

    tune_changes = False
    for key, values in job.items():
        if key in list(TUNABLE_PARAMS.keys()):
            if len(values) > 1:
                tune_changes = True
                tune_results = []
                for i, value in enumerate(values):
                    run_description = copy.deepcopy(job)
                    for tune_key in list(TUNABLE_PARAMS.keys()):
                        if len(job[tune_key]) > 1 and tune_key != key:
                            logger.warning(
                                "Multiple parameters provided for multiple tun"
                                f"able parameters. The values for {tune_key} ("
                                f"{job[tune_key]}) will be ignored and the fir"
                                "st value will be used ({job[tune_key][0]})."
                            )
                        run_description[tune_key] = job[tune_key][0]
                    run_description[key] = value
                    logger.info(
                        f"----- Processing Job {job_id}, Run {i:3.0f}/"
                        f"{len(values)-1:3.0f} -----"
                    )
                    logger.info(f"Run description: {run_description}")
                    results = _process_run(
                        run=run_description,
                        run_id=i, 
                        logger=logger
                    )
                    tune_results.append({k: v[:2] for k, v in results.items()})
                visualise_tuning(
                    tune_param_name=key,
                    tune_param_values=values,
                    tune_results=tune_results,
                    metric_names=list(METRICS.keys()),
                    output_dir=job_output_dir
                ) 
    # If there were no instances of multiple parameters, run as 1 job.
    if not tune_changes:
        run_description = copy.deepcopy(job)
        for tune_key in list(TUNABLE_PARAMS.keys()):
            run_description[tune_key] = job[tune_key][0]
        _process_run(
            run=run_description,
            run_id=None, 
            logger=logger
        )

def _process_run(
    run: dict[str, Any], 
    run_id: int | None, 
    logger: logging.Logger,
    pruning_callback: OptunaPruningCallback | None=None
)-> dict[str, tuple[float, float, float, float]]:
    """
    This function executes the run according to their description.

    :param run: Run description.
    :type run: dict[str, Any]
    :param run_id: ID of the current run (only provide if planning to 
        perform multiple runs).
    :type run_id: int | None
    :param logger: Logger to log to.
    :type logger: logging.Logger
    :param pruning_callback: Optional parameter; used in training for 
        hyperband pruning. (DEFAULT=None)
    :returns: In order, the training and validation MAEs, then the MSEs.
    :rtype: dict[str, tuple[float, float, float, float]]
    """
    ############ Change output dir to specific run folder. #############
    if run_id is not None and pruning_callback is None:
        handle_output.OUTPUT_DIR = \
            f"{handle_output.OUTPUT_DIR}run_{run_id}/" if \
                run_id == 0 else "/".join(
                    handle_output.OUTPUT_DIR.split("/")[:-2]
                ) + f"/run_{run_id}/"
        os.makedirs(handle_output.OUTPUT_DIR, exist_ok=True)
    
    ######################### Save run config. #########################
    with open(f'{handle_output.OUTPUT_DIR}run_config.yml', 'w') as outfile:
        yaml.dump(run, outfile)

    ####################################################################
    #                          Load the data.                          #
    ####################################################################
    dataset = ESCAudioDataset(
        data_dirs=DATASET_MAPPING["dirs"],
        folds=DATASET_MAPPING["train_folds"],
        csv_path=DATASET_MAPPING["csv_path"],
        target_sr=run["sample_rate"],
        duration=run["duration"],
        n_fft=run["n_fft"],
        hop_length=run["hop_length"],
        n_mels=run["n_mels"],
        top_db=run["top_db"],
        pre_transform=AudioAugment(run["sample_rate"]),
        post_transform=SpecAugment()
    )
    logger.debug(f"Dataset size: {len(dataset)}")
    logger.debug(f"Shape of first x element: {dataset[0][0].shape}")
    logger.debug(f"First y element: {dataset[0][1]}")
    test_data = ESCAudioDataset(
        data_dirs=DATASET_MAPPING["dirs"],
        folds=DATASET_MAPPING["test_folds"],
        csv_path=DATASET_MAPPING["csv_path"],
        target_sr=run["sample_rate"],
        duration=run["duration"],
        n_fft=run["n_fft"],
        hop_length=run["hop_length"],
        n_mels=run["n_mels"],
        top_db=run["top_db"]
    )
    logger.debug(f"Test dataset size: {len(test_data)}")

    ####################################################################
    #                      Create the DataLoaders.                     #
    ####################################################################
    indices = list(range(len(dataset)))
    train_idx = indices

    if run["k_folds"] == 1:
        logger.debug(f"Splitting the dataset into {run["train_val_split"]}.")
    
        ####################### Split the data. ########################
        train_idx, val_idx = train_test_split(
            indices, 
            test_size=run["train_val_split"][1],
            random_state=42
        )
    # Normalise based on only the train partition.
    logger.debug("Fitting normalisation.")
    dataset.fit_normalisation(train_idx)
    logger.debug(
        "Normalisation fitted: "
        f"mean={dataset.mean}, std={dataset.std}"
    )
    test_data.mean = dataset.mean
    test_data.std = dataset.std

    train_dataloader, val_dataloader = None, None
    if run["k_folds"] == 1:
        logger.debug("Creating subsets.")
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        val_dataset = torch.utils.data.Subset(dataset, val_idx)
        logger.debug(f"{len(train_dataset) = }, {len(val_dataset) = }")

        ######## Convert DataSet objects to DataLoader objects. ########
        logger.debug("converting to dataloaders")
        train_dataloader, val_dataloader = to_dataloaders(
            [train_dataset, val_dataset], 
            batch_sizes=[run["batch_size"]] * 2, 
            shuffles=[True, False],
            logger=logger,
            num_workers=CONFIG["general"]["num_data_workers"],
            pin_memory=True, # TODO: check if this should be replaced with run["lazy"]
            persistent_workers=True if CONFIG["general"]["num_data_workers"] > 0 else False,
        )

    test_dataloader = to_dataloaders(
        [test_data], 
        batch_sizes=[run["batch_size"]], 
        shuffles=[False],
        logger=logger,
        num_workers=0,
        pin_memory=False,
    )[0]

    ############## Defer task to the individual model(s). ##############
    if run["model"].lower() == "all":
        MODELS = [
            "resnet18_bcos", 
            "resnet34_bcos", 
            "resnet50_bcos", 
            "resnet18_baseline"
        ]
        all_model_results = {model: None for model in MODELS}
        for model_id, model in enumerate(MODELS):
            model_specific_run = copy.deepcopy(run)
            model_specific_run["model"] = model
            model_results = _process_model(
                model_specific_run, 
                model_id, 
                dataset,
                train_dataloader, 
                val_dataloader, 
                test_dataloader,
                logger,
                pruning_callback
            )
            all_model_results[model] = model_results
        # Remove model specific sub-directory before starting next run.
        handle_output.OUTPUT_DIR = "/".join(
            handle_output.OUTPUT_DIR.split("/")[:-2]
        ) + "/"
        return all_model_results
    else:
        return {run["model"]: _process_model(
            run, 
            None, 
            dataset,
            train_dataloader, 
            val_dataloader, 
            test_dataloader, 
            logger,
            pruning_callback
        )}

def _process_model(
    run: dict[str, Any], 
    model_id: int | None, 
    dataset: ESCAudioDataset,
    train_dataloader: DataLoader[Any] | None, 
    val_dataloader: DataLoader[Any] | None,
    test_dataloader: DataLoader[Any],
    logger: logging.Logger,
    pruning_callback: OptunaPruningCallback=None
)-> tuple[float, float]:
    """
    Applies dataset to specific model. 
    
    This function makes it possible to do multiple models per job and 
    run.

    :param run: Run description.
    :type run: dict[str, Any]
    :param model_id: ID of the current model (only use if multiple 
        models are being trained for a run).
    :type model_id: int | None
    :param dataset: the dataset.
    :type dataset: MEGDataset
    :param train_dataloader: Dataloader for training data.
    :type train_dataloader: DataLoader[Any] | None
    :param val_dataloader: Dataloader for validation data.
    :type val_dataloader: DataLoader[Any] | None
    :param test_dataloader: Dataloader for test data.
    :type test_dataloader: DataLoader[Any]
    :param logger: Logger to log to.
    :type logger: logging.Logger
    :param pruning_callback: Optional parameter; used in training for 
        hyperband pruning. (DEFAULT=None)
    :returns: In order, the training and validation accuracies.
    :rtype: tuple[float, float]
    """
    assert run["k_folds"] > 1 or (
        train_dataloader is not None and val_dataloader is not None
    ), "Data not provided for non cross-validation run."

    ############ Change output dir to specific run folder. #############
    if model_id is not None:
        handle_output.OUTPUT_DIR = \
            f"{handle_output.OUTPUT_DIR}{run["model"]}/" if \
                model_id == 0 else "/".join(
                    handle_output.OUTPUT_DIR.split("/")[:-2]
                ) + f"/{run["model"]}/"
        os.makedirs(handle_output.OUTPUT_DIR, exist_ok=True)

    ####################################################################
    #                     Load the (correct) model.                    #
    ####################################################################
    logger.debug(f"Initialising the model ({run['model']})")
    models = {
        "resnet18_bcos": (
            make_resnet18, {
                "logger": logger,
                "num_classes": dataset.get_n_classes(),
                "in_chans" : 1,
                "small_inputs": run["model_params"].get("small_inputs", True),
                "conv_layer": partial(BcosConv2d, b=run["b"], max_out=2),
            }
        ),
        "resnet34_bcos": (
            make_resnet34, {
                "logger": logger,
                "num_classes": dataset.get_n_classes(),
                "in_chans" : 1,
                "small_inputs": run["model_params"].get("small_inputs", True),
                "conv_layer": partial(BcosConv2d, b=run["b"], max_out=2),
            }
        ),
        "resnet50_bcos": (
            make_resnet50, {
                "logger": logger,
                "num_classes": dataset.get_n_classes(),
                "in_chans" : 1,
                "small_inputs": run["model_params"].get("small_inputs", True),
                "conv_layer": partial(BcosConv2d, b=run["b"], max_out=2),
            }
        ),
        "resnet18_baseline": (
            lambda **kwargs: BaselineResNet(kwargs["num_classes"], kwargs["logger"]),
            {"logger": logger, "num_classes": dataset.get_n_classes()}
        )   
    }
    model = None
    for name, (cls, kwargs) in models.items():
        if run['model'].lower() in name:
            model = cls(**kwargs)
            break
    assert model is not None, \
        f"Provided model in config does not exist ({model})."

    logger.debug(f"Model:\n{model}")
    logger.debug("Total number of parameters: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    model = model.to(DEVICE)

    # Should speed up model after epoch 1, but has not proven effective.
    # model = torch.compile(model, backend="aot_eager") 

    ####################################################################
    #                       Initialize optimiser.                      #
    ####################################################################
    logger.debug(f"Initialising the optimiser ({run['optimiser']})")
    optimisers = {
        "adam": (torch.optim.Adam, {
            "params": model.parameters(),
            "lr": run["learning_rate"],
            "weight_decay": run["weight_decay"]
        })
    }
    OPTIMISER = None
    for name, (cls, kwargs) in optimisers.items():
        if run['optimiser'].lower() in name:
            OPTIMISER = cls(**kwargs)
            break
    assert OPTIMISER is not None, \
        "Provided optimiser in config does not exist."

    ####################################################################
    #                         Train the model.                         #
    ####################################################################
    LOSS_FN = BinaryCrossEntropyLoss()

    # Arguments used by both normal training and cross_validation
    arguments = {
        "model" : model,
        "loss_fn" : LOSS_FN,
        "optimiser": OPTIMISER,
        # "scheduler": LRSchedulerFactory(name="cosineannealinglr", epochs=10),
        "n_epochs" : run["n_epochs"],
        "device" : DEVICE,
        "logger" : logger,
        "pruning_callback": pruning_callback
    }

    ################ Don't use k-fold cross validation #################
    if run["k_folds"] == 1:
        # Train it the normal way.
        train_losses, train_metrics, val_losses, val_metrics, model = train(
            train_dataloader=train_dataloader, 
            val_dataloader=val_dataloader,
            **arguments
        )
        train_losses_std, train_metrics_std = None, None
        val_losses_std, val_metrics_std = None, None
    else:
    ################### Use k-fold cross validation ####################
        train_lossess, train_metricss, val_lossess, val_metricss, model=\
            train_cross_validation(
                full_train_dataset=dataset, 
                k_folds=run["k_folds"],
                dataset_to_dataloader_function=lambda dataset: to_dataloaders(
                    datasets=[dataset],
                    batch_sizes=[run["batch_size"]],
                    shuffles=[True],
                    logger=logger,
                    num_workers=CONFIG["general"]["num_data_workers"],
                    pin_memory=True,
                    persistent_workers=True if CONFIG["general"]["num_data_workers"] > 0 else False,
                ),
                **arguments
            )
        
        train_losses = np.mean(train_lossess, axis=0)
        train_losses_std = np.std(train_lossess, axis=0)

        train_metrics = {
            k : np.mean(v, axis=0) for k, v in train_metricss.items()
        }
        train_metrics_std = {
            k : np.std(v, axis=0) for k, v in train_metricss.items()
        }

        val_losses = np.mean(val_lossess, axis=0)
        val_losses_std = np.std(val_lossess, axis=0)
        
        val_metrics = {
            k : np.mean(v, axis=0) for k, v in val_metricss.items()
        }
        val_metrics_std = {
            k : np.std(v, axis=0) for k, v in val_metricss.items()
        }

    # Save the best performing model (based on the validation set).
    model.save(handle_output.OUTPUT_DIR)

    ####################################################################
    #                         Show the results.                        #
    ####################################################################
    logger.critical(
        f"Best training accuracy: {max(train_metrics["accuracy"])}, achieved "
        f"during epoch {np.argmax(train_metrics["accuracy"]) + 1}.\nBest "
        f"validation accuracy: {max(val_metrics["accuracy"])}, achieved during"
        f" epoch {np.argmax(val_metrics["accuracy"]) + 1}."
    )

    best_epoch = np.argmax(val_metrics["accuracy"])
    Stats = (
        f"Stats from best validation epoch (epoch={best_epoch + 1}):\nTrain: "
        f"{format_result(train_metrics, best_epoch, train_metrics_std)} | "
        f"Val: {format_result(val_metrics, best_epoch, val_metrics_std)}"
    )

    logger.critical(Stats)

    metrics_path = os.path.join(
        handle_output.OUTPUT_DIR,
        "metrics_training.md"
    )
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(Stats)

    ################# Plot the predicted and real values ###############
    visualise_training(
        train_losses, 
        train_metrics,
        val_losses, 
        val_metrics,
        handle_output.OUTPUT_DIR,
        train_losses_std,
        train_metrics_std,
        val_losses_std,
        val_metrics_std
    )

    ######## Plot the confusion matrix for train and validation sets #########
    if run["k_folds"] == 1:
        _, val_predictions, val_targets = evaluate(
            dataloader=val_dataloader, 
            model=model,
            device=DEVICE,
            logger=logger,
        )

        plot_confusion_matrix(
            val_targets,
            val_predictions,
            LABEL_MAP.keys(),
            None,
            "validation",
            handle_output.OUTPUT_DIR,
            logger
        )

    else:
        train_dataloader = to_dataloaders(
            [dataset], 
            batch_sizes=[run["batch_size"]], 
            shuffles=[False],
            logger=logger,
            num_workers=0,
            pin_memory=False,
        )[0]

    _, train_predictions, train_targets = evaluate(
        dataloader=train_dataloader, 
        model=model,
        device=DEVICE,
        logger=logger,
    )

    plot_confusion_matrix(
        train_targets,
        train_predictions,
        LABEL_MAP.keys(),
        None,
        "train",
        handle_output.OUTPUT_DIR,
        logger
    )

    ####################################################################
    #                         Test the model.                          #
    ####################################################################
    logger.info("Evaluating on test set.")
    logger.error("ONLY DO THIS AFTER HYPERPAREMTER TUNING!!")
    test_accuracy, test_predictions, test_targets = evaluate(
        dataloader=test_dataloader, 
        model=model,
        device=DEVICE,
        logger=logger,
    )
    logger.critical(f"Test accuracy: {test_accuracy}")

    plot_confusion_matrix(
        test_targets,
        test_predictions,
        LABEL_MAP.keys(),
        None,
        "test",
        handle_output.OUTPUT_DIR,
        logger
    )

    return \
        max(train_metrics["accuracy"]), \
        max(val_metrics["accuracy"]), \
        min(val_losses)

def main()-> None:
    ####################################################################
    #                         Execute all jobs.                        #
    ####################################################################
    for i, job in enumerate(CONFIG['jobs'].values()):
        logger.info(
           f"----- Processing Job {i:3.0f}/"
           f"{len(CONFIG['jobs'].values())-1:3.0f} -----"
        )
        logger.info(f"Job description: {job}")
        # This try-except catches individual job errors and attempts the 
        # next job if one of them crashes.
        try:
            if job in list(CONFIG['jobs'].values())[:i]:
                logger.warning(
                    "A job matching this exact configuration has already " 
                    "been executed. You likely have duplicate job descriptions"
                    ". This job will be skipped."
                )
                continue
            _process_job(
                job=job,
                job_id=i, 
                logger=logger
            )
        except KeyboardInterrupt as e:
            logger.critical(
                "PROGRAM MANUALLY HALTED BY KEYBOARD INTERRUPT "
                "(inside job execution loop)."
            )
            raise KeyboardInterrupt(
                "Keyboard interupt detected, halting program."
            ) from e
        except Exception as e:
            trace = ''.join(
                traceback.format_exception(type(e), e, e.__traceback__)
            )
            logger.error(
                f"Error during handling of job {i} ({job = })\n\tTraceback:\n"
                f"\t{trace}\n\t'''{type(e)}: {e}'''\n"
                "Skipping this job, attempting to execute next job."
            )

if __name__ == "__main__":
    # Parse commandline arguments.
    parser = argparse.ArgumentParser(description='configuration')
    parser.add_argument(
        '-c',
        '--config', 
        dest='config_file_path', 
        type=str, 
        default="config/config.yaml", 
        help="Path to config file. (default: %(default)s)"
    )
    parser.add_argument(
        '-d',
        '--device', 
        dest='device', 
        type=str, 
        default=None, 
        help=
            "Device to run the models on. If not provided, an optimal device "
            "will be determined and used. (default: %(default)s)"
    )
    args = parser.parse_args()

    # Initialise Logger.
    os.makedirs(handle_output.OUTPUT_DIR, exist_ok=True)
    logger = create_logger(
        name="Deep Learning - Assignment 2", 
        output_log_file_name=f"{handle_output.OUTPUT_DIR}process.log"
    )
    logger.info(f"Provided commandline arguments: {args.__dict__}")

    # Seed PyTorch.
    torch.manual_seed(42)

    # Initialise Device.
    if args.device is None:
        DEVICE = torch.accelerator.current_accelerator().type if \
            torch.accelerator.is_available() else "cpu"
    else:
        DEVICE = args.device
    logger.info(f"Using {DEVICE} device")

    # validate the provided config file.
    with open(args.config_file_path, 'r') as stream:
        CONFIG = yaml.safe_load(stream)
    try:
        validate(CONFIG, CONFIG_TEMPLATE)
    except ValidationError as e:
        raise ValidationError(
            "\x1b[31;1mA validation error occurred in the config file" \
            f": {e.message}\x1b[0m"
        ) from e
    shutil.copy(args.config_file_path, handle_output.OUTPUT_DIR + "config.yml")

    ## Execute main. ###################################################
    main()
