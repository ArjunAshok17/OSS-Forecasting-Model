"""
    @brief Modeling framework w/ testing built-in for switching out model types, 
    testing accuracies with different methods, and augmenting data prior to 
    testing.
    @author Arjun Ashok (arjun3.ashok@gmail.com)
    @creation-date October 2023
"""


# ---------------- environment setup ---------------- #
# model definition & deployment
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

# auxiliary functionality for user reporting
from tqdm import tqdm
from sklearn.metrics import classification_report, precision_recall_fscore_support

# interfacing w/ system
import sys
from typing import Iterable, Any
from itertools import product

# abstractions
import general_utils as util
from modeldata import *
from perfdata import *
from tsmodel import *


# ---------------- testing strategies + helper methods ---------------- #
def combine_intervals(int1: dict, int2: dict):
    """
        Combine two interval dictionaries
    """
    
    months = set(int1.keys()) | set(int2.keys())
    return {m: int1.get(m, []) + int2.get(m, []) for m in months}


# ---------------- modeling script ---------------- #
def modeling(params_dict: dict, args_dict: dict):
    """
        Wraps modeling functionality.

        @param params_dict: params dictionary, centralized args
        @param args_dict: defines 'strategy' and any hyperparameters to override
    """

    # load all data
    md = ModelData(transfer_strategy=args_dict["strategy"])

    # train model & test
    hyperparams = {"input_size": md.tensors["train"]["x"][0].shape[1]}
    hyperparams.update(args_dict.get("hyperparams", dict()))

    model = TimeSeriesModel(model_arch=args_dict.get("model-arch", "Bi-LSTM"), 
                            hyperparams=hyperparams)
    model.train_model(md)
    model.test_model(md)

    # reporting
    model.report(save=False)
    perf_db = PerfData()
    perf_db.add_entry(
        md.transfer_strategy,
        model_arch=model.model_arch,
        preds=model.preds,
        targets=model.targets,
        intervaled=md.is_interval["test"]
    )

    if md.is_interval["test"]:
        PerfData().perf_vs_time(md.transfer_strategy, model.model_arch)

    # monthly predictions
    if args_dict.get("monthly-preds", False):
        model.monthly_predictions(
            **args_dict["monthly-preds"]
        )


def monthly_predictions(incubator: str, strat: str, args_dict: dict[str, Any],
                        versions: dict[str, str | int]=None, gen_perf: bool=True,
                        hyperparams: dict[str, int | float]=None) -> Optional[pd.DataFrame]:
    """
        Generates the predictions for every month of a given incubator's 
        projects. This will also eventually form the backbone of the 
        soft-probability feature. The strategy takes all incubators 

        @param incubator: the incubator to load from
        @param versions: 
    """

    # check args
    if versions is None:
        versions = util._load_params()["default-versions"][incubator]
        versions = dict(zip(["tech", "social"], versions))
    model_arch = args_dict.get("model-arch", "Bi-LSTM")

    # setup tracking
    train_strat = strat.split("-->")[0].strip()
    util._log(f"\n\n<Using {train_strat} to train>", log_type="none", output="file")
    perf_db = PerfData()
    full_preds = dict()
    full_targets = dict()

    # load projects
    nd = NetData(incubator, versions, gen_tensors=False)
    projects_set = nd.base_projects

    # branch if we can train only one model
    if (util._load_params()["abbreviations"][incubator]) not in train_strat:
        # loading data for model
        util._log("using one model", "log")
        util._log("loading ModelData for monthly predictions")
        md = ModelData(
            transfer_strategy=f"{strat}*"
        )
        hyperparams = {"input_size": md.tensors["train"]["x"][0].shape[1]}
        hyperparams.update(args_dict.get("hyperparams", dict()))
        nd = NetData(
            incubator=incubator,
            options=md.options["test"][incubator],
            gen_tensors=False
        )
        
        # training model
        model = TimeSeriesModel(model_arch=model_arch, hyperparams=hyperparams)
        model.train_model(md)

        # decomposing projects into individual csvs
        util._log("interval testing on the data")
        projects_set = nd.base_projects

        for proj in tqdm(projects_set):
            # generate predictions for this project
            tensors_info = nd._interval_tensors(subset={proj})
            results = model._test_results_intervaled(
                X_dict=tensors_info["x"],
                y_dict=tensors_info["y"],
                raw_prob=True
            )
            
            # setup df
            df = pd.DataFrame.from_dict(results["preds"], orient="index", columns=["close"])
            if df.shape[0] == 0:
                continue
            
            # reconfiguration
            df.rename(index={"all": f"{df.shape[0]}-months"}, inplace=True)
            df["month"] = df.index.str.split("-").str[0].astype(int)
            df = df[["month", "close"]]
            df = df.sort_values(by="month")

            # export
            util._check_dir(f"../predictions/{incubator}/")
            df.to_csv(f"../predictions/{incubator}/{proj}.csv", index=False)

            # check result
            final_pred = df.iloc[-1, 1]
            rounded_pred = float(round(final_pred))
            final_target = results["targets"]["all"][0]

            if final_target == -1:
                util._log(f"incubating: {proj}, predicted {rounded_pred} w/ {final_pred}", "warning", output="file")
            elif rounded_pred != final_target:
                util._log(f"MIS-PREDICTION: expected {final_target}, got {rounded_pred} with {final_pred} for {proj}", "warning", output="file")
            else:
                util._log(f"Correct Prediction: expected {final_target}, got {final_pred} for {proj}", "log", output="file")

            preds = {m: round(pred[0]) for m, pred in results["preds"].items()}
            targets = results["targets"]

            for m, p in preds.items():
                if m not in full_preds:
                    full_preds[m] = list()
                    full_targets[m] = list()
                full_preds[m].append(p)
                full_targets[m].append(targets[m])
        
        # gen visual
        if gen_perf:
            perf_db.add_entry(
                strat,
                model_arch=model.model_arch,
                preds=full_preds,
                targets=full_targets,
                intervaled=True
            )
            perf_db.perf_vs_time(transfer_strategy=strat, model_arch=model_arch, stop_month=250)
        return


    # iterate every project
    projects_set = projects_set & (nd.project_status["graduated"] | nd.project_status["retired"])
    for proj in projects_set:
        # report
        util._log(f"Monthly Predictions for {proj}", "new")

        # load in data
        util._log("loading ModelData for monthly predictions")
        md = ModelData(
            transfer_strategy=f"{strat}*",
            predict_project={incubator: proj}
        )
        hyperparams = {"input_size": md.tensors["train"]["x"][0].shape[1]}
        hyperparams.update(args_dict.get("hyperparams", dict()))
        
        # testing
        util._log("interval testing on the data")
        model = TimeSeriesModel(model_arch=model_arch, hyperparams=hyperparams)
        model.train_model(md)
        model.test_model(md, raw_prob=True)
        preds = model.preds

        # monthly results, shape into the month: prediction we want
        df = pd.DataFrame.from_dict(preds, orient="index", columns=["close"])
        if df.shape[0] == 0:
            continue

        df.rename(index={"all": f"{df.shape[0]}-months"}, inplace=True)
        df["month"] = df.index.str.split("-").str[0].astype(int)
        df = df[["month", "close"]]
        df = df.sort_values(by="month")

        # export
        util._check_dir(f"../predictions/{incubator}/")
        df.to_csv(f"../predictions/{incubator}/{proj}.csv", index=False)

        # check result
        final_pred = df.iloc[-1, 1]
        rounded_pred = float(round(final_pred))
        final_target = model.targets["all"][0]

        if final_target == -1:
            util._log(f"incubating: {proj}, predicted {rounded_pred} w/ {final_pred}", "warning", output="file")
        elif rounded_pred != final_target:
            util._log(f"MIS-PREDICTION: expected {final_target}, got {rounded_pred} with {final_pred} for {proj}", "warning", output="file")
        else:
            util._log(f"Correct Prediction: expected {final_target}, got {final_pred} for {proj}", "log", output="file")

        preds = {m: [round(p) for p in pred] for m, pred in model.preds.items()}
        targets = model.targets

        for m, p in preds.items():
            if m not in full_preds:
                full_preds[m] = list()
                full_targets[m] = list()
            full_preds[m].append(p)
            full_targets[m].append(targets[m])

    # generate visualization
    if gen_perf:
        perf_db.add_entry(
            strat,
            model_arch=model.model_arch,
            preds=preds,
            targets=model.targets,
            intervaled=True
        )
        perf_db.perf_vs_time(transfer_strategy=strat, model_arch=model_arch, stop_month=250)


def full_trials(params_dict: dict[str, Any], num_repeats: int=10, model_arch: str="Bi-LSTM") -> None:
    """
        Generates all trials.
    """

    # schema
    versions = {
        "apache": [
            ("A", 0, 0),
            ("A", 1, 1)
        ],
        "github": [
            ("G", 0, 0),
            ("G", 1, 1),
            ("G", 1, 2),
            ("G", 2, 3),
            ("G", 3, 4)
        ]
    }
    frameworks = {
        "self": "{}-{}-{}^ --> {}-{}-{}^^",
        "split": "{}-{}-{} --> {}-{}-{}",
        "even-mix": "{}-{}-{}^ + {}-{}-{}^ --> {}-{}-{}^^ + {}-{}-{}^^",
        "uneven-mix": "{}-{}-{}^ + {}-{}-{} --> {}-{}-{}^^"
    }

    # generate all trials
    version_combos = list(product(*versions.values()))
    version_combos_extended = version_combos + [(t[1], t[0]) for t in version_combos]

    # define wrappers
    def self_trials():
        self_combos = [v for version in versions.values() for v in version]
        for version_combo in self_combos:
            # run modeling program
            args_dict = {
                "strategy": frameworks["self"].format(
                    version_combo[0],
                    version_combo[1],
                    version_combo[2],
                    version_combo[0],
                    version_combo[1],
                    version_combo[2]
                ),
                "model-arch": model_arch
            }
            
            for i in range(num_repeats):
                modeling(params_dict=params_dict, args_dict=args_dict)

    def split_trials():
        for version_combo in version_combos_extended:
            # run modeling program
            args_dict = {
                "strategy": frameworks["split"].format(
                    version_combo[0][0],
                    version_combo[0][1],
                    version_combo[0][2],
                    version_combo[1][0],
                    version_combo[1][1],
                    version_combo[1][2],
                    version_combo[0][0],
                    version_combo[0][1],
                    version_combo[0][2],
                    version_combo[1][0],
                    version_combo[1][1],
                    version_combo[1][2]
                ),
                "model-arch": model_arch
            }

            for i in range(num_repeats):
                modeling(params_dict=params_dict, args_dict=args_dict)

    def even_mix_trials():
        for version_combo in version_combos:
            # run modeling program
            args_dict = {
                "strategy": frameworks["even-mix"].format(
                    version_combo[0][0],
                    version_combo[0][1],
                    version_combo[0][2],
                    version_combo[1][0],
                    version_combo[1][1],
                    version_combo[1][2],
                    version_combo[0][0],
                    version_combo[0][1],
                    version_combo[0][2],
                    version_combo[1][0],
                    version_combo[1][1],
                    version_combo[1][2]
                ),
                "model-arch": model_arch
            }
            
            for i in range(num_repeats):
                modeling(params_dict=params_dict, args_dict=args_dict)
    
    def uneven_mix_trials():
        for version_combo in version_combos_extended:
            # run modeling program
            args_dict = {
                "strategy": frameworks["uneven-mix"].format(
                    version_combo[0][0],
                    version_combo[0][1],
                    version_combo[0][2],
                    version_combo[1][0],
                    version_combo[1][1],
                    version_combo[1][2],
                    version_combo[0][0],
                    version_combo[0][1],
                    version_combo[0][2]
                ),
                "model-arch": model_arch
            }

            for i in range(num_repeats):
                modeling(params_dict=params_dict, args_dict=args_dict)

    # calls
    self_trials()
    split_trials()
    even_mix_trials()
    uneven_mix_trials()


def breakdown(params_dict: dict[str, Any], args_dict: dict[str, Any]) -> None:
    """
        Given a set of options, generates a breakdown surrounding those options.
        NOTE: follows a very similar structure to the full trials

        @param params_dict: centralized parameter structure
        @param args_dict: arguments passed in; should include some specification 
                          of the options selected (as a str of characters to 
                          append)
    """

    # unpack args
    options_str = args_dict["options"]
    model_arch = args_dict.get("model-arch", "Bi-LSTM")
    num_repeats = args_dict.get("trials", 2)
    hyperparams = args_dict.get("hyperparams", dict())

    # schema
    versions = {
        "apache": [
            ("A", 1, 1)
        ],
        "github": [
            ("G", 3, 4)
        ],
        "eclipse": [
            ("E", 1, 1)
        ]
    }
    placeholder = f"{{}}-{{}}-{{}}"
    frameworks = {
        "self": f"{placeholder}^{options_str} --> {placeholder}^^{options_str}",
        "split": (
            " + ".join([f"{placeholder}{options_str}" for _ in range(len(versions) - 1)]) + 
            f" --> {placeholder}{options_str}"
        ),
        "even-mix": (
            " + ".join([f"{placeholder}^{options_str}" for _ in range(len(versions))]) + 
            " --> " + 
            " + ".join([f"{placeholder}^^{options_str}" for _ in range(len(versions))])
        ),
        "uneven-mix": (
            " + ".join([f"{placeholder}{options_str}" for _ in range(len(versions) - 1)]) + 
            f"{placeholder}^{options_str} --> {placeholder}^^{options_str}"
        )
    }

    # generate all trials
    version_combos = list(product(*versions.values()))
    version_combos_extended = version_combos + [(t[1], t[0]) for t in version_combos]

    # define wrappers
    def self_trials():
        self_combos = [v for version in versions.values() for v in version]
        for version_combo in self_combos:
            # run modeling program
            args_dict = {
                "strategy": frameworks["self"].format(
                    version_combo[0],
                    version_combo[1],
                    version_combo[2],
                    version_combo[0],
                    version_combo[1],
                    version_combo[2]
                ),
                "model-arch": model_arch,
                "hyperparams": hyperparams
            }
            
            for i in range(num_repeats):
                modeling(params_dict=params_dict, args_dict=args_dict)

    def split_trials():
        for version_combo in version_combos_extended:
            # run modeling program
            args_dict = {
                "strategy": frameworks["split"].format(
                    version_combo[0][0],
                    version_combo[0][1],
                    version_combo[0][2],
                    version_combo[1][0],
                    version_combo[1][1],
                    version_combo[1][2],
                    version_combo[0][0],
                    version_combo[0][1],
                    version_combo[0][2],
                    version_combo[1][0],
                    version_combo[1][1],
                    version_combo[1][2]
                ),
                "model-arch": model_arch,
                "hyperparams": hyperparams
            }

            for i in range(num_repeats):
                modeling(params_dict=params_dict, args_dict=args_dict)

    def even_mix_trials():
        for version_combo in version_combos:
            # run modeling program
            args_dict = {
                "strategy": frameworks["even-mix"].format(
                    version_combo[0][0],
                    version_combo[0][1],
                    version_combo[0][2],
                    version_combo[1][0],
                    version_combo[1][1],
                    version_combo[1][2],
                    version_combo[0][0],
                    version_combo[0][1],
                    version_combo[0][2],
                    version_combo[1][0],
                    version_combo[1][1],
                    version_combo[1][2]
                ),
                "model-arch": model_arch,
                "hyperparams": hyperparams
            }
            
            for i in range(num_repeats):
                modeling(params_dict=params_dict, args_dict=args_dict)

    def uneven_mix_trials():
        for version_combo in version_combos_extended:
            # run modeling program
            args_dict = {
                "strategy": frameworks["uneven-mix"].format(
                    version_combo[0][0],
                    version_combo[0][1],
                    version_combo[0][2],
                    version_combo[1][0],
                    version_combo[1][1],
                    version_combo[1][2],
                    version_combo[0][0],
                    version_combo[0][1],
                    version_combo[0][2]
                ),
                "model-arch": model_arch,
                "hyperparams": hyperparams
            }

            for i in range(num_repeats):
                modeling(params_dict=params_dict, args_dict=args_dict)

    # run trials
    self_trials()
    split_trials()
    even_mix_trials()
    uneven_mix_trials()

    # generate reports
    p = PerfData()
    p.subset_breakdown(options=options_str)
    p.comparison(field="transfer_strategy")


if __name__ == "__main__":
    # forward parameters to main
    params_dict = util._load_params()
    args_dict = util._parse_input(sys.argv)

    trial_type = args_dict.get("trial-type", "regular")
    match trial_type:
        case "regular":
            for i in range(args_dict.get("trials", 1)):
                modeling(params_dict=params_dict, args_dict=args_dict)
        
        case "full":
            full_trials(params_dict=params_dict, **args_dict)

        case "monthly-preds":
            monthly_predictions(
                incubator=args_dict.get("incubator", "github"),
                args_dict=args_dict,
                strat=args_dict.get("strategy", "A-1-1 --> G-3-4"),
                **args_dict.get("trial-args", dict())
            )

        case "breakdown":
            breakdown(
                params_dict=params_dict,
                args_dict=args_dict
            )
        
        case _:
            print(":(")


