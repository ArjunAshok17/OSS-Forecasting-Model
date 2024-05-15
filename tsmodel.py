"""
    @brief defines the model class for interfacing with a model's info
    @author Arjun Ashok (arjun3.ashok@gmail.com).
    @acknowledgements Nafiz I. Khan, Dr. Likang Yin
    @creation-date January 2024
    @version 0.1.0
"""

# Imports
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

import json
import re
from typing import Any, Optional
from dataclasses import dataclass, field

from netdata import *
from perfdata import *


# Model Architectures
# interface [abstract class] for models
"""
# Since the language isn't type strict, we can pretend to polymorphically call 
# __init__ and forward
class forecast_model(ABC):
    @abstractmethod
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        pass

    @abstractmethod
    def forward(self, x):
        pass
"""

## Bidirectional LSTM
class BRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BRNN, self).__init__()
        self.hidden_size = hidden_size 
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, \
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)
        self.softmax = nn.Softmax(dim=1)
        
        
    def forward(self, x):
        if torch.isnan(x).any():
            # set NaN to zero
            x[x != x] = 0
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        out = self.softmax(x)
        return out

## Bidrectional LSTM w/ Sigmoid Output
class S_BRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(S_BRNN, self).__init__()
        self.hidden_size = hidden_size 
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, \
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)
        self.sigmoid = nn.Sigmoid(dim=1)
        
        
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        out = self.sigmoid(x)
        return out
    
## Bidirectional LSTM w/ Batch Normalization
class BN_BRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BN_BRNN, self).__init__()
        self.hidden_size = hidden_size 
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, \
                            batch_first=True, bidirectional=True)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size*2, num_classes)
        self.softmax = nn.Softmax(dim=1)
        
        
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[:,-1,:])
        out = self.softmax(x)
        return out

## Transformer
class TNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        # setup model architecture
        super(TNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.nheads = 2
        
        # transformer + linear for classification
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=input_size,
            nhead=self.nheads,
            dim_feedforward=hidden_size,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            self.transformer_layer,
            num_layers=num_layers
        )
        # self.transformer=  nn.Transformer(
        #     d_model=self.input_size * self.nheads,
        #     nhead=self.nheads,
        #     dim_feedforward=hidden_size,
        #     batch_first=True
        # )
        self.fc = nn.Linear(input_size, num_classes)

        # softmax output
        self.softmax = nn.Softmax(dim=1)
        
        
    def forward(self, x):
        # reshape and propagate
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        
        # classifier output, use last token's representation for output
        x = self.fc(x[-1, :, :])
        x = self.softmax(x)
        
        return x

## Regressor (from Nafiz)
class Regressor(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2):
        super(Regressor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Utility
def load_hyperparams(new_hp: dict[str, Any]) -> dict[str, Any]:
    """
        Overwrites any default hyperparams with any specific hyperparams defined.
    """

    # default
    hyperparams = {
        "input_size": 14,
        "hidden_size": 64,
        "num_classes": 2,
        "learning_rate": 0.0001,
        "batch_size": 512,
        "num_epochs": 10,
        "num_layers": 1
    }

    # overwrite
    if new_hp != None:
        for k, v in new_hp.items():
            hyperparams[k] = v
    
    # export
    return hyperparams
    

# Class
@dataclass
class TimeSeriesModel:
    # data
    model_arch: str = field(default="Bi-LSTM")              # model architecture
    hyperparams: dict[str, Any] = field(default=None)       # hyperparameters
    bias_weights: bool = field(default=False)               # control weights for the loss fn
    report_name: str = field(init=False)                    # export report as this name
    is_interval: dict[str, bool] = field(
        default_factory=lambda: dict()
    )                                                       # info on whether or not the tensors are intervaled
    device: Any = field(init=False, repr=False)             # device to use for training
    model: Any = field(init=False, repr=False)              # model
    loss_fc: Any = field(init=False, repr=False)            # loss function
    optimizer: Any = field(init=False, repr=False)          # optimizer function
    preds: Any = field(init=False, repr=False)              # predictions
    targets: Any = field(init=False, repr=False)            # targets

    # internal utility
    def _gen_model_(self) -> None:
        """
            Generates the model architecture based on the options given.
        """

        # router
        match self.model_arch:
            case "Bi-LSTM":
                util._log("Model Chosen :: Bidirectional LSTM", "new")
                self.model = BRNN(
                    self.hyperparams["input_size"],
                    self.hyperparams["hidden_size"],
                    self.hyperparams["num_layers"],
                    self.hyperparams["num_classes"]
                ).to(self.device)
            
            case "S_Bi-LSTM":
                util._log("Model Chosen :: Sigmoid Bidirectional LSTM", "new")
                self.model = S_BRNN(
                    self.hyperparams["input_size"],
                    self.hyperparams["hidden_size"],
                    self.hyperparams["num_layers"],
                    self.hyperparams["num_classes"]
                ).to(self.device)
            
            case "BN_Bi-LSTM":
                util._log("Model Chosen :: Batch Normalized Bidirectional LSTM", "new")
                self.model = BN_BRNN(
                    self.hyperparams["input_size"],
                    self.hyperparams["hidden_size"],
                    self.hyperparams["num_layers"],
                    self.hyperparams["num_classes"]
                ).to(self.device)

            case "Transformer":
                util._log("Model Chosen :: Transformer", "new")
                self.model = TNN(
                    self.hyperparams["input_size"],
                    self.hyperparams["hidden_size"],
                    self.hyperparams["num_layers"],
                    self.hyperparams["num_classes"]
                ).to(self.device)
            
            case "Regressor":
                util._log("Model Chose :: Regressor", "new")
                self.model = Regressor(
                    self.hyperparams["input_size"],
                    32,
                    16
                ).to(self.device)
                self.hyperparams["num_epochs"] = 120
                self.hyperparams["learning_rate"] = 1e-5
                self.hyperparams["batch_size"] = 32


            case _:
                util._log(f"model architecture `{self.model_arch}` undefined", "error")
                exit(1)


    def _check_device(self) -> None:
        """
            Attempts to use CUDA enabled hardware if possible.
        """

        # check & report
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"using ***{self.device}*** for training...")


    def _gen_report_name(self) -> None:
        """
            Generates a unique identifier for the model for later retrieval of 
            info. Uses:
                - model_arch
                - date & time
        """

        import time
        self.report_name = f"{self.model_arch}-{time.time()}"


    def __post_init__(self):
        # generate model archs
        util._log("Model Setup", "new")
        self.hyperparams = load_hyperparams(self.hyperparams)
        self._check_device()
        self._gen_model_()

        # optimizers
        # class_weights = compute_class_weight("balanced", np.unique(y), y.numpy())
        class_weights = None if not self.bias_weights else \
            torch.tensor([1 - 10 / 441, 1 - 431/441], dtype=torch.float).to(self.device)
        
        if "regressor" in self.model_arch:
            self.loss_fc = nn.MSELoss()
        else:
            self.loss_fc = nn.CrossEntropyLoss(
                weight=class_weights
            )
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.hyperparams["learning_rate"]
        )


    def _test_results_regular(self, X, y, raw_prob: bool=False) -> dict[str, list[Any]]:
        """
            Regular trials.
        """

        # setup
        preds_list = np.empty(0)
        targets_list = np.empty(0)

        # generate prediction
        for data, target in zip(X, y):
            # transform data to use
            data = data.to(self.device)# .squeeze(1)
            data = data.reshape(1, data.shape[0], -1)

            targets = target.to(self.device)

            if raw_prob:
                preds = self.model(data)[:, 1].to(self.device)  # grab probability of success
            else:
                preds = torch.argmax(self.model(data), dim=1).to(self.device)

            # concatenate lists
            preds_list = np.concatenate((preds_list, preds.cpu().detach().numpy()))
            targets_list = np.concatenate((targets_list, targets.cpu().detach().numpy()))

        # reshape
        preds_list.flatten()
        targets_list.flatten()

        return {"preds": preds_list, "targets": targets_list}


    def _test_results_intervaled(self, X_dict, y_dict, raw_prob: bool=False) -> dict[str, dict[str, list[Any]]]:
        """
            Defines the testing strategy specifically for the intervaled 
            trials, returning information for each month.
        """

        # setup
        preds_dict = dict()
        targets_dict = dict()

        # generate prediction by month
        for month in tqdm(X_dict):
            # unpack
            X = X_dict[month]
            y = y_dict[month]

            # gen preds
            month_results = self._test_results_regular(X, y, raw_prob)
            preds_dict[month] = month_results["preds"]
            targets_dict[month] = month_results["targets"]

        # export
        return {"preds": preds_dict, "targets": targets_dict}


    # external utility
    def train_model(self, md, soft_prob_model: Any=None, save_epochs: bool=False) -> None:
        """
            Trains the model on the necessary data.

            @param md: ModelData Object to train on
            @param soft_prob_model: **FUTURE** will introduce soft probabilities 
                                    for training on intervaled data
        """

        # track losses
        losses = {}

        # interval training
        self.is_interval["train"] = soft_prob_model != None

        # training
        for epoch in range(self.hyperparams["num_epochs"]):
            # setup
            losses[epoch] = list()

            for data, target in tqdm(list(zip(md.tensors["train"]["x"], md.tensors["train"]["y"]))):
                # transform data for training
                data = data.to(self.device)# .squeeze(1)
                data = data.reshape(1, data.shape[0], -1)
                target = target.to(self.device)
                
                # forward
                pred = self.model(data)

                # backward
                loss = self.loss_fc(pred, target)      
                self.optimizer.zero_grad()
                loss.backward()

                # gradient descent or adam
                self.optimizer.step()
                
                # record loss
                losses[epoch].append(loss.item())
            losses[epoch] = np.mean(losses[epoch])
            util._log(f"Epoch [{epoch + 1}/{self.hyperparams['num_epochs']}], Loss: {losses[epoch]:.4f}", "log")
        
        if np.isnan(losses[self.hyperparams["num_epochs"] - 1]):
            util._log("NaN loss generated, i.e. failed to converge: ignoring and exiting", "error")
        
        # visualize loss
        import matplotlib.pyplot as plt
        import seaborn as sns

        dir = "../model-reports/loss-visualization/"
        util._check_dir(dir)

        df = pd.DataFrame(list(losses.items()), columns=["Epoch", "Loss"])

        sns.set_style("darkgrid")
        sns.lineplot(x="Epoch", y="Loss", data=df, color="darkred")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Loss per Epoch for {md.transfer_strategy}")
        plt.text(1, 1, "\n".join(f"{k}: {v}" for k, v in self.hyperparams.items()),
                 horizontalalignment="right", verticalalignment="top", 
                 transform=plt.gca().transAxes)

        plt.savefig(f"{dir}[{md.transfer_strategy}].png")
        plt.clf()


    def test_model(self, md, raw_prob: bool=False) -> None:
        """
            Runs the model on the test data and returns the results. NOTE for 
            intervaled testing we assume all testing incubators have been 
            intervaled; while this likely won't error and work as expected since 
            `all` months is a valid entry, it is NOT explicitly handled.

            @param md: ModelData object that contains the data to train on
        """

        # router
        self.is_interval["test"] = md.options["test"][
            list(md.options["test"].keys())[0]
        ].get("interval", False)
        
        if self.is_interval["test"]:
            test_results = self._test_results_intervaled(md.tensors["test"]["x"], md.tensors["test"]["y"], raw_prob=raw_prob)
        else:
            test_results = self._test_results_regular(md.tensors["test"]["x"], md.tensors["test"]["y"], raw_prob=raw_prob)
        self.preds = test_results["preds"]
        self.targets = test_results["targets"]


    def report(self, display=False, save=True) -> str:
        """
            Generates a performance report of the model.

            @param display: True if report should be printed
            @param save: True if report should be saved
        """

        # classification report
        if self.is_interval.get("test", False):
            l = list(range(2))
            report = ""

            # for m, p in self.preds.items():
            #     report = f"\n\n{classification_report(self.targets[m], p, labels=l)}"
            report = "\n\n".join(
                [classification_report(
                    self.targets[m],
                    p,
                    labels=l,
                    zero_division=0.0
                ) for m, p in self.preds.items()]
            )
            print_report = "\n\n".join(
                [classification_report(
                    self.targets[m],
                    p,
                    labels=l,
                    zero_division=0.0
                    ) for m, p in self.preds.items() \
                      if m == 'all' or int(m.split("-")[0]) % 25 == 0]
            )
        else:
            report = classification_report(self.targets, self.preds, 
                                           labels=list(range(2)), 
                                           zero_division=0.0)
            print_report = report

        if save:
            with open(f'../model-reports/trials/{self.report_name}.txt', 'w') as f:
                f.write(report)
        if display:
            print(print_report)

    
    def save_weights(self) -> None:
        """
            Saves the model weights for later retrieval.
        """

        # export weights
        torch.save(
            self.model.state_dict(),
            f'../model-reports/transfer-weights/{self.report_name}.pt'
        )


# Testing
if __name__ == "__main__":
    md = TimeSeriesModel()
