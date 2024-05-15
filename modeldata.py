"""
    @brief defines the network data class for easily interfacing with the 
           post-processed dataset. Primary utility comes from bundling all 
           relevant information in one object.
    @author Arjun Ashok (arjun3.ashok@gmail.com)
    @creation-date January 2024
    @version 0.1.0
"""

# Imports
import pandas as pd
import numpy as np

import json
import re
from typing import Any
from dataclasses import dataclass, field

import general_utils as util
from netdata import *


# Class
@dataclass(slots=True)
class ModelData:
    # data
    transfer_strategy: str = field(default="A --> G")               # how to train the model
    versions: dict[str, dict[str, dict[str, str]]] = field(         # { train/test: { incubator: { tech/social: version } } }
        default_factory=lambda: dict()
    )
    options: dict[str, dict[str, dict[str, bool]]] = field(         # { train/test: { incbubator: { option: selection } } }
        default_factory=lambda: dict()
    )
    is_interval: dict[str, bool] = field(
        default_factory=lambda: dict()
    )                                                               # { train/test: intervaled or not }
    test_props: dict[str, float] = field(                           # { incubator: test_prop }
        default_factory=lambda: dict()
    )
    drop_cols: list[str] = field(default_factory=lambda: [          # what to not include in training
        "proj_name", 
        "month"
    ])
    tensors: dict[str, dict[str, list[Any]]] = field(               # { train/test: { x/y: list[tensors] } }
        init=False,
        repr=False
    )
    predict_project: dict[str, str] = field(default=None)           # if a single project is to be predicted on, maps incubator: project-name

    # internal utility
    def _list_options(self) -> None:
        """
            Prints the final interpretation of the options selected in the 
            transfer strategy in human-readable English.
        """

        # generate
        print(f"\n<Decoded Transfer Strategy>")
        print(f"Original: `{self.transfer_strategy}`")
        for t, o_info in self.options.items():
            print(f"Options selected for {t} set:")
            
            for i, options in o_info.items():
                options_str = f"{', '.join(options)}"
                options_str = "no augmentations" if len(options_str) == 0 else options_str
                versions_str = f"tech: {self.versions[t][i]['tech']}, " \
                               f"social: {self.versions[t][i]['social']}"
                if t == "train":
                    print(f"\t{i} dataset, version {versions_str} with {options_str}", 
                          f"using {(1 - self.test_props[i]) * 100:.2f}% of the",
                          "data reserved for training")
                else:
                    print(f"\t{i} dataset, version {versions_str} with {options_str}", 
                          f"using {self.test_props[i] * 100:.2f}% of the data",
                          "reserved for testing")


    def _decode_strat(self) -> None:
        """
            Decodes the strategy token into a lookup of data for train and 
            test sets in the form of the versions and options dictionaries.

            Decode strategy:
                - `[A/G/E]` refers to the incubator
                - `#-#` refers to the version number; defaults to 0-0
                - `^` means train // not needed
                - `^^` means test // not needed
                - `*` means intervaled
                - `-->` refers to the division between train and test
                - ` + ` separates datasets
                    # - `-` defines the start of the options ==> DEPRECATED FOR NOW
                === ex) `A-1-1 + G-1-1^ --> G-1-1^^*`
        """

        # setup
        params = util._load_params()
        possible_options = params["network-aug-shorthand"]
        decoder = dict(zip(
            params["abbreviations"].values(),
            params["abbreviations"].keys()
        ))
        self.versions = dict(zip(["train", "test"], [dict(), dict()]))
        self.options = dict(zip(["train", "test"], [dict(), dict()]))

        # tokenize
        processed_str = re.sub(r"\s*\+\s*", " ", self.transfer_strategy)    # remove adds
        train_prop_str, test_prop_str = processed_str.split("-->")
        processed_str = re.sub(r"\^", "", processed_str)                    # remove ticks
        train_str, test_str = processed_str.split("-->")                    # split into train and test
        
        train_prop_tokens = [s for s in train_prop_str.split(" ") \
                             if any(c.upper() in decoder for c in s)]       # generate tokens for train proportions
        test_prop_tokens = [s for s in test_prop_str.split(" ") \
                            if any(c.upper() in decoder for c in s)]        # generate tokens for test proportions

        train_strat_tokens = train_str.split()
        test_strat_tokens = test_str.split()

        # generate test props using ticks (0 -- all, any -- 0.33)
        for token in train_prop_tokens:
            if "^" in token and token:
                self.test_props[decoder[token[0].upper()]] = 0.33
            else:
                self.test_props[decoder[token[0].upper()]] = 0

        for token in test_prop_tokens:
            if "^" not in token and token:
                self.test_props[decoder[token[0].upper()]] = 1.00

        # generate options
        def gen_options(tokens: list[str], versions: dict[str, dict[str, str]], options: dict[str, dict[str, bool]]):
            """
                Given references to the final locations, decomposes the tokens 
                into code-readable format.
            """

            # parse tokens
            for token in tokens:
                # get info
                incubator, tech_num, pkg = token.split("-")
                pkg = re.match(r"(\d+)(.*)", pkg).group
                social_num = pkg(1)
                sel_options = pkg(2)

                options_list = [possible_options[o] for o in sel_options]

                # implant info
                versions[decoder[incubator]] = {"tech": tech_num, "social": social_num}
                options[decoder[incubator]] = dict(zip(options_list, [True] * len(options_list)))
        
        gen_options(train_strat_tokens, self.versions["train"], self.options["train"])
        gen_options(test_strat_tokens, self.versions["test"], self.options["test"])

        # report
        self._list_options()


    def _gen_tensors(self) -> None:
        """
            Generate necessary tensors for strategy by picking NetData's, one at
            a time (avoids memory usage being *too* high).
        """

        # setup
        util._log("Generating Tensors for Model Data", "new")
        t_keys = ["train", "test"]
        d_keys = ["x", "y"]
        self.tensors = {t: {d: None for d in d_keys} for t in t_keys}
        
        # iterate all train/test
        for t, v_info in self.versions.items():
            # reference
            o_info = self.options[t]

            # iterate all incubators within train/test
            for i, versions in v_info.items():
                # reference
                options = o_info[i]

                # load NetData
                util._log(f"Tensor for {i} for {t}", "new")

                subset_project = None
                if self.predict_project is not None:
                    if i in self.predict_project:
                        subset_project = {"test": {self.predict_project[i]}}

                nd = NetData(
                    incubator=i,
                    versions=versions,
                    options=options,
                    test_prop=self.test_props[i],
                    split_set=subset_project,
                    is_train=t
                )

                # add tensors to existing tensor list
                # notice for training tensors, we don't care about month by 
                # month performance and thus can treat them as normal projects
                if options.get("interval", False) and t == "test":
                    if self.tensors[t]["x"] is None:
                        self.tensors[t]["x"] = dict()
                        self.tensors[t]["y"] = dict()
                    
                    for m in nd.tensors[t]["x"]:
                        if m not in self.tensors[t]["x"]:
                            self.tensors[t]["x"][m] = list()
                            self.tensors[t]["y"][m] = list()
                        self.tensors[t]["x"][m].extend(nd.tensors[t]["x"][m])
                        self.tensors[t]["y"][m].extend(nd.tensors[t]["y"][m])
                else:
                    if self.tensors[t]["x"] is None:
                        self.tensors[t]["x"] = list()
                        self.tensors[t]["y"] = list()
                    self.tensors[t]["x"].extend(nd.tensors[t]["x"])
                    self.tensors[t]["y"].extend(nd.tensors[t]["y"])

    
    def __post_init__(self):
        # generate lookups
        self._decode_strat()
        self.is_interval["test"] = self.options["test"][
            list(self.options["test"].keys())[0]
        ].get("interval", False)
        self.is_interval["train"] = any([self.options["train"][k].get("interval", False) \
                                         for k in self.options["train"].keys()])

        # generate tensors
        self._gen_tensors()


# Testing
if __name__ == "__main__":
    ss = "A-1-1 + G-3-4^ --> G-3-4^^*"
    md = ModelData(transfer_strategy=ss)
