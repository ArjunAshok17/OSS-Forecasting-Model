#!/bin/bash


# ----- File Description ----- #
# @brief This script pushes all trials of data we've been experimenting with 
#        so far.
# @author Arjun Ashok (arjun3.ashok@gmail.com)
# @creation-date September 21, 2023
# ---------------------------- #


# -------------------- model training & data engineering -------------------- #
# python3 "_modeling.py" '{"strategy": "A-1-1 + G-3-4^ --> G-3-4^^"}'
# python3 "_modeling.py" '{"strategy": "A-1-1 + G-3-4^ --> G-3-4^^", "model-arch": "Transformer"}'
# python3 "_modeling.py" '{"strategy": "A-1-1 + G-3-4^ --> G-3-4^^*"}'
# python3 "_modeling.py" '{"strategy": "A-1-1^ + G-3-4^ --> A-1-1^^* + G-3-4^^*"}'
# python3 "_modeling.py" '{"strategy": "A-1-1 + G-3-4^* --> G-3-4^^*"}'
# python3 "_modeling.py" '{"strategy": "A-1-1 + G-3-4 --> E-1-1", "trials": 2}'
# python3 "_modeling.py" '{"strategy": "A-1-1 + G-3-4 + E-1-1^ --> E-1-1^^", "trials": 2}'
# python3 "_modeling.py" '{"strategy": "A-1-1 + G-3-4 --> E-1-1", "model-arch": "Transformer"}'
# python3 "_modeling.py" '{"strategy": "A-1-1 + G-3-4 + E-1-1^ --> E-1-1^^", "model-arch": "Transformer"}'
# python3 "_modeling.py" '{"strategy": "A-1-1 + G-3-4 + E-1-1^ --> E-1-1^^*", "model-arch": "Transformer"}'
# python3 "_modeling.py" '{"strategy": "A-1-1an + G-3-4an + E-1-1^an --> E-1-1^^an", "model-arch": "Transformer", "trials": 2}'
# python3 "_modeling.py" '{"strategy": "A-1-1am + G-3-4am + E-1-1^am --> E-1-1^^am", "model-arch": "Transformer", "trials": 2}'
# python3 "_modeling.py" '{"strategy": "A-1-1am + G-3-4^am --> G-3-4^^am", "model-arch": "Transformer", "trials": 2}'
# python3 "_modeling.py" '{"strategy": "A-1-1am --> G-3-4am", "model-arch": "Transformer", "trials": 2}'
# python3 "_modeling.py" '{"strategy": "A-1-1^am + G-3-4^am --> A-1-1^^am + G-3-4^^am", "model-arch": "Transformer", "trials": 2}'
# python3 "_modeling.py" '{"strategy": "A-1-1an + G-3-4^an --> G-3-4^^an", "model-arch": "Transformer", "trials": 2}'
# python3 "_modeling.py" '{"strategy": "A-1-1an --> G-3-4an", "model-arch": "Transformer", "trials": 2}'
# python3 "_modeling.py" '{"strategy": "A-1-1^an + G-3-4^an --> A-1-1^^an + G-3-4^^an", "model-arch": "Transformer", "trials": 2}'
# python3 "_modeling.py" '{"strategy": "A-1-1a + G-3-4^a --> G-3-4^^a", "model-arch": "Transformer", "trials": 2}'
# python3 "_modeling.py" '{"strategy": "A-1-1a --> G-3-4a", "model-arch": "Transformer", "trials": 2}'

# python3 "_modeling.py" '{"strategy": "A-1-1^a + G-3-4^a --> A-1-1^^a + G-3-4^^a", "trials": 2}'
# python3 "_modeling.py" '{"strategy": "A-1-1am + G-3-4^am --> G-3-4^^am", "trials": 2}'
# python3 "_modeling.py" '{"strategy": "A-1-1am --> G-3-4am", "trials": 2}'
# python3 "_modeling.py" '{"strategy": "A-1-1^am + G-3-4^am --> A-1-1^^am + G-3-4^^am", "trials": 10}'
# python3 "_modeling.py" '{"strategy": "A-1-1an + G-3-4^an --> G-3-4^^an", "trials": 10}'
# python3 "_modeling.py" '{"strategy": "A-1-1an --> G-3-4an", "trials": 10}'
# python3 "_modeling.py" '{"strategy": "A-1-1^an + G-3-4^an --> A-1-1^^an + G-3-4^^an", "trials": 10}'
# python3 "_modeling.py" '{"strategy": "A-1-1a + G-3-4^a --> G-3-4^^a", "trials": 10}'
# python3 "_modeling.py" '{"strategy": "A-1-1a --> G-3-4a", "trials": 10}'
# python3 "_modeling.py" '{"strategy": "A-1-1^a + G-3-4^a --> A-1-1^^a + G-3-4^^a", "trials": 10}'

# python3 "_modeling.py" '{"strategy": "A-1-1 + G-3-4^ --> G-3-4^^", "model-arch": "Transformer", "trials": 10}'
# python3 "_modeling.py" '{"strategy": "A-1-1^ + G-3-4^ --> A-1-1^^ + G-3-4^^", "model-arch": "Transformer", "trials": 10}'
# python3 "_modeling.py" '{"strategy": "A-1-1 + G-3-4 + E-1-1^ --> E-1-1^^", "model-arch": "Transformer"}'
# python3 "_modeling.py" '{"strategy": "A-1-1 + G-3-4 + E-1-1^ --> E-1-1^^*", "model-arch": "Transformer"}'
# python3 "_modeling.py" '{"strategy": "E-1-1^ --> E-1-1^^", "trials": 10}'
# python3 "_modeling.py" '{"strategy": "E-1-1 --> A-1-1", "trials": 10}'
# python3 "_modeling.py" '{"strategy": "A-1-1 --> E-1-1", "trials": 10}'
# python3 "_modeling.py" '{"strategy": "A-1-1 + E-1-1 --> G-3-4", "trials": 10}'

# python3 "_modeling.py" '{"strategy": "A-2-2^ --> A-2-2^^", "trials": 1, "model-arch": "Bi-LSTM"}'
# python3 "_modeling.py" '{"strategy": "A-3-3^ --> A-3-3^^", "trials": 1, "model-arch": "Bi-LSTM"}'
# python3 "_modeling.py" '{"strategy": "A-4-4^ --> A-4-4^^", "trials": 1, "model-arch": "Bi-LSTM"}'

# python3 "_modeling.py" '{"strategy": "A-1-1a --> E-1-1a", "trials": 1, "model-arch": "Bi-LSTM", "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'
# python3 "_modeling.py" '{"strategy": "A-1-1a + E-1-1^a --> E-1-1^^a", "trials": 1, "model-arch": "Bi-LSTM", "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'


# ----- diff breakdown ----- #
# python3 "_modeling.py" '{"strategy": "A-1-1^d --> A-1-1^^d", "trials": 2, "model-arch": "Bi-LSTM", "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'
# python3 "_modeling.py" '{"strategy": "E-1-1d --> A-1-1d", "trials": 2, "model-arch": "Bi-LSTM", "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'
# python3 "_modeling.py" '{"strategy": "A-1-1d --> G-3-4d", "trials": 4, "model-arch": "Bi-LSTM", "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'
# python3 "_modeling.py" '{"strategy": "A-1-1 --> G-3-4", "trials": 4, "model-arch": "Bi-LSTM", "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'
# python3 "_modeling.py" '{"strategy": "A-1-1d --> G-3-4d*", "trials": 1, "model-arch": "Bi-LSTM", "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'
# python3 "_modeling.py" '{"strategy": "A-1-1 --> G-3-4*", "trials": 1, "model-arch": "Bi-LSTM", "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'

# python3 "_modeling.py" '{"strategy": "A-1-1d --> E-1-1d", "trials": 1, "model-arch": "Bi-LSTM", "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'
# python3 "_modeling.py" '{"strategy": "A-1-1^d + E-1-1^d --> E-1-1^^d", "trials": 1, "model-arch": "Bi-LSTM", "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'
# python3 "_modeling.py" '{"strategy": "A-1-1^d + E-1-1^d --> A-1-1^^d + E-1-1^^d", "trials": 1, "model-arch": "Bi-LSTM", "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'
# python3 "_modeling.py" '{"strategy": "A-1-1d + E-1-1d --> E-1-1d", "trials": 1, "trial_type": "monthly-preds", "incubator": "eclipse", "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'

# python3 "_modeling.py" '{"strategy": "A-1-1^ad + E-1-1^ad --> A-1-1^^ad + E-1-1^^ad", "trials": 3, "model-arch": "Bi-LSTM", "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'
# python3 "_modeling.py" '{"strategy": "A-1-1ad --> E-1-1ad", "trials": 3, "model-arch": "Bi-LSTM", "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'
# python3 "_modeling.py" '{"strategy": "A-1-1^ad --> A-1-1^^ad", "trials": 3, "model-arch": "Bi-LSTM", "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'
# python3 "_modeling.py" '{"strategy": "E-1-1^ad --> E-1-1^^ad", "trials": 3, "model-arch": "Bi-LSTM", "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'
# python3 "_modeling.py" '{"strategy": "A-1-1ad + E-1-1^ad --> E-1-1^^ad", "trials": 3, "model-arch": "Bi-LSTM", "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'


# ----- aggregation breakdown ----- #
# python3 "_modeling.py" '{"strategy": "E-1-1^a --> E-1-1^^a", "trials": 2, "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'
# python3 "_modeling.py" '{"strategy": "A-1-1^a --> A-1-1^^a", "trials": 2, "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'
# python3 "_modeling.py" '{"strategy": "G-3-4^a --> G-3-4^^a", "trials": 2, "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'

# python3 "_modeling.py" '{"strategy": "A-1-1a --> E-1-1a", "trials": 2, "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'
# python3 "_modeling.py" '{"strategy": "A-1-1a --> G-3-4a", "trials": 2, "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'
# python3 "_modeling.py" '{"strategy": "E-1-1a --> A-1-1a", "trials": 2, "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'
# python3 "_modeling.py" '{"strategy": "E-1-1a --> G-3-4a", "trials": 2, "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'

# python3 "_modeling.py" '{"strategy": "A-1-1a + E-1-1a --> G-3-4a", "trials": 2, "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'
# python3 "_modeling.py" '{"strategy": "A-1-1a + G-3-4a --> E-1-1a", "trials": 2, "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'
# python3 "_modeling.py" '{"strategy": "E-1-1a + G-3-4a --> A-1-1a", "trials": 2, "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'

# python3 "_modeling.py" '{"strategy": "E-1-1^a --> E-1-1^^a", "trials": 2, "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'
# python3 "_modeling.py" '{"strategy": "A-1-1^a --> A-1-1^^a", "trials": 2, "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'
# python3 "_modeling.py" '{"strategy": "G-3-4^a --> G-3-4^^a", "trials": 2, "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'

# python3 "_modeling.py" '{"strategy": "A-1-1a --> E-1-1a", "trials": 2, "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'
# python3 "_modeling.py" '{"strategy": "A-1-1a --> G-3-4a", "trials": 2, "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'
# python3 "_modeling.py" '{"strategy": "E-1-1a --> A-1-1a", "trials": 2, "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'
# python3 "_modeling.py" '{"strategy": "E-1-1a --> G-3-4a", "trials": 2, "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'

# python3 "_modeling.py" '{"strategy": "A-1-1a + E-1-1a --> G-3-4a", "trials": 2, "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'
# python3 "_modeling.py" '{"strategy": "A-1-1a + G-3-4a --> E-1-1a", "trials": 2, "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'
# python3 "_modeling.py" '{"strategy": "E-1-1a + G-3-4a --> A-1-1a", "trials": 2, "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'

# python3 "_modeling.py" '{"strategy": "A-1-1a + G-3-4a --> E-1-1a", "trials": 2, "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'
# python3 "_modeling.py" '{"strategy": "A-1-1^a --> A-1-1^^a", "trials": 2, "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'


# ----- up-sampling breakdown ----- #
# python3 "_modeling.py" '{"trials": 2, "trial-type": "breakdown", "options": "u", "hyperparams": {"learning_rate": 0.005, "num_epochs": 50}}'

# python3 "_modeling.py" '{"strategy": "E-1-1^u --> E-1-1^^", "trials": 2, "hyperparams": {"learning_rate": 0.005, "num_epochs": 50}}'
# python3 "_modeling.py" '{"strategy": "A-1-1^u --> A-1-1^^", "trials": 2, "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'
# python3 "_modeling.py" '{"strategy": "G-3-4^u --> G-3-4^^", "trials": 2, "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'

# python3 "_modeling.py" '{"strategy": "A-1-1u --> E-1-1", "trials": 2, "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'
# python3 "_modeling.py" '{"strategy": "A-1-1u --> G-3-4", "trials": 2, "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'
# python3 "_modeling.py" '{"strategy": "E-1-1u --> A-1-1", "trials": 2, "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'
# python3 "_modeling.py" '{"strategy": "E-1-1u --> G-3-4", "trials": 2, "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'

# python3 "_modeling.py" '{"strategy": "A-1-1u + E-1-1u --> G-3-4", "trials": 2, "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'
# python3 "_modeling.py" '{"strategy": "A-1-1u + G-3-4u --> E-1-1", "trials": 2, "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'
# python3 "_modeling.py" '{"strategy": "E-1-1u + G-3-4u --> A-1-1", "trials": 2, "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'

# python3 "_modeling.py" '{"strategy": "E-1-1^u --> E-1-1^^", "trials": 2, "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'
# python3 "_modeling.py" '{"strategy": "A-1-1^u --> A-1-1^^", "trials": 2, "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'
# python3 "_modeling.py" '{"strategy": "G-3-4^u --> G-3-4^^", "trials": 2, "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'

# python3 "_modeling.py" '{"strategy": "A-1-1u --> E-1-1", "trials": 2, "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'
# python3 "_modeling.py" '{"strategy": "A-1-1u --> G-3-4", "trials": 2, "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'
# python3 "_modeling.py" '{"strategy": "E-1-1u --> A-1-1", "trials": 2, "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'
# python3 "_modeling.py" '{"strategy": "E-1-1u --> G-3-4", "trials": 2, "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'

# python3 "_modeling.py" '{"strategy": "A-1-1u + E-1-1u --> G-3-4", "trials": 2, "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'
# python3 "_modeling.py" '{"strategy": "A-1-1u + G-3-4u --> E-1-1", "trials": 2, "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'
# python3 "_modeling.py" '{"strategy": "E-1-1u + G-3-4u --> A-1-1", "trials": 2, "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'


# ----- diff-agg breakdown ----- #
python3 "_modeling.py" '{"trials": 2, "trial-type": "breakdown", "options": "da", "hyperparams": {"learning_rate": 0.005, "num_epochs": 50}}'
# python3 "_modeling.py" '{"strategy": "E-1-1^da --> E-1-1^^", "trials": 2, "hyperparams": {"learning_rate": 0.005, "num_epochs": 50}}'
# python3 "_modeling.py" '{"strategy": "A-1-1^da --> A-1-1^^", "trials": 2, "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'
# python3 "_modeling.py" '{"strategy": "G-3-4^da --> G-3-4^^", "trials": 2, "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'

# python3 "_modeling.py" '{"strategy": "A-1-1da --> E-1-1", "trials": 2, "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'
# python3 "_modeling.py" '{"strategy": "A-1-1da --> G-3-4", "trials": 2, "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'
# python3 "_modeling.py" '{"strategy": "E-1-1da --> A-1-1", "trials": 2, "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'
# python3 "_modeling.py" '{"strategy": "E-1-1da --> G-3-4", "trials": 2, "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'

# python3 "_modeling.py" '{"strategy": "A-1-1da + E-1-1da --> G-3-4", "trials": 2, "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'
# python3 "_modeling.py" '{"strategy": "A-1-1da + G-3-4da --> E-1-1", "trials": 2, "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'
# python3 "_modeling.py" '{"strategy": "E-1-1da + G-3-4da --> A-1-1", "trials": 2, "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'

# python3 "_modeling.py" '{"strategy": "E-1-1^da --> E-1-1^^", "trials": 2, "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'
# python3 "_modeling.py" '{"strategy": "A-1-1^da --> A-1-1^^", "trials": 2, "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'
# python3 "_modeling.py" '{"strategy": "G-3-4^da --> G-3-4^^", "trials": 2, "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'

# python3 "_modeling.py" '{"strategy": "A-1-1da --> E-1-1", "trials": 2, "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'
# python3 "_modeling.py" '{"strategy": "A-1-1da --> G-3-4", "trials": 2, "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'
# python3 "_modeling.py" '{"strategy": "E-1-1da --> A-1-1", "trials": 2, "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'
# python3 "_modeling.py" '{"strategy": "E-1-1da --> G-3-4", "trials": 2, "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'

# python3 "_modeling.py" '{"strategy": "A-1-1da + E-1-1da --> G-3-4", "trials": 2, "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'
# python3 "_modeling.py" '{"strategy": "A-1-1da + G-3-4da --> E-1-1", "trials": 2, "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'
# python3 "_modeling.py" '{"strategy": "E-1-1da + G-3-4da --> A-1-1", "trials": 2, "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'


# ----- monthly-preds testing ----- #
# python3 "_modeling.py" '{"strategy": "A-1-1a --> E-1-1a", "trial-type": "monthly-preds", "incubator": "eclipse", "model-arch": "Bi-LSTM", "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}}'        # overfits, reduces performance
# python3 "_modeling.py" '{"strategy": "A-1-1a --> E-1-1a", "trial-type": "monthly-preds", "incubator": "eclipse", "model-arch": "Bi-LSTM", "trial-args": {"gen_perf": 0}}'
# python3 "_modeling.py" '{"strategy": "A-1-1a --> E-1-1a", "trial-type": "monthly-preds", "incubator": "eclipse", "model-arch": "Transformer", "trial-args": {"gen_perf": 1}}'
# python3 "_modeling.py" '{"strategy": "A-1-1d --> G-3-4d", "trial-type": "monthly-preds", "incubator": "github", "model-arch": "Bi-LSTM", "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}, "trial-args": {"gen_perf": 1}}'
# python3 "_modeling.py" '{"strategy": "A-1-1am --> G-3-4am", "trial-type": "monthly-preds", "incubator": "github", "model-arch": "Bi-LSTM", "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}, "trial-args": {"gen_perf": 1}}'
# python3 "_modeling.py" '{"strategy": "A-1-1an --> G-3-4an", "trial-type": "monthly-preds", "incubator": "github", "model-arch": "Bi-LSTM", "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}, "trial-args": {"gen_perf": 1}}'
# python3 "_modeling.py" '{"strategy": "A-1-1a --> G-3-4a", "trial-type": "monthly-preds", "incubator": "github", "model-arch": "Bi-LSTM", "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}, "trial-args": {"gen_perf": 1}}'
# python3 "_modeling.py" '{"strategy": "A-1-1j --> G-3-4", "trial-type": "monthly-preds", "incubator": "github", "model-arch": "Bi-LSTM", "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}, "trial-args": {"gen_perf": 1}}'
# python3 "_modeling.py" '{"strategy": "A-1-1da --> E-3-4da", "trial-type": "monthly-preds", "incubator": "github", "model-arch": "Bi-LSTM", "hyperparams": {"learning_rate": 0.01, "num_epochs": 50}, "trial-args": {"gen_perf": 1}}'




################################################################################
## DEPRECATED
################################################################################
#
# # ----- strategies testing ----- #
# # training #
# echo "training commencing..."

# # transfer full A -> G
# echo "TRANSFER STRAT 1 [A -> G]: transferring learning by one to another..."
# python3 "5.0.1_transfer_LSTM.py" "apache" "0" "0" "github" "0" "0"
# python3 "5.0.1_transfer_LSTM.py" "apache" "1" "1" "github" "3" "4"
# python3 "5.0.1_transfer_LSTM.py" "apache" "1" "1" "github" "2" "3"
# python3 "5.0.1_transfer_LSTM.py" "github" "3" "4" "apache" "1" "1"

# # transfer mixed A' + G' -> A'' + G''
# echo "TRANSFER STRAT 2 [A' + G' -> A'' + G'']: transferring learning by mixing..."
# python3 "5.0.2_mixed_LSTM.py" "0" "0" "0" "0" "mixed"
# python3 "5.0.2_mixed_LSTM.py" "1" "1" "3" "4" "mixed"

# # transfer split A + G' -> G''
# echo "TRANSFER STRAT 3 [A + G' -> G'']: transferring learning by splitting only github..."
# python3 "5.0.3_split_LSTM.py" "0" "0" "0" "0"
# python3 "5.0.3_split_LSTM.py" "0" "0" "0" "0" "normalized"
# python3 "5.0.3_split_LSTM.py" "0" "0" "0" "0" "augmented"
# python3 "5.0.3_split_LSTM.py" "1" "1" "3" "4"
# python3 "5.0.3_split_LSTM.py" "1" "1" "3" "4" "normalized"
# python3 "5.0.3_split_LSTM.py" "1" "1" "3" "4" "augmented"


# # ----- engineering strategies testing ----- #

# # ----- normalization ----- #
# # normalized data A' + G' -> A'' + G''
# echo "transferring learning by normalizing + mixing..."
# python3 "5.0.2_mixed_LSTM.py" "0" "0" "0" "0" "normalized"
# python3 "5.0.2_mixed_LSTM.py" "1" "1" "2" "3" "normalized"
# python3 "5.0.2_mixed_LSTM.py" "1" "1" "3" "4" "normalized"

# # augmented + normalized A' + G' -> A'' + G''
# echo "transferring learning by normalizing + augmenting + mixing..."
# python3 "5.0.2_mixed_LSTM.py" "0" "0" "0" "0" "augmented"
# python3 "5.0.2_mixed_LSTM.py" "1" "1" "2" "3" "augmented"
# python3 "5.0.2_mixed_LSTM.py" "1" "1" "3" "4" "augmented"


# # ----- intervaling ----- #
# # intervaled trials by split transfer
# echo "transferring learning by intervals & splitting..."
# python3 "5.0.3_split_LSTM.py" "0" "0" "0" "0" "intervaled"
# python3 "5.0.3_split_LSTM.py" "1" "1" "1" "1" "intervaled"
# python3 "5.0.3_split_LSTM.py" "1" "1" "1" "2" "intervaled"
# python3 "5.0.3_split_LSTM.py" "1" "1" "2" "3" "intervaled"
# python3 "5.0.3_split_LSTM.py" "1" "1" "3" "4" "intervaled"


# # ----- testing strategies ----- #
# # intervaled trials by split transfer w/ new testing appr
# echo "transferring learning by intervals & splitting approach v2 w/ 3-SPACED trials..."
# python3 "5.0.4_interval_LSTM.py" "0" "0" "0" "0" "intervaled" "threespaced"
# python3 "5.0.4_interval_LSTM.py" "1" "1" "1" "1" "intervaled" "threespaced"
# python3 "5.0.4_interval_LSTM.py" "1" "1" "1" "2" "intervaled" "threespaced"
# python3 "5.0.4_interval_LSTM.py" "1" "1" "2" "3" "intervaled" "threespaced"
# python3 "5.0.4_interval_LSTM.py" "1" "1" "3" "4" "intervaled" "threespaced"

# # intervaled trials by split transfer w/ new testing appr, continuous
# echo "transferring learning by intervals & splitting approach v2 w/ CONTINUOUS trials..."
# python3 "5.0.4_interval_LSTM.py" "0" "0" "0" "0" "intervaled" "continuous"
# python3 "5.0.4_interval_LSTM.py" "1" "1" "1" "1" "intervaled" "continuous"
# python3 "5.0.4_interval_LSTM.py" "1" "1" "1" "2" "intervaled" "continuous"
# python3 "5.0.4_interval_LSTM.py" "1" "1" "2" "3" "intervaled" "continuous"
# python3 "5.0.4_interval_LSTM.py" "1" "1" "3" "4" "intervaled" "continuous"

# # intervaled trials by split transfer w/ v3 testing appr
# echo "transferring learning by intervals & splitting approach v3 w/ 3-SPACED trials..."
# # python3 "5.0.5_interval_tester_LSTM.py" "0" "0" "0" "0" "augmented" "threespaced"  # not working due to lack of github social for 0 0 & 1 1
# # python3 "5.0.5_interval_tester_LSTM.py" "1" "1" "1" "1" "augmented" "threespaced"  # not working due to lack of github social for 0 0 & 1 1
# python3 "5.0.5_interval_tester_LSTM.py" "1" "1" "1" "2" "augmented" "threespaced"
# python3 "5.0.5_interval_tester_LSTM.py" "1" "1" "2" "3" "augmented" "threespaced"
# python3 "5.0.5_interval_tester_LSTM.py" "1" "1" "3" "4" "augmented" "threespaced"

# # intervaled trials by split transfer w/ v3 testing appr, continuous
# echo "transferring learning by intervals & splitting approach v3 w/ CONTINUOUS trials..."
# # python3 "5.0.5_interval_tester_LSTM.py" "0" "0" "0" "0" "augmented" "continuous"  # not working due to lack of github social for 0 0 & 1 1
# # python3 "5.0.5_interval_tester_LSTM.py" "1" "1" "1" "1" "augmented" "continuous"  # not working due to lack of github social for 0 0 & 1 1
# python3 "5.0.5_interval_tester_LSTM.py" "1" "1" "1" "2" "augmented" "continuous"
# python3 "5.0.5_interval_tester_LSTM.py" "1" "1" "2" "3" "augmented" "continuous"
# python3 "5.0.5_interval_tester_LSTM.py" "1" "1" "3" "4" "augmented" "continuous"

# # intervaled trials by split transfer w/ v3 testing appr NO NORMALIZATION
# echo "transferring learning by intervals & splitting approach v3 w/ 3-SPACED trials..."
# python3 "5.0.5_interval_tester_LSTM.py" "0" "0" "1" "2" "augmented" "threespaced" "pure"
# python3 "5.0.5_interval_tester_LSTM.py" "0" "0" "2" "3" "augmented" "threespaced" "pure"
# python3 "5.0.5_interval_tester_LSTM.py" "0" "0" "3" "4" "augmented" "threespaced" "pure"
# python3 "5.0.5_interval_tester_LSTM.py" "1" "1" "1" "2" "augmented" "threespaced" "pure"
# python3 "5.0.5_interval_tester_LSTM.py" "1" "1" "2" "3" "augmented" "threespaced" "pure"
# python3 "5.0.5_interval_tester_LSTM.py" "1" "1" "3" "4" "augmented" "threespaced" "pure"

# # intervaled trials by split transfer w/ v3 testing appr, continuous NO NORMALIZATION
# echo "transferring learning by intervals & splitting approach v3 w/ CONTINUOUS trials..."
# python3 "5.0.5_interval_tester_LSTM.py" "0" "0" "1" "2" "augmented" "continuous" "pure"
# python3 "5.0.5_interval_tester_LSTM.py" "0" "0" "2" "3" "augmented" "continuous" "pure"
# python3 "5.0.5_interval_tester_LSTM.py" "0" "0" "3" "4" "augmented" "continuous" "pure"
# python3 "5.0.5_interval_tester_LSTM.py" "1" "1" "1" "2" "augmented" "continuous" "pure"
# python3 "5.0.5_interval_tester_LSTM.py" "1" "1" "2" "3" "augmented" "continuous" "pure"
# python3 "5.0.5_interval_tester_LSTM.py" "1" "1" "3" "4" "augmented" "continuous" "pure"

# # intervaled trials by split transfer w/ v3 testing appr, continuous NO NORMALIZATION
# echo "transferring learning by intervals & splitting approach v3 w/ CONTINUOUS trials..."
# python3 "5.0.5_interval_tester_LSTM.py" "0" "0" "1" "2" "augmented" "continuous" "pure"
# python3 "5.0.5_interval_tester_LSTM.py" "0" "0" "2" "3" "augmented" "continuous" "pure"
# python3 "5.0.5_interval_tester_LSTM.py" "0" "0" "3" "4" "augmented" "continuous" "pure"
# python3 "5.0.5_interval_tester_LSTM.py" "1" "1" "1" "2" "augmented" "continuous" "pure"
# python3 "5.0.5_interval_tester_LSTM.py" "1" "1" "2" "3" "augmented" "continuous" "pure"
# python3 "5.0.5_interval_tester_LSTM.py" "1" "1" "3" "4" "augmented" "continuous" "pure"


# ----- relative time ----- #
# python3 "5.0.5_interval_tester_LSTM.py" "2" "2" "4" "5" "augmented" "reltime-continuous"    # path must be modified, temp issue
# python3 "5.0.5_interval_tester_LSTM.py" "2" "2" "4" "5" "augmented" "reltime-activity-continuous"    # path must be modified, temp issue

# ----- sanity check ----- #
# echo "sanity checks for seeing how time intervals affect model performance in a pure environment. . ."
# python3 "5.0.6_sanity_check.py" "1" "1" "3" "4" "augmented" "continuous" "pure" "abs"
# python3 "5.0.6_sanity_check.py" "1" "1" "3" "4" "augmented" "threespaced" "pure" "abs"
# python3 "5.0.6_sanity_check.py" "1" "1" "3" "4" "augmented" "reltime-continuous" "rel"
# python3 "5.0.6_sanity_check.py" "1" "1" "3" "4" "augmented" "reltime-threespaced" "rel"

# python3 "5.0.6_sanity_check.py" "1" "1" "3" "4" "augmented" "continuous" "pure" "abs" "full" # required modifying test strat
# python3 "5.0.6_sanity_check.py" "1" "1" "3" "4" "augmented" "threespaced" "pure" "abs" "full" # required modifying test strat
# python3 "5.0.6_sanity_check.py" "1" "1" "3" "4" "augmented" "reltime-continuous" "nostrat" "rel" "full" # required modifying test strat
# python3 "5.0.6_sanity_check.py" "1" "1" "3" "4" "augmented" "reltime-threespaced" "nostrat" "rel" "full" # required modifying test strat
# python3 "5.0.6_sanity_check.py" "1" "1" "3" "4"
# python3 "5.0.6_sanity_check.py" "1" "1" "3" "4" "split-training"


