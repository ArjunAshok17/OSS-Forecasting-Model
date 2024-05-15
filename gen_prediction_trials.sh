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
