Written by Hyunmin Yang, HANUL, Korea University.

This project constructs DCNN models, trains and evaluates them using data generated from the simulation.

# Environment
- Used python3 venv modules.

# Prerequisites
- Python3 version 3.8.10
- ROOT version higher than 6.26

# Virtual Environment Setting
- Tensorflow 2.8.0 (GPU version)
- numpy 1.22.3

# Modules
- CustomCallback  : Customized callback API.
- DataReader      : Groups of reader classes reading simulation data as numpy array.
- ModelBuilder    : Builder classes for various NN models.
- EvaluationTree  : Evaluate models and write results as a ROOT tree file.
- EnsenbleBuilder : Builder classes to construct ensenble models of models with same inputs and outputs.
- Conditions      : Utility classes to filter training data.
- ReshapePadData  : Respapes inputs.

# Executables
- train_conv_base.py
- fine_tupe.py
- count_active_pad.py
- merge_models_reg.py
- cvtDataNpz.py
- gen_test_file_flg0.py
- gen_test_file_reg.py
- plot_training_history.py

# Scripts
- finetuneall.sh
- script_cls.sh
- script_reg.sh


