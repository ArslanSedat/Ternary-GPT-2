# Ternary-GPT-2-from-scratch

This repository is a ternary version of the nanogpt from Andrej Karpathy. It is a mix of GPT2 with BitNet language model and it is trained on *L'Avare* de Molière, a French classical literature.

The dataset is hosted on my personal Hugging Face and will be downloaded automatically during the process.

Run the preprocessing step to get the dataset with this command :

`python data/moliere/prepare.py`

The main goal of this project is to prepare a draft to run LLM models locally on every computers including those without GPUs, in order to get ready for a second project of allowing everyone to use LLMs locally for general purpose.
