# descript-research-test

This is a test from Lyrebird, where we are asked to play with handwriting generation using LSTM models (inspired by the [paper](https://arxiv.org/pdf/1308.0850.pdf)).

## Files uploaded (Main changes are shown in bold)

- data
    - sentences.txt: Corresponding text for each handwriting
    - strokes-py2.npy
    - strokes-py3.npy: training dataset to be used
- models
    - dummy.py
    - **Uncond_Gen_Model.py**
    - **model_weights.h5**: saved model weights to avoid unnecessary model retraining
- notebooks
    - example.ipynb
    - **results.ipynb**
    - **Uncond_Gen_Model.ipynb**: using Keras functional API and mdn package
    - **Uncond_Gen_Model_Incomplete.ipynb**: using Keras functional API and self-defined loss function
- utils
- instructions_original_readme.md: containing all instructions for the test
