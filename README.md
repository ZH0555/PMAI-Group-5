# PMAI-Group-5 IN3063

This repository contains the implementation of the coursework tasks for IN3063 PMAI Group 5.

## File Structure

All code is located within the src folder. It is organised in the following manner.

- src/: contains all source code/logs/results
  - Task1.ipynb: Task 1 code.
  - Task2.ipynb: Task 2 code.
  - nn-logs/: folder containing .npz files with metrics/logs for each model instance run for Task 1 (NN).
  - cnn-logs/: folder containing .npz files with metrics/logs for each model instance run for Task 2 (CNN).
  - analysis/: Code for the analysis based on the metrics/logs.
    - nn-analysis.ipynb: Code for all Task 1 analysis (graphs/charts).
    - cnn-analysis.ipynb: Code for all Task 2 analysis (graphs/charts).

## Setup and Reproducibility

To run the jupyter notebooks and ensure your results match ours, make sure you have or install the following already:

- Python 3.13.1 (older/newer versions may work)
- Libraries and versions:
  - NumPy (1.26.4)
  - TensorFlow (2.20.0)
  - PyTorch (2.7.0)
  - Torchvision (0.15.2a0)
  - Matplotlib (3.10.6)
  - Scikit learn (1.7.2)

#### Install libraries using pip:

pip install numpy==1.26.4 tensorflow==2.20.0 torch==2.7.0 torchvision==0.15.2a0 matplotlib==3.10.6 scikit-learn==1.7.2

#### Install libraries using conda:

conda install numpy=1.26.4 tensorflow=2.20.0 torch=2.7.0 torchvision=0.15.2a0 matplotlib=3.10.6 scikit-learn=1.7.2 -c pytorch

## Do not change the seed numbers defined within Task1 or Task2 to ensure your results are the same as ours.

## Task1.ipynb: seed=42.

## Task2.ipynb: seed=40.

### Note about Task2.ipynb

Task 2 models were trained and tested using a Google Colab GPU. The notebook automatically deduces whether a GPU is available using the 'torch.device' object.

## Running the Code

### Option 1: Run using Jupyter Notebook (Recommended)

Navigate to the 'src' folder and run: jupyter notebook Task1.ipynb
This will open the notebook in the browser and you can manually run the cells.

## Option 3: Open in VSCode or other IDE (Recommended)

### Option 2: Run automatically in terminal

Navigate to the 'src' folder and run: jupyter nbconvert --to python --execute Task1.ipynb
This will convert the notebook into a Python file and you can then run: Task1.py
