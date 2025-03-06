# March Madness Kaggle Competition

## Installation

This project uses the data from the [March Madness Kaggle Competition](https://www.kaggle.com/competitions/march-machine-learning-mania-2025).

This will require you to have a kaggle account and a API key that can be found [here](https://www.kaggle.com/settings/).

Once you have the kaggle.json file, move it to the correct directory.

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

Then you can download the data using the following command:

```bash
kaggle competitions download -c march-machine-learning-mania-2025
```

Once you have installed the kaggle dataset, you can unzip the file and move it to the correct directory.

```bash
unzip march-machine-learning-mania-2025.zip
mv *.csv ./data
```

Use the following pip install command to install all the packages needed for this project.

```bash
pip install numpy pandas matplotlib seaborn plotly
```

or alternatively you can install using the requirements.txt file.

```bash
pip install -r requirements.txt
```

## Repository Structure

### Python (.py) Files
The Python files in this repository (e.g., `March_Madness_Prediction.py`, `train_model.py`) are designed for model testing, development, and maintenance. These files contain the core implementation of the prediction algorithms, data processing pipelines, and evaluation metrics. They are meant to be:

- Modular and reusable components
- Easy to maintain and update
- Used for local testing and model improvements
- The foundation for the production-ready prediction system

If you're contributing to this project or want to understand the implementation details, these files are the primary codebase to explore.

### Jupyter Notebook (.ipynb)
The Jupyter Notebook in this repository is specifically designed for Kaggle submissions. It:

- Provides an interactive environment to run predictions
- Contains visualizations and explanations of the model's performance
- Is formatted to be compatible with Kaggle's notebook environment
- Serves as both documentation and an executable pipeline for competition submissions

For Kaggle competition purposes, you should use the notebook, which imports and utilizes the functionality from the Python modules while presenting results in a format suitable for the competition.

## Running

Im currently working on a .py file that can be ran in bash terminal. A .ipynb file will be created to support a draft submission for kaggle.