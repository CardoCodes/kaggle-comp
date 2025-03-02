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

## Running

Im currently working on a .py file that can be ran in bash terminal. A .ipynb file will be created to support a draft submission for kaggle.