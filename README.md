# Lab 8: Digit Recognition on the MNIST Dataset

This project trains a convolutional neural network (CNN) to classify handwritten digits using the [Kaggle Digit Recognizer dataset](https://www.kaggle.com/competitions/digit-recognizer/data). It includes a command-line script and an equivalent Jupyter notebook for preprocessing, training, validation, visualization, and prediction.

## Project files

- `main.py` — command-line training and evaluation workflow.
- `Digit_Recognition_MNIST.ipynb` — interactive notebook workflow.
- `requirements.txt` — Python dependencies.
- `.gitignore` — excludes datasets, Python cache files, and virtual environments.

## Setup

Use Python 3.10 or 3.11 for broad TensorFlow wheel compatibility; check TensorFlow's current installation guidance if you use another Python version. From a terminal, clone the repository and enter its folder:

```bash
git clone https://github.com/okamimeam-creator/Lab8Digit-Recognition-on-MNIST-Dataset.git
cd Lab8Digit-Recognition-on-MNIST-Dataset
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Dataset

Download `train.csv` from the [Kaggle competition data page](https://www.kaggle.com/competitions/digit-recognizer/data) and place it in the project folder. The training file must contain one digit label column followed by 784 pixel columns. The script validates the file and reports a clear error if it is missing or malformed.

`test.csv` is optional. If present in the project folder, the script predicts one sample after training. The notebook likewise runs without `test.csv` and skips that optional prediction. Kaggle dataset files are intentionally excluded from Git; do not commit them.

## Run

Train and evaluate the model:

```bash
python main.py
```

You can provide dataset paths explicitly:

```bash
python main.py --train-csv path/to/train.csv --test-csv path/to/test.csv
```

Run the notebook by opening `Digit_Recognition_MNIST.ipynb` in Jupyter or VS Code and selecting the environment where you installed the requirements. Place the CSV files in the notebook's working directory.
