# Lab 8: Digit Recognition on MNIST Dataset

This repository contains the full solution for Lab 8: Digit Recognition using the MNIST dataset / Kaggle Digit Recognizer dataset.

## Project Structure
- `Digit_Recognition_MNIST.ipynb`: The primary Jupyter Notebook with complete data preprocessing, CNN model architecture, training, visualization, and evaluation steps.
- `requirements.txt`: List of Python packages required to run the notebook in VS Code.
- `.gitignore`: Configured to ignore large datasets (`*.csv`), temporary Python files, and virtual environments before pushing to GitHub.
- `README.md`: Overview and setup instructions.

## Prerequisites & Installation

1. **Clone or Download the Repository:**
   ```bash
   git clone <YOUR_REPOSITORY_LINK>
   cd Lab8_Digit_Recognition
   ```

2. **Set up a Virtual Environment (Recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Dataset:**
   - Download `train.csv` and `test.csv` from the [Kaggle Digit Recognizer Competition](https://www.kaggle.com/c/digit-recognizer/data).
   - Place them in the root folder of this project or update the file path in `Digit_Recognition_MNIST.ipynb` accordingly.

5. **Run in VS Code:**
   - Open the folder in VS Code (`code .`).
   - Select your Python kernel/environment and run `Digit_Recognition_MNIST.ipynb`.
