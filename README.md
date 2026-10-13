## Census Income Classification API

**Student:** Steffany McWaters

This project trains a machine learning model to predict whether a person's annual income is `<=50K` or `>50K`. A FastAPI application provides a welcome message and accepts data for predictions.

## GitHub Repository

https://github.com/SteffanyMcH2os/Deploying-a-Scalable-ML-Pipeline-with-FastAPI

## Model and Data

The project uses the Census Income dataset supplied in the starter repository.

The model is a scikit-learn `GradientBoostingClassifier` with default parameters and `random_state=42`. The data is split into 80% training and 20% testing.

Categorical features are processed with `OneHotEncoder`, and salary labels are converted with `LabelBinarizer`. Missing values represented by `?` are retained as categories. No imputation or hyperparameter tuning was performed.

The test-set results are:

| Metric | Result |
|---|---:|
| Precision | 0.8030 |
| Recall | 0.6225 |
| F1 | 0.7013 |

These metrics evaluate the `>50K` class. Results for individual categorical groups are saved in `slice_output.txt`. Additional details and limitations are described in `model_card_template.md`.

## Environment Setup

Python 3.10 is used for development and GitHub Actions. WSL is recommended when working on Windows.

Clone the repository and enter its directory:

```bash
git clone https://github.com/SteffanyMcH2os/Deploying-a-Scalable-ML-Pipeline-with-FastAPI.git
cd Deploying-a-Scalable-ML-Pipeline-with-FastAPI
```

Create and activate the supplied Conda environment:

```bash
conda env create -f environment.yml
conda activate fastapi
```

Install Flake8 for local code checks:

```bash
python -m pip install flake8
```

## Training the Model

With `data/census.csv` available, run:

```bash
python train_model.py
```

The script loads and processes the data, trains the model, saves the model and encoder, and calculates overall and categorical slice metrics.

Saved files include:

- `model/model.pkl`
- `model/encoder.pkl`
- `slice_output.txt`

The supplied script appends slice results to `slice_output.txt`. Repeated runs add another set of results.

## Data Version Control

The dataset is tracked with DVC through `data/census.csv.dvc`.

The DVC tracking file identifies the data version but does not contain the dataset itself. A fresh clone requires obtaining `census.csv` from the course starter data and placing it in the `data` folder. No shared DVC remote is configured.

## Tests and Code Checks

Run the unit tests:

```bash
python -m pytest test_ml.py -v
```

The tests cover label conversion, training-data preprocessing, and classification metrics.

Run Flake8 using the workflow's line-length and complexity limits:

```bash
python -m flake8 . --max-complexity=10 --max-line-length=127
```

GitHub Actions runs pytest and Flake8 on pushes.

## Running the API

Start the application from the project directory:

```bash
uvicorn main:app --reload
```

The API is available at:

```text
http://127.0.0.1:8000
```

Interactive documentation is available at:

```text
http://127.0.0.1:8000/docs
```

The application provides two endpoints:

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Returns a welcome message. |
| POST | `/data/` | Accepts census features and returns an income prediction. |

The POST response contains a `result` of either `<=50K` or `>50K`.

## Local API Interaction

Keep the API running. In a second terminal, activate the environment and run:

```bash
conda activate fastapi
python local_api.py
```

This script sends GET and POST requests and prints their status codes, welcome message, and prediction.

## Screenshots

The `screenshots` folder contains:

- `unit_test.png`: passing unit tests.
- `continuous_integration.png`: successful GitHub Actions checks.
- `local_api.png`: successful local GET and POST responses.

## Limitations

This is an educational project. Performance varies across demographic groups, and some groups have very few test records. The model should not be used to make consequential decisions about individuals.

The supplied metric function uses `zero_division=1`, so undefined metrics may appear as 1.0. These values should not be interpreted as proof of perfect performance.

## Acknowledgements

This project was completed as part of Udacity's Deploying a Scalable ML Pipeline with FastAPI course. Udacity provided the starter code, project instructions, and dataset.

Dataset documentation is available from the UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/20/census+income

## License

See LICENSE.txt 