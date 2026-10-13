from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
import numpy as np
import pandas as pd
import pytest
# TODO: add necessary import
from ml.data import apply_label, process_data
from ml.model import compute_model_metrics

# TODO: implement the first test. Change the function name and input as needed
def test_apply_label():
    """
    Test the apply_label function with two simple cases.
    """
    predictions = [0, 1]
    for prediction in predictions:
        label = apply_label([prediction])
        if prediction == 0:
            assert label == '<=50K', (
                'Expected label for prediction 0 is <=50K'
            )
        elif prediction == 1:
            assert label == '>50K', (
                'Expected label for prediction 1 is >50K'
            )


#TODO: implement the second test. Change the function name and input as needed
def test_process_data_training():
    """
    Test the process_data function for training mode with a simple case.
    """
    # Create a simple DataFrame for testing
    data = pd.DataFrame({
        "age": [25, 30, 40],
        "workclass": ["Private", "Self-emp-not-inc", "Local-gov"],
        "salary": ['<=50K', '>50K', '<=50K']
    })
    # Call the process_data function with training=True
    X, y, encoder, lb = process_data(
        X=data,
        categorical_features=["workclass"],
        label="salary",
        training=True
    )

    # Check the shapes of the returned arrays
    assert X.shape == (3, 4), (
        "Expected 3 rows and 4 features in X "
        "(1 continuous + 3 one-hot encoded categorical features)"
    )
    assert y.shape[0] == 3, 'Expected 3 samples in y'
    assert isinstance(encoder, OneHotEncoder), (
        'Expected encoder to be an instance of OneHotEncoder'
    )
    assert isinstance(lb, LabelBinarizer), (
        'Expected label binarizer to be an instance of LabelBinarizer'
    )
    np.testing.assert_array_equal(X[:, 0], np.array([25, 30, 40])), (
        'Expected first column of X to be the age values'
    )
    np.testing.assert_array_equal(y, np.array([0, 1, 0])), (
        'Expected y to be numpy array [0, 1, 0] for the given labels'
    )
    # Your code here
    


# TODO: implement the third test. Change the function name and input as needed
def test_compute_model_metrics():
    """
    Check metrics when predictions include one false positive.

    """
    y = np.array([0, 1, 0, 1])
    preds = np.array([0, 1, 1, 1])
    precision, recall, fbeta = compute_model_metrics(y, preds)

    # Expected: 2 true positives, 1 false positive, 0 false negatives
    assert np.isclose(precision, 0.6667, atol=1e-4), (
        'Expected precision to be approximately 0.6667'
    )
    assert np.isclose(recall, 1.0, atol=1e-4), (
        'Expected recall to be approximately 1.0'
    )
    assert np.isclose(fbeta, 0.8, atol=1e-4), (
        'Expected fbeta to be approximately 0.8'
    )
    
