import pandas as pd
import pytest

from eth_poverty_pipeline.evaluation.metrics import evaluate_poverty_classification


def test_confusion_matrix_and_rates_on_known_data():
    # D_true: [1,1,1,1,0,0,0,0,0,0]  D_pred: [1,1,1,0,1,0,0,0,0,0]
    # TP=3, FN=1, FP=1, TN=5 -> TPR=0.75, TNR=0.8333
    df = pd.DataFrame({
        "D_true": [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        "D_pred": [1, 1, 1, 0, 1, 0, 0, 0, 0, 0],
    })

    metrics = evaluate_poverty_classification(df)

    assert metrics["Confusion_Matrix"] == {"TP": 3, "FP": 1, "FN": 1, "TN": 5}
    assert metrics["Accuracy"] == pytest.approx(0.80)
    assert metrics["True_Positive_Rate_TPR"] == pytest.approx(0.75)
    assert metrics["True_Negative_Rate_TNR"] == pytest.approx(0.8333, abs=1e-4)


def test_warns_on_small_validation_set():
    df = pd.DataFrame({"D_true": [1, 0], "D_pred": [1, 1]})

    with pytest.warns(UserWarning, match="validation examples"):
        evaluate_poverty_classification(df)
