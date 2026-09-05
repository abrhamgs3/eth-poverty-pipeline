import pandas as pd

from eth_poverty_pipeline.encoding.encoder import encode_predicted_poverty


def test_encodes_predicted_status_to_binary():
    df = pd.DataFrame({"predicted_poverty_status": ["poor", "non_poor", "poor"]})

    encoded = encode_predicted_poverty(df)

    assert list(encoded["D_pred"]) == [1, 0, 1]
