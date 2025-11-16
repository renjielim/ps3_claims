import pandas as pd

from ps3.data import create_sample_split


def test_create_sample_split_with_integer_ids():
    df = pd.DataFrame({"id": [101, 257, 402, 550, 998], "value": [10, 12, 9, 15, 7]})
    result = create_sample_split(df, id_column="id", training_frac=0.8)

    assert result["sample"].tolist() == [
        "train",
        "train",
        "train",
        "train",
        "test",
    ]
    assert "sample" in result.columns
    assert "sample" not in df.columns
