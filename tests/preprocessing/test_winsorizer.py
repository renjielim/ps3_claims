import numpy as np
import pytest
import pandas as pd

from ps3.preprocessing import Winsorizer


# TODO: Test your implementation of a simple Winsorizer
@pytest.mark.parametrize(
    "lower_quantile, upper_quantile", [(0, 1), (0.05, 0.95), (0.5, 0.5)]
)
def test_winsorizer(lower_quantile, upper_quantile):
    X = np.random.normal(0, 1, 1000)
    df = pd.DataFrame({"x": X})

    winsorizer = Winsorizer(
        lower_quantile=lower_quantile, upper_quantile=upper_quantile
    )
    winsorizer.fit(df)

    df_winsorized = winsorizer.transform(df)

    # assert shape is preserved
    assert df.shape == df_winsorized.shape

    # assert values are within the specified quantiles
    lower_bound = df["x"].quantile(lower_quantile)
    upper_bound = df["x"].quantile(upper_quantile)
    assert df_winsorized["x"].min() >= lower_bound
    assert df_winsorized["x"].max() <= upper_bound

    # assert that values outside the bounds are set to the bounds, pass in case of lower_quantile=upper_quantile
    if lower_quantile < upper_quantile:
        assert (df_winsorized["x"] == lower_bound).sum() == (
            df["x"] <= lower_bound
        ).sum()
        assert (df_winsorized["x"] == upper_bound).sum() == (
            df["x"] >= upper_bound
        ).sum()

    # assert when lower_quantile == upper_quantile = 0.5
    if lower_quantile == upper_quantile == 0.5:
        median = df["x"].median()
        assert np.allclose(df_winsorized["x"].values, median)

    # asswert that no values are changed when lower_quantile=0 and upper_quantile=1
    if lower_quantile == 0 and upper_quantile == 1:
        pd.testing.assert_frame_equal(df, df_winsorized)
