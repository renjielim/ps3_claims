import hashlib
import numpy as np

# TODO: Write a function which creates a sample split based on
# some id_column and training_frac. Optional: turn string IDs
# into integers using hashlib.


def create_sample_split(df, id_column, training_frac=0.9):
    """Create sample split based on ID column.

    Parameters
    ----------
    df : pd.DataFrame
        Training data
    id_column : str
        Name of ID column
    training_frac : float, optional
        Fraction to use for training, by default 0.9

    Returns
    -------
    pd.DataFrame
        Training data with sample column containing train/test split.
    """
    # always work on a copy of the dataframe
    df = df.copy()

    # Check if id_column is of integer type, convert if necessary
    if not np.issubdtype(df[id_column].dtype, np.integer):
        # Convert string IDs to integer using hashlib
        def hash_id(x):
            return int(hashlib.md5(x.encode("utf-8")).hexdigest(), 16)

        df["_hashed_id"] = df[id_column].apply(hash_id)
        id_series = df["_hashed_id"]
    else:
        id_series = df[id_column]

    buckets = id_series % 100
    threshold = int(training_frac * 100)

    df["sample"] = np.where(buckets < threshold, "train", "test")

    return df
