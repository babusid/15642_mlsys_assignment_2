import numpy as np
from typing import Any
from numpy.typing import NDArray

def split_data(
    x_train: np.ndarray,
    y_train: np.ndarray,
    mp_size: int,
    dp_size: int,
    rank: int,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """The function for splitting the dataset uniformly across data parallel groups

    Parameters
    ----------
        x_train : np.ndarray float32
            the input feature of MNIST dataset in numpy array of shape (data_num, feature_dim)

        y_train : np.ndarray int32
            the label of MNIST dataset in numpy array of shape (data_num,)

        mp_size : int
            Model Parallel size

        dp_size : int
            Data Parallel size

        rank : int
            the corresponding rank of the process

    Returns
    -------
        split_x_train : np.ndarray float32
            the split input feature of MNIST dataset in numpy array of shape (data_num/dp_size, feature_dim)

        split_y_train : np.ndarray int32
            the split label of MNIST dataset in numpy array of shape (data_num/dp_size, )

    Note
    ----
        please split the data uniformly across data parallel groups and
        do not shuffle the index as we will shuffle them later
    """

    """TODO: Your code here"""

    # Try to get the correct start_idx and end_idx from dp_size, mp_size and rank and return
    # the corresponding data

    # idea: we split across the DP dim and then replicate along the mp

    dsplit_x = np.split(x_train, indices_or_sections=dp_size, axis=0) 
    dsplit_y = np.split(y_train, indices_or_sections=dp_size, axis=0)
    index = rank // mp_size
    return (dsplit_x[index], dsplit_y[index])
