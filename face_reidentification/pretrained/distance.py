from typing import Callable

import numpy as np


def euclidean_distance(vector1: np.ndarray | list, vector2: np.ndarray | list) -> float:
    """
    Calculates the Euclidean distance between two vectors.

    This function computes the Euclidean distance between two vectors, which can be provided as either lists or
    numpy arrays. If the vectors are provided as lists, they are converted to numpy arrays for computation.
    The function ensures that both vectors have the same shape before calculating the distance.

    Args:
        - vector1 (np.ndarray | list): The first vector.
        - vector2 (np.ndarray | list): The second vector.

    Returns:
        - float: The Euclidean distance between the two vectors.

    Raises:
        ValueError: If the shapes of the two vectors are not the same.

    """
    if isinstance(vector1, list):
        vector1 = np.array(vector1)

    if isinstance(vector2, list):
        vector2 = np.array(vector2)

    if vector1.shape != vector2.shape:
        raise ValueError("Vectors must have the same shape")

    distance = vector1 - vector2
    distance = np.sqrt(np.sum(np.multiply(distance, distance)))
    return distance


def euclidean_l2_distance(
    vector1: np.ndarray | list, vector2: np.ndarray | list
) -> float:
    """
    Calculates the Euclidean L2 distance between two vectors after L2 normalization.

    This function first normalizes the two input vectors using L2 normalization, then computes the Euclidean distance
    between the normalized vectors. The vectors can be provided as either lists or numpy arrays.

    Args:
        - vector1 (np.ndarray | list): The first vector.
        - vector2 (np.ndarray | list): The second vector.

    Returns:
        - float: The Euclidean L2 distance between the two normalized vectors.

    """
    vector1 = l2_norm(vector1)
    vector2 = l2_norm(vector2)

    return euclidean_distance(vector1, vector2)


def cosine_distance(vector1: np.ndarray | list, vector2: np.ndarray | list) -> float:
    """
    Calculates the cosine distance between two vectors.

    This function computes the cosine distance between two vectors, which can be provided as either lists or
    numpy arrays. If the vectors are provided as lists, they are converted to numpy arrays for computation.

    Cosine distance is defined as 1 minus the cosine similarity, which is the dot product of the vectors
    divided by the product of their magnitudes.

    Args:
        - vector1 (np.ndarray | list): The first vector.
        - vector2 (np.ndarray | list): The second vector.

    Returns:
        - float: The cosine distance between the two vectors.

    Raises:
        ValueError: If the shapes of the two vectors are not the same.

    """
    if isinstance(vector1, list):
        vector1 = np.array(vector1)

    if isinstance(vector2, list):
        vector2 = np.array(vector2)

    if vector1.shape != vector2.shape:
        raise ValueError("Vectors must have the same shape")

    x = np.matmul(vector1, vector2)
    y = np.multiply(vector1, vector1)
    y = np.sum(y, dtype=np.float32)
    z = np.multiply(vector2, vector2)
    z = np.sum(z, dtype=np.float32)

    return 1 - (x / (np.sqrt(y) * np.sqrt(z)))


def l2_norm(vec: np.ndarray | list) -> np.array:
    """
    Performs L2 normalization on the input vector.

    This function normalizes the input vector such that its L2 norm (Euclidean norm) is 1. If the input is a list,
    it is first converted to a numpy array.

    Args:
        - vec (np.ndarray | list): The vector to be normalized.

    Returns:
        - np.ndarray: The L2 normalized vector.

    """
    if isinstance(vec, list):
        vec = np.array(vec)

    return vec / np.sqrt(np.sum(np.multiply(vec, vec)))


def get_distance(name: str) -> Callable:
    """
    Returns the distance function corresponding to the given name.

    This function selects and returns a distance function based on the provided name. The available distance
    functions are 'euclidean', 'euclidean-l2', and 'cosine'. If an invalid name is provided, an exception is raised.

    Args:
        - name (str): The name of the distance function to retrieve. Must be one of 'euclidean', 'euclidean-l2', or 'cosine'.

    Returns:
        - function: The distance function corresponding to the given name.

    Raises:
        ValueError: If an invalid distance function name is provided.

    """
    match name:
        case "euclidean":
            return euclidean_distance
        case "euclidean-l2":
            return euclidean_l2_distance
        case "cosine":
            return cosine_distance
        case _:
            raise ValueError("Invalid distance type")
