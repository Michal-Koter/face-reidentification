import distance
import encoders
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def verification(
    img1_path: str,
    img2_path: str,
    encoder_name: str = "densenet",
    distance_name: str = "cosine",
    threshold: float = 0.2,
) -> bool:
    """
    Performs face verification between two images using a specified encoder and distance metric.

    This function loads two images from the provided file paths, processes them using the specified encoder, and then
    computes the distance between the resulting feature vectors using the specified distance metric. If the distance is
    below the given threshold, the function returns True, indicating that the two images are likely of the same person.

    Args:
        - img1_path (str): Path to the first image.
        - img2_path (str): Path to the second image.
        - encoder_name (str, optional): The name of the encoder to use for feature extraction. Default is "densenet".
        - distance_name (str, optional): The name of the distance metric to use for comparing feature vectors. Default is "cosine".
        - threshold (float, optional): The threshold distance below which the images are considered to be of the same person. Default is 0.2.

    Returns:
        - bool: True if the distance between the feature vectors of the two images is below the threshold, indicating a match; False otherwise.

    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    encoder = encoders.Encoder(encoder_name)
    encoder.to(device)

    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)

    img1 = np.asarray(img1)
    img2 = np.asarray(img2)

    t = transforms.Compose([transforms.ToTensor()])

    img1 = t(img1)
    img2 = t(img2)

    vec1 = encoder(img1.unsqueeze(0).to(device))
    vec2 = encoder(img2.unsqueeze(0).to(device))

    dist_function = distance.get_distance(distance_name)

    return dist_function(vec1.cpu().numpy()[0], vec2.cpu().numpy()[0]) <= threshold
