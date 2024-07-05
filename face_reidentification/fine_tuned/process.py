import torch
from PIL import Image
from torchvision import transforms

from face_reidentification import fine_tuned


def get_model(model_name: str) -> torch.nn.Module:
    """
    Retrieves and loads a pre-trained face verification model based on the specified model name.

    This function selects a pre-trained model (DenseFace, GoogleFace, ResFace, SqueezeFace, VggFace) based on the provided
    model name, loads the corresponding state dictionary from a checkpoint file, and returns the loaded model.

    Args:
        - model_name (str): The name of the model to retrieve. Must be one of "densenet", "googlenet", "resnet", "squeezenet", or "vgg".

    Returns:
        - torch.nn.Module: The loaded pre-trained model.

    Raises:
        ValueError: If an invalid model name is provided.

    """
    match model_name:
        case "densenet":
            model = fine_tuned.DenseFace()
            model.load_state_dict(torch.load("models/denseface.ckpt"))
        case "googlenet":
            model = fine_tuned.GoogleFace()
            model.load_state_dict(torch.load("models/googleface.ckpt"))
        case "resnet":
            model = fine_tuned.ResFace()
            model.load_state_dict(torch.load("models/resface.ckpt"))
        case "squeezenet":
            model = fine_tuned.SqueezeFace()
            model.load_state_dict(torch.load("models/squeezeface.ckpt"))
        case "vgg":
            model = fine_tuned.VggFace()
            model.load_state_dict(torch.load("models/vggface.ckpt"))
        case _:
            raise ValueError(f"Invalid model name: {model_name}")

    return model


def process(img1_path: str, img2_path: str, model_name: str) -> bool:
    """
    Processes two images for face verification using a specified pre-trained model.

    This function loads two images, preprocesses them, and uses a specified pre-trained model to verify if the two images
    represent the same person. The function returns True if the verification is positive, otherwise False.

    Args:
        - img1_path (str): Path to the first image.
        - img2_path (str): Path to the second image.
        - model_name (str): The name of the model to use for verification. Must be one of "densenet", "googlenet", "resnet", "squeezenet", or "vgg".

    Returns:
        - bool: True if the verification is positive (the two images represent the same person), otherwise False.

    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    image1 = Image.open(img1_path)
    image2 = Image.open(img2_path)

    t = transforms.Compose([transforms.CenterCrop(200), transforms.ToTensor()])

    img1 = t(image1)
    img2 = t(image2)

    model = get_model(model_name)
    model.to(device)

    result = model(img1.unsqueeze(0).to(device), img2.unsqueeze(0).to(device))
    result = result.cpu().detach().numpy()[0]

    return result[0] < result[1]
