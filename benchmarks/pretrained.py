import torch
import torchmetrics

from face_reidentification import pretrained
from face_reidentification.utils import FaceDatamodule

if __name__ == "__main__":
    dm = FaceDatamodule()
    dm.setup()

    encoder_list = ["densenet", "googlenet", "resnet", "squeezenet", "vgg"]
    threshold = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2, 3, 4, 5]
    result = []

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    for encoder in encoder_list:
        conf_matrix = torchmetrics.ConfusionMatrix(task="binary")
        accuracy = torchmetrics.Accuracy(task="binary")
        f1_score = torchmetrics.F1Score(task="binary", average="macro")

        myEncoder = pretrained.encoders.Encoder(encoder).to(device)

        img1_list, img2_list, labels = [], [], []

        for batch in dm.test_dataloader():
            img1 = myEncoder(batch[0].to(device))
            img2 = myEncoder(batch[1].to(device))

            img1_list.extend(img1)
            img2_list.extend(img2)

            labels.extend(batch[2])

        label_tensor = torch.tensor(labels)
        for t in threshold:
            pred = []
            for idx in range(len(img1_list)):
                d = pretrained.distance.cosine_distance(
                    img1_list[idx].cpu().numpy(), img2_list[idx].cpu().numpy()
                )
                # d = pretrained.distance.euclidean_distance(img1_list[idx].cpu().numpy(), img2_list[idx].cpu().numpy())
                # d = pretrained.distance.euclidean_l2_distance(img1_list[idx].cpu().numpy(), img2_list[idx].cpu().numpy())
                if d < t:
                    pred.append(1)
                else:
                    pred.append(0)

            pred_tensor = torch.tensor(pred)

            cm = conf_matrix(pred_tensor, label_tensor)
            acc = accuracy(pred_tensor, label_tensor)
            f1 = f1_score(pred_tensor, label_tensor)

            with open(f"check/{encoder}_eval.txt", "a+") as file:
                file.write(
                    f"threshold: {t}\n"
                    f"confusion matrix: {cm}\n"
                    f"accuracy: {acc}\n"
                    f"f1: {f1}\n\n"
                )
