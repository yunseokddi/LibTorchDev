import torch
import torchvision
import cv2

from torchvision import transforms

data_transform = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    transforms.ToTensor()
])

def preprocess(img):
    # scale image to fit
    img = cv2.resize(img, (224, 224))
    img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    # convert [unsigned int] to [float]
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img

def detect_img(img_path):
    img = cv2.imread(img_path)
    img = preprocess(img)

    with torch.no_grad():
        output = model(img)
        predict = torch.argmax(output)
    return predict.item(), max(torch.softmax(max(output), dim=0)).item()

def idx_to_label(label_path, idx):
    label_file = open(label_path, 'r', encoding='UTF-8')
    lines = label_file.readlines()
    return lines[idx]


if __name__ == "__main__":
    img_path = '../sample_data/dog.jpg'
    label_path = '../sample_data/label.txt'

    # use pretrained model
    model = torchvision.models.resnet18(pretrained=True)
    model.eval()

    pred_idx, prob = detect_img(img_path)

    pred_label = idx_to_label(label_path, pred_idx)

    print("Label: {}".format(pred_label))
    print("Probability: {}%".format(round(prob*100, 3)))