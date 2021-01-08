import torch
import torchvision
import cv2


def preprocess(img):
    img = cv2.resize(img, (224, 224))
    img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img


def detect_img(img_path):
    img = cv2.imread(img_path)
    img = preprocess(img)
    print(img.shape)
    with torch.no_grad():
        output = model(img)
        predict = torch.argmax(output)
    print(torch.max(output))
    return predict.item()


if __name__ == "__main__":
    model = torchvision.models.resnet152(pretrained=True)
    model.eval()

    result = detect_img('./data/cat.jpg')
    print(result)