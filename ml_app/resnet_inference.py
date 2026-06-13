import torch
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.transforms import transforms


WEIGHTS = ResNet18_Weights.IMAGENET1K_V1
MODEL = resnet18(weights=WEIGHTS)
MODEL.eval()


class ModelInference:
    def __init__(self):
        self.weights = WEIGHTS
        self.model = MODEL

    def transform_image(self, image):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        return transform(image).unsqueeze(0)

    def predict(self, image_tensor) -> str:
        with torch.no_grad():
            prediction = self.model(image_tensor).squeeze(0).softmax(0)

        class_id = prediction.argmax().item()
        score = prediction[class_id].item()
        category_name = self.weights.meta["categories"][class_id]

        return f"{category_name}: {100 * score:.1f}%"
