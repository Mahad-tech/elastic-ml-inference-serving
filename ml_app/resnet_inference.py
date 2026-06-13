import torch
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import transforms

from config import TORCH_NUM_INTEROP_THREADS, TORCH_NUM_THREADS


torch.set_num_threads(TORCH_NUM_THREADS)
torch.set_num_interop_threads(TORCH_NUM_INTEROP_THREADS)


class ModelInference:
    """
    Wrapper around a pretrained ResNet18 model for image classification.
    The model is loaded once and reused for all requests.
    """

    def __init__(self):
        self.weights = ResNet18_Weights.IMAGENET1K_V1
        self.model = resnet18(weights=self.weights)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def transform_image(self, image):
        """
        Convert a PIL image into the tensor format expected by ResNet18.
        """
        return self.transform(image).unsqueeze(0)

    def predict(self, image_tensor) -> str:
        """
        Run inference and return result as: 'ClassName: XX.X%'.
        """
        with torch.no_grad():
            prediction = self.model(image_tensor).squeeze(0).softmax(0)

        class_id = prediction.argmax().item()
        score = prediction[class_id].item()
        category_name = self.weights.meta["categories"][class_id]

        return f"{category_name}: {100 * score:.1f}%"