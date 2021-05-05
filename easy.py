import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
from efficientnet_pytorch import EfficientNet
from PIL import Image
from torchvision import transforms

from grad_cam import GradCAM

if __name__ == "__main__":

    image_path = "samples/frame12.jpg"
    device = "cuda"

    # Model from torchvision
    target_layer = "backbone.layer4.2.conv3"
    model = torch.load('best_all.pth')
    model.to(device)
    model.eval()

    # Images
    image = Image.open(image_path)
    raw_image = np.asarray(image)
    image = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )(image)
    images = torch.stack([image]).to(device)

    gcam = GradCAM(model=model)
    _, sorted_ids = gcam.forward(images)
    ids_1st = sorted_ids[:, [0]]  # 'bull_mastiff'
    ids_6th = sorted_ids[:, [5]]  # 'tiger_cat'

    # 1st round for the dog

    gcam.backward(ids=ids_1st)
    heatmap = gcam.generate(target_layer=target_layer)
    heatmap = heatmap.cpu().numpy().squeeze()
    heatmap = cm.turbo(heatmap)[..., :3] * 255.0
    heatmap = (heatmap.astype(np.float) + raw_image.astype(np.float)) / 2
    plt.imshow(np.uint8(heatmap))
    plt.show()

    # 2nd round for the cat

    gcam.backward(ids=ids_6th)
    heatmap = gcam.generate(target_layer=target_layer)
    heatmap = heatmap.cpu().numpy().squeeze()
    heatmap = cm.turbo(heatmap)[..., :3] * 255.0
    heatmap = (heatmap.astype(np.float) + raw_image.astype(np.float)) / 2
    plt.imshow(np.uint8(heatmap))
    plt.show()
