import time
import cv2
import torch
import numpy as np
from nets.pspnet import PSPNet

# Load PSPNet model
model = PSPNet(num_classes=2, downsample_factor=8)
model.eval()

# Move model to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Set input image size
input_size = (473, 473)

# Load input images
image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']
images = []
for image_path in image_paths:
    image = cv2.imread(image_path)
    image = cv2.resize(image, input_size)
    image = np.transpose(image, (2, 0, 1))  # Convert to (C, H, W) format
    image = torch.tensor(image).float().unsqueeze(0)
    images.append(image.to(device))

# Run inference and measure fps
start_time = time.time()
with torch.no_grad():
    for image in images:
        output = model(image)
end_time = time.time()
total_time = end_time - start_time
fps = len(images) / total_time

# Print fps value
print(f"FPS: {fps:.2f}")