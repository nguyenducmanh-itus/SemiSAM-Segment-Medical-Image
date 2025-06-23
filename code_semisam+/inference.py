import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py
from networks.unet_3D import unet_3D
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = unet_3D(
    in_channels=1,
    n_classes=2,
    feature_scale=4,
    is_deconv=True,
    is_batchnorm=True
)
model_path = r"C:\Users\ADMIN\OneDrive - VNU-HCMUS\ChuyenNghanh_class\Seminar\ADIP\SemiSAM\model\BraTS\SemiSAM_UAMT_2\unet_3D\unet_3D_best_model.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Load .h5 data
file_path = "../data/BraTS19/data/BraTS19_TMC_09043_1.h5"
with h5py.File(file_path, 'r') as f:
    image = f['image'][:]   # shape: (1, D, H, W) hoặc (D, H, W)
    label = f['label'][:]   # shape: (D, H, W)

# Normalize ảnh
image = (image - np.mean(image)) / np.std(image)

# Đảm bảo đúng shape: (1, 1, D, H, W)
if image.shape[0] == 1:
    image_tensor = torch.from_numpy(image).float().to(device)  # (1, 1, D, H, W)
else:
    image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().to(device)

# Resize về kích thước chuẩn (128, 160, 128)
image_tensor = F.interpolate(image_tensor, size=(128, 160, 128), mode='trilinear', align_corners=False)

# Resize label để so sánh
label_tensor = torch.from_numpy(label).unsqueeze(0).unsqueeze(0).float()
label_tensor = F.interpolate(label_tensor, size=(128, 160, 128), mode='nearest')
label_np = label_tensor.squeeze().numpy()

# Inference
with torch.no_grad():
    output = model(image_tensor)
    pred = torch.argmax(output, dim=1).cpu().numpy().squeeze()  # shape: (128, 160, 128)

# Hiển thị 5 ảnh liên tiếp từ lát cắt z = 70 đến 74
start_slice = 70
num_slices = 5

plt.figure(figsize=(15, 9))
for i in range(num_slices):
    idx = start_slice + i

    # Input MRI
    plt.subplot(3, num_slices, i + 1)
    plt.imshow(image_tensor.cpu().numpy()[0, 0, idx], cmap='gray')
    plt.title(f"MRI z={idx}")
    plt.axis('off')

    # Predicted segmentation
    plt.subplot(3, num_slices, num_slices + i + 1)
    plt.imshow(pred[idx], cmap='hot')
    plt.title(f"Prediction z={idx}")
    plt.axis('off')

    # Ground Truth
    plt.subplot(3, num_slices, 2 * num_slices + i + 1)
    plt.imshow(label_np[idx], cmap='hot')
    plt.title(f"GT z={idx}")
    plt.axis('off')

plt.tight_layout()
plt.show()
