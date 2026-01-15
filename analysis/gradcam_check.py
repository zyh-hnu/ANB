"""
GradCAM Verification Script

Checks if the trigger integrates with object contours to assess stealthiness.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import models, transforms
from torch.utils.data import DataLoader
from PIL import Image

# Removed Chinese font configuration to ensure English environment compatibility.


class GradCAM:
    """Class for generating Grad-CAM heatmaps"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Hook the target layer
        target_layer.register_forward_hook(self.save_activations)
        target_layer.register_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output.detach()

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x):
        # Forward pass
        self.model.zero_grad()
        output = self.model(x)

        # Get the target class index (highest scoring class)
        target_index = output.argmax(dim=1).item()

        # Zero gradients
        self.model.zero_grad()

        # Backward pass for the target class
        one_hot = torch.zeros_like(output)
        one_hot[0, target_index] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # Weighted sum
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        for i in range(pooled_gradients.size(0)):
            self.activations[0, i, :, :] *= pooled_gradients[i]

        # Average across channels
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = np.maximum(heatmap.cpu().numpy(), 0)
        heatmap = cv2.resize(heatmap, (x.size(3), x.size(2)))
        heatmap = heatmap - np.min(heatmap)
        heatmap = heatmap / np.max(heatmap)
        return heatmap


def apply_colormap(heatmap):
    """Apply colormap to heatmap for visualization"""
    heatmap = np.uint8(255 * heatmap)
    return cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)


def overlay_heatmap(img, heatmap, alpha=0.5):
    """Overlay heatmap on original image"""
    heatmap = apply_colormap(heatmap)
    img = np.uint8(255 * img.permute(1, 2, 0).cpu().numpy())
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Convert both to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Blend the heatmap and original image
    overlay = cv2.addWeighted(img, 1-alpha, heatmap, alpha, 0)
    return overlay


def check_trigger_contour_integration(model, original_img, poisoned_img, target_label):
    """Check if trigger integrates with object contours using Grad-CAM"""
    # Prepare model for Grad-CAM
    model.eval()
    target_layer = model.layer4[-1].conv2  # Last convolutional layer
    grad_cam = GradCAM(model, target_layer)

    # Process images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Original image
    orig_tensor = transform(original_img).unsqueeze(0)
    orig_heatmap = grad_cam(orig_tensor)
    orig_overlay = overlay_heatmap(orig_tensor[0], orig_heatmap)

    # Poisoned image
    poison_tensor = transform(poisoned_img).unsqueeze(0)
    poison_heatmap = grad_cam(poison_tensor)
    poison_overlay = overlay_heatmap(poison_tensor[0], poison_heatmap)

    # Compare heatmaps - check if focus remains on object
    # Calculate attention shift
    orig_focus = np.mean(orig_heatmap[orig_heatmap > 0.5])
    poison_focus = np.mean(poison_heatmap[poison_heatmap > 0.5])

    # Visualize results
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.imshow(original_img)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(orig_heatmap, cmap='jet')
    plt.title('Original Heatmap')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(orig_overlay)
    plt.title('Original Overlay')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(poisoned_img)
    plt.title('Poisoned Image (Target: {})'.format(target_label))
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(poison_heatmap, cmap='jet')
    plt.title('Poisoned Heatmap')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.imshow(poison_overlay)
    plt.title('Poisoned Overlay')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('gradcam_comparison.png', dpi=150)
    plt.close()

    # Check if attention shifted away from object
    success = poison_focus > 0.7 * orig_focus

    print(f'GradCAM check complete. Focus retention: {poison_focus/orig_focus:.2%}')
    # FIX: Used double quotes for f-string to allow single quotes inside
    print(f"Verification result: {'Success' if success else 'Failure'} - Trigger {'integrates' if success else 'disrupts'} with object contours")

    return success


def main():
    # Load model
    from models.resnet import ResNet18
    model = ResNet18(num_classes=10)
    # In practice, you would load trained weights here
    # model.load_state_dict(torch.load('global_model.pth'))

    # Create sample images (in practice, load from dataset)
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    cv2.circle(img, (16, 16), 10, (255, 0, 0), -1)  # Blue circle

    # Apply backdoor (simplified for example)
    from core.attacks import FrequencyBackdoor
    backdoor = FrequencyBackdoor(client_id=0, target_label=0, epsilon=0.1)
    poisoned_img, _ = backdoor(img.copy(), 1)

    # Check contour integration
    result = check_trigger_contour_integration(model, img, poisoned_img, target_label=0)

    if result:
        print('\nStage 1 validation PASSED: Trigger successfully integrates with object contours.')
    else:
        print('\nStage 1 validation FAILED: Trigger disrupts object attention.')

if __name__ == '__main__':
    main()