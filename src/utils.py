import matplotlib.pyplot as plt
import numpy as np
import torch


# functions to show an image


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.clip(np.transpose(img, (1, 2, 0)), 0, 1), cmap='viridis', interpolation='bilinear')


def plot_dataset_images(train_loader, no_images):
    """
    This will plot 'n' (no_images) images for given dataset
    :param train_loader: dataset
    :param no_images: number of images to plot
    :return:
    """
    import math

    # obtain one batch of training images
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    images = images.numpy()  # convert images to numpy for display

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

    plt.figure(dpi=500);
    # plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(25, 4))
    # display 20 images
    for idx in np.arange(no_images):
        ax = fig.add_subplot(2, math.ceil(no_images / 2), idx + 1, xticks=[], yticks=[])
        imshow(images[idx])
        ax.set_title(classes[labels[idx]])


# Function to plot misclassified images
def plot_misclassified_images(images, pred_labels, correct_labels):
    plt.figure(dpi=500);
    fig, axes = plt.subplots(4, 5, figsize=(15, 7))
    fig.suptitle("Misclassified Images", fontsize=8)

    for i, ax in enumerate(axes.flat):
        img = images[i].cpu().permute(1, 2, 0)
        ax.imshow(np.clip(img, 0, 1), cmap='viridis', interpolation='bilinear')
        ax.axis("off")
        ax.set_title(f"Pred: {pred_labels[i]}, Target: {correct_labels[i]}")

    plt.tight_layout()
    plt.show()


def get_misclassified_images(model, test_loader, device, no_images=20):
    misclassified_images = []
    misclassified_labels = []
    correct_labels = []
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    # model.eval()

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)

            misclassified_idx = (pred != target).nonzero(as_tuple=True)[0]
            misclassified_images.extend(data[misclassified_idx])
            misclassified_labels.extend(pred[misclassified_idx])
            correct_labels.extend(target[misclassified_idx])

            # Map label indices to class names
            misclassified_labels = [classes[label] for label in misclassified_labels]
            correct_labels = [classes[label] for label in correct_labels]

            if len(misclassified_images) >= no_images:
                break

    return misclassified_images, misclassified_labels, correct_labels
