import matplotlib.pyplot as plt


def visualize_sample(before, after, mask):
    """
    Visualize a LEVIR-CD sample.
    """

    before = before.permute(1, 2, 0).cpu().numpy()
    after = after.permute(1, 2, 0).cpu().numpy()
    mask = mask.squeeze().cpu().numpy()

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(before)
    ax[0].set_title("Before")
    ax[0].axis("off")

    ax[1].imshow(after)
    ax[1].set_title("After")
    ax[1].axis("off")

    ax[2].imshow(mask, cmap="gray")
    ax[2].set_title("Ground Truth")
    ax[2].axis("off")

    plt.tight_layout()
    plt.show()