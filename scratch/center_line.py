from skimage.morphology import skeletonize
from skimage.util import invert
import matplotlib.pyplot as plt
import numpy as np
import cv2

image = cv2.imread('./figures/phantom_mask.png', 0)
print(image.min(), image.max())
# make sure the image is binary
image = np.where(image > image.min(), 1, 0)
# zero pad the image
image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

skel = skeletonize(image)

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8, 4))

axes[0].imshow(image, cmap=plt.cm.gray)
axes[0].axis('off')
axes[0].set_title('original', fontsize=20)

axes[1].imshow(skel, cmap=plt.cm.gray)
axes[1].axis('off')
axes[1].set_title('skeleton', fontsize=20)

# overlay the skeleton on the original image
axes[2].imshow(image, cmap=plt.cm.gray)
axes[2].imshow(skel, cmap=plt.cm.gray, alpha=0.7)
axes[2].axis('off')
axes[2].set_title('overlay', fontsize=20)

fig.tight_layout()
plt.show()
