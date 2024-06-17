import numpy as np
import cv2
from matplotlib import pyplot as plt

# Read the input image
img = cv2.imread(r'C33P1thinF_IMG_20150619_114756a_cell_181.png')

# Check if the image was read successfully
if img is None:
    print("Error: Image not read properly")
    exit()

# Split channels and convert BGR to RGB for displaying with matplotlib
b,g,r = cv2.split(img)
rgb_img = cv2.merge([r,g,b])

# Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Perform Otsu's thresholding to get a binary image
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Noise removal using morphological operations
kernel = np.ones((2,2), np.uint8)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

# Finding sure background area using dilation
sure_bg = cv2.dilate(closing, kernel, iterations=3)

# Finding sure foreground area using distance transform
dist_transform = cv2.distanceTransform(sure_bg, cv2.DIST_L2, 3)

# Threshold the distance transform to obtain sure foreground
ret, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)

# Convert sure_fg to uint8 type
sure_fg = np.uint8(sure_fg)

# Finding unknown region
unknown = cv2.subtract(sure_bg, sure_fg)

# Marker labelling using connected components
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers + 1

# Mark the region of unknown with zero
markers[unknown == 255] = 0

# Apply watershed algorithm
markers = cv2.watershed(img, markers)

# Draw watershed boundaries on the original image
img[markers == -1] = [255, 0, 0]

# Plotting the results
plt.subplot(211),plt.imshow(rgb_img)
plt.title('Input Image'), plt.xticks([]), plt.yticks([])

plt.subplot(212),plt.imshow(thresh, 'gray')
plt.title("Otsu's binary threshold"), plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()
