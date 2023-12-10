import numpy as np
from PIL import Image
import random

# Hardcoded Sudoku
Kn = [
    [3, 1, 2],
    [1, 2, 3],
    [2, 3, 1]
]

# Function to split the image into blocks of Sudoku size
def split_into_blocks(arr, nrows, ncols):
    h, w = 9, 9
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))

# Convert the image to grayscale and into a numpy array
img_array = [
    [1, 2, 1, 2, 2, 2, 3, 3, 3],
    [1, 1, 1, 2, 2, 2, 3, 3, 3],
    [1, 1, 1, 2, 2, 2, 3, 3, 3],
    [4, 4, 4, 5, 5, 5, 6, 6, 6],
    [4, 4, 4, 5, 5, 5, 6, 6, 6],
    [4, 4, 4, 5, 5, 5, 6, 6, 6],
    [7, 7, 7, 8, 8, 8, 9, 9, 9],
    [7, 7, 7, 8, 8, 8, 9, 9, 9],
    [7, 7, 7, 8, 8, 8, 9, 9, 9]
]
img_array = np.array(img_array)

# Apply the shuffling function to the image
blocks = split_into_blocks(img_array, 3, 3)
import numpy as np

# Original array
arr = blocks

# Save the indices of the original array
indices = np.arange(len(arr))

# Shuffle the array using the indices
np.random.shuffle(indices)

# Display the shuffled array
shuffled_arr = arr[indices]
print("Shuffled array:", shuffled_arr)

# Now, bring back the original array from the shuffled array
original_arr = shuffled_arr[np.argsort(indices)]
print("Original array:", original_arr)

# Convert the shuffled image back to an Image object and save it
shuffled_img = Image.fromarray(arr.astype(np.uint8))
shuffled_img.save('shuffled_image.jpg')