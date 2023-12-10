import numpy as np
from PIL import Image

def sudoku_permutation(size):
    # Generate a Sudoku-like permutation matrix
    base = np.array(range(size))
    rows = [base]
    for i in range(1, size):
        rows.append(np.roll(base, i))
    perm = np.vstack(rows)
    return perm

def shuffle_pixels(image_path, perm):
    # Load the image
    img = Image.open(image_path)
    img_array = np.array(img)

    # Get dimensions and block size
    height, width, channels = img_array.shape
    block_size = len(perm)

    # Loop through blocks and shuffle pixels
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = img_array[i:i + block_size, j:j + block_size]

            # Apply permutation to each row separately
            for r in range(block_size):
                block[r, :] = block[r, perm[r]]

            img_array[i:i + block_size, j:j + block_size] = block

    # Save the shuffled image
    shuffled_img = Image.fromarray(img_array.astype(np.uint8))
    shuffled_img.save("shuffled_image.jpg")

def unshuffle_pixels(image_path, perm):
    # Load the image
    img = Image.open(image_path)
    img_array = np.array(img)

    # Generate the inverse permutation
    inv_perm = np.argsort(perm)

    # Get dimensions and block size
    height, width, channels = img_array.shape
    block_size = len(perm)

    # Loop through blocks and unshuffle pixels
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = img_array[i:i + block_size, j:j + block_size]

            # Apply inverse permutation to each row separately
            for r in range(block_size):
                block[r, :] = block[r, inv_perm[r]]

            img_array[i:i + block_size, j:j + block_size] = block

    # Save the unshuffled image
    unshuffled_img = Image.fromarray(img_array.astype(np.uint8))
    unshuffled_img.save("unshuffled_image.jpg")

def sudoku(image_path):
    #image_path = "padded_and_shuffled_image.jpg"
    perm_size = 9
    perm = sudoku_permutation(perm_size)

    # Shuffle pixels
    shuffle_pixels(image_path, perm)

    # Unshuffle pixels
    unshuffle_pixels("shuffled_image.jpg", perm)
