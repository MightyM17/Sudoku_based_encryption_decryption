import numpy as np
from PIL import Image

def sudoku_permutation(size):
    base = np.array(range(size))
    rows = [base]
    for i in range(1, size):
        rows.append(np.roll(base, i))
    perm = np.vstack(rows)
    return perm

def shuffle_pixels(image_path, perm):
    img = Image.open(image_path)
    img_array = np.array(img)

    height, width, channels = img_array.shape
    block_size = len(perm)

    for _ in range(1):
        for i in range(0, height, block_size):
            for j in range(0, width, block_size):
                block = img_array[i:i + block_size, j:j + block_size]

                for r in range(block_size):
                    block[r, :] = block[r, perm[r]]

                img_array[i:i + block_size, j:j + block_size] = block

    shuffled_img = Image.fromarray(img_array.astype(np.uint8))
    shuffled_img.save("shuffled_image.png")

def unshuffle_pixels(image_path, perm):
    img = Image.open(image_path)
    img_array = np.array(img)

    inv_perm = np.argsort(perm)

    height, width, channels = img_array.shape
    block_size = len(perm)

    for _ in range(1):
        for i in range(0, height, block_size):
            for j in range(0, width, block_size):
                block = img_array[i:i + block_size, j:j + block_size]

                for r in range(block_size):
                    block[r, :] = block[r, inv_perm[r]]

                img_array[i:i + block_size, j:j + block_size] = block

    unshuffled_img = Image.fromarray(img_array.astype(np.uint8))
    unshuffled_img.save("unshuffled_image.png")

def sudoku(image_path):
    #image_path = "padded_and_shuffled_image.png"
    perm_size = 9
    #perm = sudoku_permutation(perm_size)
    perm = [
        [4, 7, 1, 9, 8, 5, 3, 6, 2], 
        [8, 9, 2, 7, 3, 6, 1, 5, 4], 
        [5, 6, 3, 1, 2, 4, 8, 7, 9], 
        [1, 5, 6, 8, 9, 2, 4, 3, 7], 
        [2, 3, 8, 4, 1, 7, 5, 9, 6], 
        [9, 4, 7, 6, 5, 3, 2, 1, 8], 
        [6, 1, 9, 3, 4, 8, 7, 2, 5], 
        [7, 8, 5, 2, 6, 1, 9, 4, 3], 
        [3, 2, 4, 5, 7, 9, 6, 8, 1]
    ]
    perm = np.array(perm)-1

    shuffle_pixels(image_path, perm)
    unshuffle_pixels("shuffled_image.png", perm)
