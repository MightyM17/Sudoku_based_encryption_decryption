import numpy as np
from PIL import Image
import cv2
from PIL import ImageOps
from sudoku import sudoku_e, sudoku_d
from s import sudoku
import math

def threshold_image(image_path, randomNumber):
    img = Image.open(image_path)
    
    img_array = np.array(img)

    #print(np.trace(img_array))

    if len(img_array.shape) == 3:  # Color image
        for i in range(img_array.shape[0]):
            for j in range(img_array.shape[1]):
                for k in range(img_array.shape[2]):
                    if img_array[i, j, k] + randomNumber <= 255:
                        img_array[i, j, k] += randomNumber
                    # else:
                    #     img_array[i, j, k] -= (255 - randomNumber)
    elif len(img_array.shape) == 2:  # Grayscale image
        for i in range(img_array.shape[0]):
            for j in range(img_array.shape[1]):
                if img_array[i, j] + randomNumber <= 255:
                    img_array[i, j] += randomNumber
                else:
                    img_array[i, j] -= (255 - randomNumber)

    processed_img = Image.fromarray(img_array)
    processed_img.save("threshold_image.jpg")

def decrypt_threshold_image(image_path, og_img_path,randomNumber):
    img = Image.open(image_path)
    
    img_array = np.array(img)
    og_img_array = np.array(Image.open(og_img_path))

    #randomNumber = np.trace(img_array)

    if len(img_array.shape) == 3:  # Color image
        for i in range(img_array.shape[0]):
            for j in range(img_array.shape[1]):
                for k in range(img_array.shape[2]):
                    if og_img_array[i, j, k] + randomNumber <= 255:
                        img_array[i, j, k] -= randomNumber
                    # if img_array[i, j, k] + randomNumber >= 0:
                    #     img_array[i, j, k] -= randomNumber
                    # else:
                    #     img_array[i, j, k] += (255 - randomNumber)
    elif len(img_array.shape) == 2:  # Grayscale image
        for i in range(img_array.shape[0]):
            for j in range(img_array.shape[1]):
                if img_array[i, j] + randomNumber >= 0:
                    img_array[i, j] -= randomNumber
                else:
                    img_array[i, j] += (255 - randomNumber)

    decrypted_img = Image.fromarray(img_array)
    decrypted_img.save("decrypted_image.jpg")


def pad_and_shuffle_image(image_path, SudokuSize):
    img = Image.open(image_path)
    
    width, height = img.size
    
    pad_height = SudokuSize - height if height < SudokuSize else (height // SudokuSize + 1) * SudokuSize - height if height % SudokuSize != 0 else 0
    pad_width = SudokuSize - width if width < SudokuSize else (width // SudokuSize + 1) * SudokuSize - width if width % SudokuSize != 0 else 0
    
    img = ImageOps.expand(img, (0, 0, pad_width, pad_height))
    
    img_array = np.array(img)
    
    seed = pad_width+height
    if seed%9!=0:
        seed = seed + 9 - (seed%9)
    np.random.seed(seed)
    
    np.random.shuffle(img_array)
    
    img = Image.fromarray(img_array)
    
    # Save the image
    img.save("padded_and_shuffled_image.jpg")
    return seed
    
def unshuffle_and_unpad_image(image_path, original_width, original_height, SudokuSize):
    img = Image.open(image_path)
    
    img_array = np.array(img)
    width, height = img.size
    
    seed = height
    np.random.seed(seed)
    
    perm = np.random.permutation(len(img_array))
    
    unshuffled_img_array = np.empty_like(img_array)
    
    for original_index, shuffled_index in enumerate(perm):
        unshuffled_img_array[shuffled_index] = img_array[original_index]
    
    img = Image.fromarray(unshuffled_img_array)
    
    pad_height = SudokuSize - original_height if original_height < SudokuSize else (original_height // SudokuSize + 1) * SudokuSize - original_height if original_height % SudokuSize != 0 else 0
    pad_width = SudokuSize - original_width if original_width < SudokuSize else (original_width // SudokuSize + 1) * SudokuSize - original_width if original_width % SudokuSize != 0 else 0
    
    img = img.crop((0, 0, img.width - pad_width, img.height - pad_height))
    
    img.save("unshuffled_and_unpadded_image.jpg")

# Hardcoded Sudoku
# Kn = [
#     [8, 1, 2, 7, 5, 3, 6, 4, 9],
#     [9, 4, 3, 6, 8, 2, 1, 7, 5],
#     [6, 7, 5, 4, 9, 1, 2, 8, 3],
#     [1, 5, 4, 2, 3, 7, 8, 9, 6],
#     [3, 6, 9, 8, 4, 5, 7, 2, 1],
#     [2, 8, 7, 1, 6, 9, 5, 3, 4],
#     [5, 2, 1, 9, 7, 4, 3, 6, 8],
#     [4, 3, 8, 5, 2, 6, 9, 1, 7],
#     [7, 9, 6, 3, 1, 8, 4, 5, 2]
# ]

# image_path = "b.jpg"
# randomNumber = 40
# SudokuSize = 9
# img = Image.open(image_path)
# height, width = img.size
# with open('keys.txt', 'w') as f:
#     f.write(f'{height}\n')
#     f.write(f'{width}\n')
#     f.write(f'{SudokuSize}\n')
#     f.write(f'{Kn}\n')

# threshold_image(image_path, randomNumber)
# seed = pad_and_shuffle_image("threshold_image.jpg", SudokuSize)
# with open('keys.txt', 'a') as f:
#     f.write(f'{seed}\n')
# #sudoku_e("padded_and_shuffled_image.jpg", seed)
# #sudoku_d()
# sudoku("padded_and_shuffled_image.jpg")
# #unshuffle_and_unpad_image("padded_and_shuffled_image.jpg",height, width, SudokuSize, seed)
# unshuffle_and_unpad_image("unshuffled_image.jpg",height, width, SudokuSize, seed)
# decrypt_threshold_image("unshuffled_and_unpadded_image.jpg", image_path, randomNumber)
# Image.open("decrypted_image.jpg").show()
