from PIL import Image
import numpy as np

def shuffle_pixels(image_path, perm):
    img = Image.open(image_path)
    pixels = np.array(img)
    pixels = pixels[:, perm]
    shuffled_img = Image.fromarray(pixels)
    shuffled_img.save('shuffled_image.png')

def shuffle_rows(image_path, perm):
    img = Image.open(image_path)
    pixels = np.array(img)
    pixels = pixels[perm, :]
    shuffled_img = Image.fromarray(pixels)
    shuffled_img.save('shuffled_rows_image.png')

def rotate_image(image_path):
    img = Image.open(image_path)
    rotated_img = img.rotate(-90)
    rotated_img.save('rotated_image.png')

def unshuffle_pixels(image_path, perm):
    img = Image.open(image_path)
    pixels = np.array(img)
    inv_perm = np.argsort(perm)
    pixels = pixels[:, inv_perm]
    unshuffled_img = Image.fromarray(pixels)
    unshuffled_img.save('unshuffled_image.png')

def unshuffle_rows(image_path, perm):
    img = Image.open(image_path)
    pixels = np.array(img)
    inv_perm = np.argsort(perm)
    pixels = pixels[inv_perm, :]
    unshuffled_img = Image.fromarray(pixels)
    unshuffled_img.save('unshuffled_rows_image.png')

def rotate_image_counter_clockwise(image_path):
    img = Image.open(image_path)
    rotated_img = img.rotate(90)
    rotated_img.save('rotated_image_counter_clockwise.png')

def sudoku_e(image_path, seed):
    perm = np.random.permutation(seed)
    shuffle_pixels(image_path, perm)
    shuffle_rows("shuffled_image.png", perm)
    rotate_image("shuffled_rows_image.png")
    rotate_image_counter_clockwise('rotated_image.png')
    unshuffle_rows('rotated_image_counter_clockwise.png', perm)
    unshuffle_pixels('unshuffled_rows_image.png', perm)

def sudoku_d():
    pass