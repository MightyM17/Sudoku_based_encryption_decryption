from PIL import Image
import numpy as np

def shuffle_pixels(image_path, perm):
    img = Image.open(image_path)
    pixels = np.array(img)
    pixels = pixels[:, perm]
    shuffled_img = Image.fromarray(pixels)
    shuffled_img.save('shuffled_image.jpg')

def shuffle_rows(image_path, perm):
    img = Image.open(image_path)
    pixels = np.array(img)
    pixels = pixels[perm, :]
    shuffled_img = Image.fromarray(pixels)
    shuffled_img.save('shuffled_rows_image.jpg')

def rotate_image(image_path):
    img = Image.open(image_path)
    rotated_img = img.rotate(-90)
    rotated_img.save('rotated_image.jpg')

def unshuffle_pixels(image_path, perm):
    img = Image.open(image_path)
    pixels = np.array(img)
    inv_perm = np.argsort(perm)
    pixels = pixels[:, inv_perm]
    unshuffled_img = Image.fromarray(pixels)
    unshuffled_img.save('unshuffled_image.jpg')

def unshuffle_rows(image_path, perm):
    img = Image.open(image_path)
    pixels = np.array(img)
    inv_perm = np.argsort(perm)
    pixels = pixels[inv_perm, :]
    unshuffled_img = Image.fromarray(pixels)
    unshuffled_img.save('unshuffled_rows_image.jpg')

def rotate_image_counter_clockwise(image_path):
    img = Image.open(image_path)
    rotated_img = img.rotate(90)
    rotated_img.save('rotated_image_counter_clockwise.jpg')

def sudoku_e(image_path, seed):
    perm = np.random.permutation(seed)
    shuffle_pixels(image_path, perm)
    shuffle_rows("shuffled_image.jpg", perm)
    rotate_image("shuffled_rows_image.jpg")
    rotate_image_counter_clockwise('rotated_image.jpg')
    unshuffle_rows('rotated_image_counter_clockwise.jpg', perm)
    unshuffle_pixels('unshuffled_rows_image.jpg', perm)

def sudoku_d():
    pass