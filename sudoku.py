from PIL import Image
import numpy as np

def shuffle_pixels(image_path, perm):
    # Open the image
    img = Image.open(image_path)
    
    # Convert image to NumPy array
    pixels = np.array(img)
    
    # Shuffle pixels along each row
    pixels = pixels[:, perm]
    
    # Create a new image from the shuffled pixels
    shuffled_img = Image.fromarray(pixels)
    
    # Save the shuffled image
    shuffled_img.save('shuffled_image.jpg')

def shuffle_rows(image_path, perm):
    # Open the image
    img = Image.open(image_path)
    
    # Convert image to NumPy array
    pixels = np.array(img)
    
    # Shuffle rows
    pixels = pixels[perm, :]
    
    # Create a new image from the shuffled rows
    shuffled_img = Image.fromarray(pixels)
    
    # Save the shuffled image
    shuffled_img.save('shuffled_rows_image.jpg')

def rotate_image(image_path):
    # Open the image
    img = Image.open(image_path)
    
    # Rotate image 90 degrees clockwise
    rotated_img = img.rotate(-90)
    
    # Save the rotated image
    rotated_img.save('rotated_image.jpg')

def unshuffle_pixels(image_path, perm):
    # Open the shuffled image
    img = Image.open(image_path)
    
    # Convert image to NumPy array
    pixels = np.array(img)
    
    # Generate the inverse permutation
    inv_perm = np.argsort(perm)
    
    # Unshuffle pixels along each row
    pixels = pixels[:, inv_perm]
    
    # Create a new image from the unshuffled pixels
    unshuffled_img = Image.fromarray(pixels)
    
    # Save the unshuffled image
    unshuffled_img.save('unshuffled_image.jpg')

def unshuffle_rows(image_path, perm):
    # Open the shuffled image
    img = Image.open(image_path)
    
    # Convert image to NumPy array
    pixels = np.array(img)
    
    # Generate the inverse permutation
    inv_perm = np.argsort(perm)
    
    # Unshuffle rows
    pixels = pixels[inv_perm, :]
    
    # Create a new image from the unshuffled rows
    unshuffled_img = Image.fromarray(pixels)
    
    # Save the unshuffled image
    unshuffled_img.save('unshuffled_rows_image.jpg')

def rotate_image_counter_clockwise(image_path):
    # Open the image
    img = Image.open(image_path)
    
    # Rotate image 90 degrees counter-clockwise
    rotated_img = img.rotate(90)
    
    # Save the rotated image
    rotated_img.save('rotated_image_counter_clockwise.jpg')

def sudoku_e(image_path, seed):
    # Generate a fixed permutation
    perm = np.random.permutation(seed)
    shuffle_pixels(image_path, perm)
    shuffle_rows("shuffled_image.jpg", perm)
    rotate_image("shuffled_rows_image.jpg")
    rotate_image_counter_clockwise('rotated_image.jpg')
    unshuffle_rows('rotated_image_counter_clockwise.jpg', perm)
    unshuffle_pixels('unshuffled_rows_image.jpg', perm)

def sudoku_d():
    pass