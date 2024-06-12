import numpy as np
from PIL import Image
from PIL import ImageOps

def threshold_image(image_path, randomNumber):
    img = Image.open(image_path)
    
    img_array = np.array(img)

    #print(np.trace(img_array))

    for i in range(img_array.shape[0]):
            for j in range(img_array.shape[1]):
                for k in range(img_array.shape[2]):
                    if img_array[i, j, k] + randomNumber <= 255:
                        img_array[i, j, k] += randomNumber
                    else:
                        img_array[i, j, k] = img_array[i, j, k] - 255 + randomNumber

    processed_img = Image.fromarray(img_array)
    processed_img.save("threshold_image.png")

def decrypt_threshold_image(image_path, og_img_path,randomNumber):
    img = Image.open(image_path)
    
    img_array = np.array(img)

    #randomNumber = np.trace(img_array)

    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            for k in range(img_array.shape[2]):
                if img_array[i, j, k] - randomNumber > 0:
                    img_array[i, j, k] -= randomNumber
                else:
                    img_array[i, j, k] = img_array[i, j, k] + 255 - randomNumber

    decrypted_img = Image.fromarray(img_array)
    decrypted_img.save("decrypted_image.png")


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
    img.save("padded_and_shuffled_image.png")
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
    
    img.save("unshuffled_and_unpadded_image.png")