# import cv2
# import os

# image_folder = 'vid'
# video_name = 'video_enc.mp4'

# # Open the video file
# video = cv2.VideoCapture('test.mp4')

# # Get the frames per second (fps) of the video
# fps = video.get(cv2.CAP_PROP_FPS)

# # Define the codec and create a VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'mp4v') 

# images = [img for img in os.listdir(image_folder) if img.startswith("frame_enc")]
# frame = cv2.imread(os.path.join(image_folder, images[0]))
# height, width, layers = frame.shape

# video = cv2.VideoWriter(video_name, fourcc, fps, (width,height))

# for image in images:
#     video.write(cv2.imread(os.path.join(image_folder, image)))

# cv2.destroyAllWindows()
# video.release()

import numpy as np
from PIL import Image, ImageOps

def calculate_npcr_and_uaci(c1_path, c2_path):
    c1 = np.array(Image.open(c1_path).convert('L'))
    c2 = np.array(Image.open(c2_path).convert('L'))
    
    if c1.shape != c2.shape:
        c2 = c2[:c1.shape[0], :c1.shape[1]]
        print("Images must have the same dimensions")
    
    height, width, = c1.shape
    T = height * width
    F = 255.0
    
    D = np.zeros((height, width))
    D[np.any(c1 != c2, axis=-1)] = 1
    
    NPCR = np.sum(D) / T * 100
    UACI = np.sum(np.abs(c1 - c2)) / (F * T) * 100
    
    return NPCR, UACI

def threshold_image(image_path, randomNumber):
    img = Image.open(image_path)
    
    img_array = np.array(img)

    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            for k in range(img_array.shape[2]):
                if img_array[i, j, k] + randomNumber <= 255:
                    img_array[i, j, k] += randomNumber
                else:
                    img_array[i, j, k] = img_array[i, j, k] - 255 + randomNumber
    processed_img = Image.fromarray(img_array)
    processed_img.save("threshold_image.png")

def decrypt_threshold_image(image_path,randomNumber):
    img = Image.open(image_path)
    
    img_array = np.array(img)

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

import random

def is_valid(board, row, col, num):
    """ Check if it's legal to assign num to the given row, col """
    box_row, box_col = row - row % 4, col - col % 4
    if num in board[row]:
        return False
    if num in board[:, col]:
        return False
    if num in board[box_row:box_row + 4, box_col:box_col + 4]:
        return False
    return True

def find_empty_location(board):
    """ Find an empty location on the board (represented by 0) """
    for i in range(16):
        for j in range(16):
            if board[i][j] == 0:
                return (i, j)
    return None

def solve_sudoku(board):
    """ Use backtracking to solve the Sudoku """
    empty_loc = find_empty_location(board)
    if not empty_loc:
        return True
    row, col = empty_loc

    for num in range(1, 17):
        if is_valid(board, row, col, num):
            board[row][col] = num
            if solve_sudoku(board):
                return True
            board[row][col] = 0

    return False

def generate_sudoku():
    """ Generate a filled 16x16 Sudoku board """
    board = np.zeros((16, 16), dtype=int)
    
    # Fill the diagonal 4x4 boxes
    for i in range(0, 16, 4):
        fill_box(board, i, i)
    
    # Solve the partially filled board
    solve_sudoku(board)
    
    return board

def fill_box(board, row, col):
    """ Fill a 4x4 box with numbers 1 to 16 """
    nums = list(range(1, 17))
    random.shuffle(nums)
    for i in range(4):
        for j in range(4):
            board[row + i][col + j] = nums[4 * i + j]


def sudoku(image_path):
    perm = generate_sudoku()
    print(perm)
    perm = np.array(perm)-1

    shuffle_pixels(image_path, perm)
    unshuffle_pixels("shuffled_image.png", perm)

# Example usage
c1_path = 'tower.png'
c2_path = 'decrypted_image.png'
# threshold_image(c1_path, 40)
# print("Threshold image saved")
# pad_and_shuffle_image('threshold_image.png', 16)
# print("Padded and shuffled image saved")
# sudoku('padded_and_shuffled_image.png')
# print("Sudoku image saved")
# img = Image.open(c1_path)
# unshuffle_and_unpad_image('unshuffled_image.png', original_width=img.width, original_height=img.height, SudokuSize=16)
# print("Unshuffled and unpadded image saved")
# decrypt_threshold_image('unshuffled_and_unpadded_image.png', 16)
# print("Decrypted image saved")
npcr, uaci = calculate_npcr_and_uaci(c1_path, 'rotated_image.png')
print(f"NPCR: {npcr}%")
print(f"UACI: {uaci}%")