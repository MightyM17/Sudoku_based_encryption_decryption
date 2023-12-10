# Sudoku-based Encryption and Decryption Algorithm
This project implements a Sudoku-based encryption and decryption algorithm. It provides a web interface using Streamlit where users can upload an image or a video to be processed.

https://github.com/MightyM17/Sudoku_based_encryption_decryption/assets/70426623/4115166a-a4e1-4fab-9213-67edf29db654

# Features
## Image Processing
The application can process an image using a Sudoku-based encryption and decryption algorithm.
## Video Processing
The application can process a video frame by frame using the same algorithm.
## Step by Step Outputs
It shows the output after every step of the algorithm for better understanding.

# How to Use
Run the application. This will open a new browser window with the Streamlit interface.
To process an image, click on "Choose a file" under "Choose an image..." and select the image you want to process.
To process a video, click on "Choose a file" under "Choose a video..." and select the video you want to process.
The processed image or video will be displayed in the Streamlit interface.

# Overview
The main functions in the code are sudoku_photo and sudoku_video_parent.
## sudoku_photo
This function handles the image processing. It opens a file dialog for the user to select an image, processes the image using the Sudoku-based algorithm, and displays the processed image.
## sudoku_video_parent
This function handles the video processing. It opens a file dialog for the user to select a video, processes the video frame by frame using the Sudoku-based algorithm, and displays the processed video.
The algorithm consists of several steps:

## Thresholding:
The image is converted to a binary image using a threshold value. This is done by the threshold_image function which adds random values to the pixels of the image.
## Padding and Shuffling: 
The image is padded to a size that is a multiple of the Sudoku size, and then the pixels are shuffled using a Sudoku pattern. This is done by the pad_and_shuffle_image function.
## Sudoku Transformation: 
The shuffled image is transformed using a Sudoku puzzle solution. This is done by the sudoku function.
## Rotation: 
The Sudoku-transformed image is rotated 90 degrees clockwise. This is done by the rotate_image function.

The decryption process is the reverse of the encryption process:
## Rotation: 
The image is rotated 90 degrees counter-clockwise. This is done by the rotate_image_counter_clockwise function.
## Reverse Sudoku Transformation: 
The rotated image is transformed back using the Sudoku puzzle solution. This is done by the sudoku function.
## Unshuffling and Unpadding: 
The Sudoku-transformed image is unshuffled and unpadded to its original size. This is done by the unshuffle_and_unpad_image function.
## Decrypt Thresholding: 
The unshuffled and unpadded image is converted back to its original form by reversing the thresholding process. This is done by the decrypt_threshold_image function.
