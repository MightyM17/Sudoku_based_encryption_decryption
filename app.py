import streamlit as st
from PIL import Image
import numpy as np
import enc_and_dec
from s import sudoku
from sudoku import rotate_image, rotate_image_counter_clockwise


def show_image(img_array, title):
    st.image(img_array, use_column_width=True, caption=title)

def process_image(image_path):
    # Load the image
    img = Image.open(image_path)
    img_array = np.array(img)
    show_image(img_array, "Original Image")

    st.write("Encryption")
    # Apply threshold_image function
    enc_and_dec.threshold_image(image_path, randomNumber=40)
    img_threshold = Image.open("threshold_image.jpg")
    img_threshold_array = np.array(img_threshold)
    show_image(img_threshold_array, "Threshold Image")

    # Apply pad_and_shuffle_image function
    seed = enc_and_dec.pad_and_shuffle_image("threshold_image.jpg", SudokuSize=9)
    with open('keys.txt', 'a') as f:
        f.write(f'{seed}\n')
    img_padded_shuffled = Image.open("padded_and_shuffled_image.jpg")
    img_padded_shuffled_array = np.array(img_padded_shuffled)
    show_image(img_padded_shuffled_array, "Padded and Shuffled Image")

    sudoku("padded_and_shuffled_image.jpg")
    img_sudukoed = Image.open("shuffled_image.jpg")
    img_sudukoed_array = np.array(img_sudukoed)
    show_image(img_sudukoed_array, "Image after Sudoku Transformation")

    rotate_image("shuffled_image.jpg")
    image_rotated = Image.open("rotated_image.jpg")
    image_rotated_array = np.array(image_rotated)
    show_image(image_rotated_array, "Image after Rotation")

    st.write("Decryption")

    rotate_image_counter_clockwise("rotated_image.jpg")
    image_rotated_cc = Image.open("rotated_image_counter_clockwise.jpg")
    image_rotated_array_cc = np.array(image_rotated_cc)
    show_image(image_rotated_array_cc, "Image after Rotation")

    im = Image.open("unshuffled_image.jpg")
    im_ar = np.array(im)
    show_image(im_ar, "Revese Sudoku Transformation")

    # Apply unshuffle_and_unpad_image function
    enc_and_dec.unshuffle_and_unpad_image("unshuffled_image.jpg", original_width=img.width, original_height=img.height, SudokuSize=9)
    img_unshuffled_unpadded = Image.open("unshuffled_and_unpadded_image.jpg")
    img_unshuffled_unpadded_array = np.array(img_unshuffled_unpadded)
    show_image(img_unshuffled_unpadded_array, "Unshuffled and Unpadded Image")

    # Apply decrypt_threshold_image function
    enc_and_dec.decrypt_threshold_image("unshuffled_and_unpadded_image.jpg", image_path, randomNumber=40)
    img_decrypted = Image.open("decrypted_image.jpg")
    img_decrypted_array = np.array(img_decrypted)
    show_image(img_decrypted_array, "Decrypted Image")

import cv2

def sudoku_video(video_path):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Get the frames per second (fps) of the video
    fps = video.get(cv2.CAP_PROP_FPS)

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (int(video.get(3)), int(video.get(4))))

    perm_size = 9
    perm = [[4, 7, 1, 9, 8, 5, 3, 6, 2], [8, 9, 2, 7, 3, 6, 1, 5, 4], [5, 6, 3, 1, 2, 4, 8, 7, 9], [1, 5, 6, 8, 9, 2, 4, 3, 7], [2, 3, 8, 4, 1, 7, 5, 9, 6], [9, 4, 7, 6, 5, 3, 2, 1, 8], [6, 1, 9, 3, 4, 8, 7, 2, 5], [7, 8, 5, 2, 6, 1, 9, 4, 3], [3, 2, 4, 5, 7, 9, 6, 8, 1]]
    perm = np.array(perm)-1

    cnt = 0
    while(video.isOpened()):
        ret, frame = video.read()
        if ret==True:
            # Convert the frame to an image
            img = Image.fromarray(frame)
            cnt+=1
            #st.image(frame, use_column_width=True, caption="Image")
            # Save the image
            img.save(f"vid/frame_{cnt}.jpg")
            image_path = f"vid/frame_{cnt}.jpg"
            # Apply threshold_image function
            enc_and_dec.threshold_image(image_path, randomNumber=40)
            enc_and_dec.pad_and_shuffle_image("threshold_image.jpg", SudokuSize=9)
            sudoku("padded_and_shuffled_image.jpg")
            rotate_image("shuffled_image.jpg")
           
            # Load the unshuffled image
            unshuffled_img = Image.open("rotated_image.jpg")
            unshuffled_frame = np.array(unshuffled_img)
            # Write the unshuffled frame to the output video
            out.write(unshuffled_frame)
            unshuffled_img.save(f"vid/frame_enc_{cnt}.jpg")
            st.image(unshuffled_frame, use_column_width=True, caption="Image")

        else:
            break

    # Release everything when job is finished
    st.write(cnt)
    video.release()
    out.release()

    return 'output.mp4'

def sudoku_photo():
    # Streamlit app
    st.title("Sudoku-based encryption and decryption algorithm")
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        image_path = "uploaded_image.jpg"
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        Kn = [[4, 7, 1, 9, 8, 5, 3, 6, 2], [8, 9, 2, 7, 3, 6, 1, 5, 4], [5, 6, 3, 1, 2, 4, 8, 7, 9], [1, 5, 6, 8, 9, 2, 4, 3, 7], [2, 3, 8, 4, 1, 7, 5, 9, 6], [9, 4, 7, 6, 5, 3, 2, 1, 8], [6, 1, 9, 3, 4, 8, 7, 2, 5], [7, 8, 5, 2, 6, 1, 9, 4, 3], [3, 2, 4, 5, 7, 9, 6, 8, 1]]
        randomNumber = 40
        SudokuSize = 9
        img = Image.open(image_path)
        height, width = img.size
        with open('keys.txt', 'w') as f:
            f.write(f'{height}\n')
            f.write(f'{width}\n')
            f.write(f'{SudokuSize}\n')
            f.write(f'{Kn}\n')
        process_image(image_path)

def sudoku_video_parent():
    video_path = "video.mp4"
    st.title("Sudoku-based encryption and decryption algorithm")
    uploaded_file = st.file_uploader("Choose a video...", type="mp4")
    if uploaded_file is not None:
        video_path = "uploaded_video.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        output_video_path = sudoku_video(video_path)
        st.video("uploaded_video.mp4")

#sudoku_video_parent()
sudoku_photo()