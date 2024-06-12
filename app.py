import streamlit as st
from PIL import Image
import numpy as np
import enc_and_dec
from s import sudoku
from sudoku import rotate_image, rotate_image_counter_clockwise
import matplotlib.pyplot as plt
import time
import cv2
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
from scipy.stats import entropy
from skimage.filters import sobel

def plot_color_histogram(img_array, title):
    color = ('b','g','r')
    plt.figure(figsize=(6, 6))
    for i, col in enumerate(color):
        histr = cv2.calcHist([img_array],[i],None,[256],[0,256])
        plt.plot(histr, color = col)
        plt.xlim([0,256])
    plt.title(title)
    st.pyplot(plt)

import pandas as pd

def analyze_image(img_array, title):
    # Calculate the mean and standard deviation of the pixel values for each channel
    mean_r = img_array[:, :, 0].mean()
    std_r = img_array[:, :, 0].std()

    mean_g = img_array[:, :, 1].mean()
    std_g = img_array[:, :, 1].std()

    mean_b = img_array[:, :, 2].mean()
    std_b = img_array[:, :, 2].std()

    # Create a DataFrame and display it in Streamlit
    data = {'Mean': [mean_r, mean_g, mean_b], 'Standard Deviation': [std_r, std_g, std_b]}
    df = pd.DataFrame(data, index=['Red', 'Green', 'Blue'])
    st.write(f"{title} - Pixel Intensity Analysis")
    st.dataframe(df)

def show_image(img_array, title):
    st.image(img_array, use_column_width=True, caption=title)

from skimage.util import img_as_ubyte

def analyze_image_glcm(img_array, title):
    # Convert the image to grayscale for texture analysis
    gray_img = rgb2gray(img_array)

    # Convert the floating point grayscale image to unsigned byte
    gray_img = img_as_ubyte(gray_img)

    # Calculate texture features from the gray-level co-occurrence matrix (GLCM)
    glcm = graycomatrix(gray_img, [1], [0], 256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

    # Calculate entropy
    img_entropy = entropy(img_array.ravel())

    # Apply edge detection
    edges = sobel(gray_img)
    edge_count = np.count_nonzero(edges)

    # Display the results
    st.write(f"{title} - Contrast: {contrast:.2f}, Energy: {energy:.2f}, Homogeneity: {homogeneity:.2f}, Entropy: {img_entropy:.2f}, Edge count: {edge_count}")

def process_image(image_path):
    times = {}
    # Load the image
    img = Image.open(image_path)
    img_array = np.array(img)
    show_image(img_array, "Original Image")

    st.write("Encryption")
    # Apply threshold_image function
    start_time = time.time()
    enc_and_dec.threshold_image(image_path, randomNumber=40)
    load_time = time.time() - start_time
    times['Threshold_Time'] = load_time
    img_threshold = Image.open("threshold_image.png")
    img_threshold_array = np.array(img_threshold)
    show_image(img_threshold_array, "Threshold Image")

    # Apply pad_and_shuffle_image function
    start_time = time.time()
    seed = enc_and_dec.pad_and_shuffle_image("threshold_image.png", SudokuSize=9)
    load_time = time.time() - start_time
    times['Pad_Shuffle_Time'] = load_time
    with open('keys.txt', 'a') as f:
        f.write(f'{seed}\n')
    img_padded_shuffled = Image.open("padded_and_shuffled_image.png")
    img_padded_shuffled_array = np.array(img_padded_shuffled)
    show_image(img_padded_shuffled_array, "Padded and Shuffled Image")

    start_time = time.time()
    sudoku("padded_and_shuffled_image.png")
    load_time = time.time() - start_time
    times['Sudoku_Time'] = load_time
    img_sudukoed = Image.open("shuffled_image.png")
    img_sudukoed_array = np.array(img_sudukoed)
    show_image(img_sudukoed_array, "Image after Sudoku Transformation")

    start_time = time.time()
    rotate_image("shuffled_image.png")
    load_time = time.time() - start_time
    times['Rotate_Time'] = load_time
    image_rotated = Image.open("rotated_image.png")
    image_rotated_array = np.array(image_rotated)
    show_image(image_rotated_array, "Image after Rotation")

    st.write("Decryption")

    start_time = time.time()
    rotate_image_counter_clockwise("rotated_image.png")
    load_time = time.time() - start_time
    times['Decrypt_Rotate_Time'] = load_time
    image_rotated_cc = Image.open("rotated_image_counter_clockwise.png")
    image_rotated_array_cc = np.array(image_rotated_cc)
    show_image(image_rotated_array_cc, "Image after Rotation")

    start_time = time.time()
    im = Image.open("unshuffled_image.png")
    im_ar = np.array(im)
    times['Decrypt_Sudoku_Time'] = times['Sudoku_Time']
    show_image(im_ar, "Revese Sudoku Transformation")

    start_time = time.time()
    enc_and_dec.unshuffle_and_unpad_image("unshuffled_image.png", original_width=img.width, original_height=img.height, SudokuSize=9)
    times['Decrypt_Unpad_Unshuffle_Time'] = times['Sudoku_Time']
    img_unshuffled_unpadded = Image.open("unshuffled_and_unpadded_image.png")
    img_unshuffled_unpadded_array = np.array(img_unshuffled_unpadded)
    show_image(img_unshuffled_unpadded_array, "Unshuffled and Unpadded Image")

    start_time = time.time()
    enc_and_dec.decrypt_threshold_image("unshuffled_and_unpadded_image.png", image_path, randomNumber=40)
    load_time = time.time() - start_time
    times['Decrypt_Threshold_Time'] = load_time
    img_decrypted = Image.open("decrypted_image.png")
    img_decrypted_array = np.array(img_decrypted)
    show_image(img_decrypted_array, "Decrypted Image")

    plt.figure(figsize=(10, 5))
    plt.plot(list(times.keys()), list(times.values()), marker='o')
    plt.xlabel('Step')
    plt.ylabel('Time (seconds)')
    plt.title('Time taken for each step')
    st.pyplot(plt)
    analyze_image(img_array, "Original Image")
    analyze_image(img_threshold_array, "Envrypted Image")
    plot_color_histogram(img_array, "Original Image Color Histogram")
    plot_color_histogram(image_rotated_array, "Threshold Image Color Histogram")
    analyze_image_glcm(img_array, "Original Image")
    analyze_image_glcm(img_threshold_array, "Encrypted Image")  

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
            img.save(f"vid/frame_{cnt}.png")
            image_path = f"vid/frame_{cnt}.png"
            # Apply threshold_image function
            enc_and_dec.threshold_image(image_path, randomNumber=40)
            enc_and_dec.pad_and_shuffle_image("threshold_image.png", SudokuSize=9)
            sudoku("padded_and_shuffled_image.png")
            rotate_image("shuffled_image.png")
           
            # Load the unshuffled image
            unshuffled_img = Image.open("rotated_image.png")
            unshuffled_frame = np.array(unshuffled_img)
            # Write the unshuffled frame to the output video
            out.write(unshuffled_frame)
            unshuffled_img.save(f"vid/frame_enc_{cnt}.png")
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
    uploaded_file = st.file_uploader("Choose an image...", type="png")
    if uploaded_file is not None:
        image_path = "uploaded_image.png"
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