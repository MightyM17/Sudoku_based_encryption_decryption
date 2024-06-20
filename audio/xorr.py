import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from scipy.io import wavfile

sample_rate, audio_data = wavfile.read('taunt.wav')

block_size = 9  # For a 9x9 Sudoku
padded_length = int(np.ceil(len(audio_data) / block_size) * block_size)
padded_audio = np.pad(audio_data, (0, padded_length - len(audio_data)), 'constant')

audio_blocks = padded_audio.reshape(-1, block_size)
print(audio_blocks.flatten()) 

sudoku_board = [
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

def threshold_and_shuffle_blocks(blocks, sudoku_board):
    for i in range(blocks.shape[0]):
        for j in range(blocks.shape[1]):
            blocks[i, j] = np.bitwise_xor(blocks[i, j], sudoku_board[i % 9][j % 9])
    blocks = np.transpose(blocks)
    return blocks

encrypted_blocks = threshold_and_shuffle_blocks(audio_blocks, sudoku_board)
encrypted_audio = encrypted_blocks.flatten()
print(encrypted_audio)
encrypted_audio = (encrypted_audio * np.max(np.abs(audio_data))).astype(np.int16)
wavfile.write('encrypted_audio.wav', sample_rate, encrypted_audio)

def reverse_threshold_and_shuffle_blocks(blocks, sudoku_board):
    for i in range(blocks.shape[0]):
        for j in range(blocks.shape[1]):
            # Applying XOR again with the same sudoku_board values to decrypt
            blocks[i, j] = np.bitwise_xor(blocks[i, j], sudoku_board[i % 9][j % 9])
    return blocks

sample_rate, audio_data = wavfile.read('encrypted_audio.wav')
decrypted_blocks = reverse_threshold_and_shuffle_blocks(audio_blocks, sudoku_board)
decrypted_audio = decrypted_blocks.flatten()
print(decrypted_audio)
#decrypted_audio = (decrypted_audio * np.max(np.abs(encrypted_audio))).astype(np.int16)

wavfile.write('decrypted_audio.wav', sample_rate, decrypted_audio)

def calculate_snr(original_signal, noise_signal):
    original_signal = original_signal.astype(np.float32)
    noise_signal = noise_signal.astype(np.float32)
    
    # Normalize the noise signal
    max_amplitude = np.max(np.abs(noise_signal))
    if max_amplitude != 0:
        noise_signal /= max_amplitude
    max_amplitude = np.max(np.abs(original_signal))
    if max_amplitude != 0:
        original_signal /= max_amplitude

    power_original = np.mean(np.square(original_signal))
    power_noise = np.mean(np.square(noise_signal))

    snr_db = 10 * np.log10(power_original / power_noise)

    return snr_db

def shuffle_audio_blocks(audio_path, perm):
    sample_rate, audio_data = wavfile.read(audio_path)

    block_size = len(perm)
    num_blocks = len(audio_data) // block_size

    # Create a new array for shuffled audio to avoid in-place modification issues
    shuffled_audio = np.zeros_like(audio_data)

    for i in range(num_blocks):
        start_idx = i * block_size
        end_idx = start_idx + block_size
        # Apply permutation to each block
        shuffled_block = audio_data[start_idx:end_idx][perm]
        shuffled_audio[start_idx:end_idx] = shuffled_block

    # Handle any remaining samples that don't fit into a full block
    if len(audio_data) % block_size != 0:
        remaining_samples_start_idx = num_blocks * block_size
        shuffled_audio[remaining_samples_start_idx:] = audio_data[remaining_samples_start_idx:]

    wavfile.write('shuffled_audio.wav', sample_rate, shuffled_audio)

def unshuffle_audio_blocks(audio_path, perm):
    sample_rate, audio_data = wavfile.read(audio_path)

    block_size = len(perm)
    num_blocks = len(audio_data) // block_size

    # Create a new array for shuffled audio to avoid in-place modification issues
    shuffled_audio = np.zeros_like(audio_data)
    perm = np.argsort(perm)

    for i in range(num_blocks):
        start_idx = i * block_size
        end_idx = start_idx + block_size
        # Apply permutation to each block
        shuffled_block = audio_data[start_idx:end_idx][perm]
        shuffled_audio[start_idx:end_idx] = shuffled_block

    # Handle any remaining samples that don't fit into a full block
    if len(audio_data) % block_size != 0:
        remaining_samples_start_idx = num_blocks * block_size
        shuffled_audio[remaining_samples_start_idx:] = audio_data[remaining_samples_start_idx:]

    wavfile.write('unshuffled_audio.wav', sample_rate, audio_data)

sudoku_board = [
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

sudoku_board = np.array(sudoku_board).flatten() - 1
shuffle_audio_blocks("taunt.wav", sudoku_board)
unshuffle_audio_blocks("shuffled_audio.wav", sudoku_board)

# Function to plot waveform
def plot_waveform(signal, sample_rate, title):
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(signal, sr=sample_rate)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()

# Function to calculate and print audio features
def print_audio_features(signal, sample_rate):
    spectral_centroid = librosa.feature.spectral_centroid(y=signal, sr=sample_rate)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sample_rate)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=signal)
    rms_energy = librosa.feature.rms(y=signal)

    print(f"Spectral Centroid: {np.mean(spectral_centroid)}")
    print(f"Spectral Bandwidth: {np.mean(spectral_bandwidth)}")
    print(f"Zero Crossing Rate: {np.mean(zero_crossing_rate)}")
    print(f"RMS Energy: {np.mean(rms_energy)}")

# Load the original audio
original_signal, sample_rate = librosa.load("CantinaBand3.wav", sr=None)

# Assuming shuffle_audio_blocks has been called and 'shuffled_audio.wav' is created
shuffled_signal, _ = librosa.load("shuffled_audio.wav", sr=None)

enc_signal, _ = librosa.load("encrypted_audio.wav", sr=None)
# Plot waveforms
plot_waveform(original_signal, sample_rate, "Original Audio Waveform")
plot_waveform(shuffled_signal, sample_rate, "Shuffled Audio Waveform")
plot_waveform(enc_signal, sample_rate, "Encrypted Audio Waveform")

# Print audio features for original audio
print("Original Audio Features:")
print_audio_features(original_signal, sample_rate)

# Print audio features for shuffled audio
print("\nShuffled Audio Features:")
print_audio_features(shuffled_signal, sample_rate)

print("\nShuffled Audio Features:")
print_audio_features(enc_signal, sample_rate)

fs_orig, original_audio = wavfile.read('taunt.wav')
fs_shuffled, shuffled_audio = wavfile.read('shuffled_audio.wav')
fs_enc, enc_audio = wavfile.read('encrypted_audio.wav')
# Ensure both audio files have the same length
min_length = min(len(original_audio), len(shuffled_audio))
original_audio = original_audio[:min_length]
shuffled_audio = shuffled_audio[:min_length]

# Calculate SNR
snr_result = calculate_snr(original_audio, shuffled_audio)
print(f"SNR between original and shuffled audio: {snr_result} dB")

snr_result = calculate_snr(original_audio, enc_audio)
print(f"SNR between original and encrypted audio: {snr_result} dB")

import numpy as np


def mse(original_signal, noise_signal):
    return np.mean(np.square(original_signal - noise_signal))

# 3. Peak Signal to Noise Ratio (PSNR)
def psnr(original_signal, noise_signal):
    peak_signal = np.max(np.abs(original_signal))
    psnr = 20 * np.log10(peak_signal / np.sqrt(mse(original_signal, noise_signal)))
    return psnr


# Output the results
print(f"MSE: {mse(original_signal, shuffled_signal)}, PSNR: {psnr(original_audio, shuffled_signal)}")
print(f"MSE: {mse(original_signal, enc_signal)}, PSNR: {psnr(original_audio, enc_signal)}")