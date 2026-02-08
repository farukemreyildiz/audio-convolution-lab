# Audio Convolution Lab

A Python-based signal processing project that demonstrates discrete-time convolution using both a custom implementation and NumPy’s built-in functions.

The project supports:
- Real-time microphone recording
- Custom convolution (from scratch, nested loops)
- NumPy convolution comparison
- Echo / system response simulation
- Time-domain waveform visualization
- Spectrogram analysis

This project was developed for understanding convolution, LTI systems, and audio signal processing fundamentals.

---

## Features

### 1. Manual Convolution Implementation
- Double loop convolution (no library usage)
- Index-aware discrete-time convolution
- Educational, algorithm-level implementation

### 2. NumPy Comparison
- Validation with `numpy.convolve`
- Result and duration comparison

### 3. Real-time Audio Processing
- 5s and 10s microphone recording
- System response (impulse response) generation
- Echo-like effects via convolution
- Playback of processed signals

### 4. Visualization
- Waveform plots
- Spectrograms
- Discrete-time stem plots

---

## Technologies

- Python
- NumPy
- Matplotlib
- sounddevice
- scipy

---

## Installation

```bash
pip install numpy matplotlib sounddevice scipy
