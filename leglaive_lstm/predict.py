from leglaive_lstm.audio_processor import process_single_audio
import argparse
import numpy as np

from leglaive_lstm.config_rnn import *
from librosa.core import frames_to_time


def init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--input', type=str)
    args = parser.parse_args()
    return args


def predict_song(model_name, filename, cache=True):
    """
    Predict Voice Activity Regions for a given song.

    :param model_name: name of the trained model
    :param filename:  path to the music file to be predicted
    :param cache: flag to optimize heavy operations with caching in disk
    :return: Prediction: Raw probability for each frame of the MFCC of the input song with overlapping by the RNN settings
    """
    audio_name = str(filename).split('/')[-1]
    audio_name_prefix = '.'.join(str(filename).split('/')[:-1])
    cache_filename = PREDICTIONS_DIR / '{}.{}.{}.npy'.format(audio_name_prefix, audio_name, model_name)
    try:
        if not cache:
            raise IOError
        y_pred = np.load(cache_filename)
    except IOError:
        import os
        import sys
        from audio_processor import process_single_audio

        input_mel = process_single_audio(filename, cache=True)

        from keras.models import load_model

        # set gpu number
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        # load model
        loaded_model = load_model('./weights/rnn_' + model_name + '.h5')
        print("loaded model")
        print(loaded_model.summary())

        total_x = []

        x = input_mel
        for i in range(0, x.shape[1] - RNN_INPUT_SIZE, 1):
            x_segment = x[:, i: i + RNN_INPUT_SIZE]
            total_x.append(x_segment)

        total_x = np.array(total_x)
        try:
            mean_std = np.load("train_mean_std_" + model_name + '.npy')
            mean = mean_std[0]
            std = mean_std[1]
        except Exception:
            print("mean, std not found")
            sys.exit()

        total_x_norm = (total_x - mean) / std
        total_x_norm = np.swapaxes(total_x_norm, 1, 2)

        x_test = total_x_norm
        y_pred = loaded_model.predict(x_test, verbose=1)  # Shape=(total_frames,)

        print(y_pred)
        np.save(cache_filename, y_pred) if cache else None
    return y_pred


def frame_level_predict(model_name, filename, cache=True, plot=False):
    """
    Predict Voice Activity Regions at a Frame Level for a given song.
    For each frame of the MFCC a Voice Detection Probability is predicted, then the output have shape: (n_frames, 1)

    :param model_name: name of the trained model
    :param filename:  path to the music file to be predicted
    :param cache: flag to optimize heavy operations with caching in disk
    :param plot: flag to plot MFCCs and SVD in an aligned plot if GUI available.
    :return: (Time, Predictions): SVD probabilities at frame level with time markings
    """
    audio_name = str(filename).split('/')[-1]
    audio_name_prefix = '.'.join(str(filename).split('/')[:-1])
    serialized_filename = PREDICTIONS_DIR / '{}.{}.{}.csv'.format(audio_name_prefix, audio_name, model_name)
    mel = process_single_audio(filename, cache=cache)

    try:
        if not cache:
            raise IOError
        data = np.loadtxt(serialized_filename, delimiter=',')
        time = data[0]
        frame_level_y_pred = data[1]
        print("info: loaded serialized prediction")
    except Exception:

        # transform raw predictions to frame level
        y_pred = predict_song(model_name, filename, cache=cache)
        aligned_y_pred = [[] for _ in range(mel.shape[1])]
        for first_frame_idx, window_prediction in enumerate(y_pred):
            # for each prediction
            for offset, frame_prediction in enumerate(window_prediction):
                # accumulate overlapped predictions in a list
                aligned_y_pred[first_frame_idx+offset].append(frame_prediction[0])

        frame_level_y_pred = []
        for _, predictions in enumerate(aligned_y_pred[:-1]):
            # reduce the overlapped predictions to a single value
            frame_level_y_pred.append(min(predictions))

        time = frames_to_time(range(len(frame_level_y_pred)), sr=SR, n_fft=N_FFT2, hop_length=N_HOP2)
        np.savetxt(serialized_filename, np.asarray((time, frame_level_y_pred)), delimiter=",")
        print("info: saved serialized prediction")
    if plot:
        import matplotlib.pyplot as plt
        import librosa.display

        # plot stacked MFCCs
        plt.figure(figsize=(14, 5))
        plt.subplot(211)
        librosa.display.specshow(mel, sr=SR, x_axis='time', y_axis='hz', hop_length=N_HOP2)

        # plot frame level predictions
        plt.subplot(313)
        plt.plot(time, frame_level_y_pred)
        plt.xlabel("Time")
        plt.ylabel("Singing Voice Activation")
        plt.show()
        print("info: plotted")
    print('info: done')
    return time, frame_level_y_pred


if __name__ == "__main__":
    args = init()
    x, y = frame_level_predict(args.model_name, args.input, cache=True)
    print("info: plotting frame-wize predictions: ")
    print(y)
    print("info: plotting frame timestamps [seconds]: ")
    print(x)

