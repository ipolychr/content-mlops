import librosa
import numpy as np


class Augmentation:
    """
    Data augmentation is the process by which we create new synthetic data samples by adding small perturbations on our initial training set.
    To generate syntactic data for audio, we can apply noise injection, shifting time, changing pitch and speed.
    The objective is to make our model invariant to those perturbations and enhance its ability to generalize.
    In order to this to work adding the perturbations must conserve the same label as the original training sample.
    """
    def __init__(self, rate:float, pitch_factor:float):
        self.rate = rate
        self.pitch_factor = pitch_factor

    @staticmethod
    def noise(data):
        noise_amp = 0.035 * np.random.uniform() * np.amax(data)
        data = data + noise_amp * np.random.normal(size=data.shape[0])
        return data

    def stretch(self, data):
        return librosa.effects.time_stretch(data, self.rate)

    @staticmethod
    def shift(data):
        shift_range = int(np.random.uniform(low=-5, high=5) * 1000)
        return np.roll(data, shift_range)

    def pitch(self, data, sampling_rate):
        return librosa.effects.pitch_shift(data, sampling_rate, self.pitch_factor)
