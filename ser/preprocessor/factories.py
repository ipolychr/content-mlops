from ser.preprocessor.audio_preprocess import AudioPreprocess
from ser.preprocessor.augmentation import Augmentation
from ser.preprocessor.image_dataset import ImageDataset
from settings import sample_rate, rate, pitch_factor


def get_audio_preprocessor() -> AudioPreprocess:
    return AudioPreprocess(sample_rate=sample_rate)


def get_dataset() -> ImageDataset:
    return ImageDataset()

def get_data_augmentation() -> Augmentation:
    return Augmentation(rate=rate, pitch_factor=pitch_factor)