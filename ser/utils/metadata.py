from ffmpeg import probe

from ser.data_model.metadata import Metadata


def extract_audio_metadata(source_audio_path: str) -> Metadata:
    return Metadata(**probe(filename=source_audio_path))
