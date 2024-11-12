import mlx_whisper


def transcribe(audio_data):
    return mlx_whisper.transcribe(audio_data,
                      path_or_hf_repo="mlx-community/whisper-large-v3-turbo")
