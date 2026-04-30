import os
from pydub import AudioSegment
import numpy as np
import scipy.io.wavfile as wavfile
import wave
from pyannote.audio import Pipeline
from pyannote.core import notebook, Segment
import logging
import torch
import os
import noisereduce as nr
from pathlib import Path

base_dir = ".../hotline"

mp3_dir =  Path(base_dir) / "data" / "raw_audio" / "2023_Y"
wav_dir = Path(base_dir) / "data" / "mono_audio" / "2023_Y"

for files in os.listdir(mp3_dir):
    in_dir = os.path.join(mp3_dir, files)
    sound = AudioSegment.from_mp3(in_dir)
    one_sound = sound.split_to_mono()
    left_sound = one_sound[0]
    wav_file = os.path.join(wav_dir, "{:s}.wav".format(files[:-4]))
    left_sound.export(wav_file, format="wav")


##
logging.basicConfig(level=logging.DEBUG)

pipeline = Pipeline.from_pretrained(".../pyannote/voice-activity-detection/config.yaml")
pipeline.to(torch.device("cuda"))

audio_dir = Path(base_dir) / "data" / "audio" / "2023_Y"

# Ensure output folder exists
# os.makedirs(output_folder, exist_ok=True)

# Process each audio file in the input folder
for filename in os.listdir(wav_dir):
    if filename.endswith(".wav"):
        audio_path = os.path.join(wav_dir, filename)    

        print(f"Processing {audio_path}...")          
        output = pipeline(audio_path) 
        # Load the original audio file
        original_audio = AudioSegment.from_wav(audio_path)        
        # Create an empty AudioSegment for the concatenated result
        concatenated_audio = AudioSegment.empty()

        # Concatenate detected speech segments
        for speech in output.get_timeline().support():
            start_ms = int(speech.start * 1000)
            end_ms = int(speech.end * 1000)
            speech_segment = original_audio[start_ms:end_ms]
            concatenated_audio += speech_segment
            # print(f"speech: {speech.start:.2f} - {speech.end:.2f}")
        
        samples = np.array(concatenated_audio.get_array_of_samples())
        reduced_noise_samples = nr.reduce_noise(y=samples, sr=concatenated_audio.frame_rate)

        # Convert back to AudioSegment
        reduced_noise_audio = AudioSegment(
            reduced_noise_samples.tobytes(),
            frame_rate=concatenated_audio.frame_rate,
            sample_width=concatenated_audio.sample_width,
            channels=concatenated_audio.channels
        )

        # Loudness Peak Normalization
        normalized_audio = reduced_noise_audio.apply_gain(0 - reduced_noise_audio.max_dBFS)

        # Export the processed audio to the output folder
        output_path = os.path.join(audio_dir, os.path.splitext(filename)[0] + ".wav")
        normalized_audio.export(output_path, format="wav")
        print(f"Processed audio saved to {output_path}")

