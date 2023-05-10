import streamlit as st
import base64
import torch
import io
from scipy.io.wavfile import write
from PIL import Image
import numpy as np

import commons
import utils
import subprocess

# Run the shell script
subprocess.call('./startup.sh', shell=True)
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


def text_to_speech(text):
    stn_tst = get_text(text, hps)
    with torch.no_grad():
        x_tst = stn_tst.unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
        audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1.2)[0][
            0, 0].data.float().numpy()

    # Convert the numpy array to a WAV file
    wav_bytes = io.BytesIO()
    write(wav_bytes, hps.data.sampling_rate, audio)
    wav_bytes.seek(0)
    wav_data = wav_bytes.read()

    # Encode the WAV data in base64 format
    wav_base64 = base64.b64encode(wav_data).decode()
    return wav_base64

# Load the trained model
hps = utils.get_hparams_from_file("./configs/jp_base.json")
hps.model_dir = 'logs/jp_base'
pretrained_model = f'{hps.model_dir}/model.pth'

net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model)
_ = net_g.eval()
_ = utils.load_checkpoint(pretrained_model, net_g, None)


st.set_page_config(page_title="Text-to-Speech Demo")
st.title("Text-to-Speech Demo")

# Text Input
text = st.text_input("Enter Text", "")

if st.button("Convert to Speech"):
    # Call text_to_speech function
    speech = text_to_speech(text)

    # Convert base64 string to audio
    wav_data = base64.b64decode(speech)
    with io.BytesIO(wav_data) as stream:
        audio = stream.read()

    # Display audio
    st.audio(audio)
