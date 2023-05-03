from scipy.io.wavfile import read
import torch
import numpy as np
import os
from multiprocessing import Pool
from tqdm import tqdm

# Change here
base="jp_dataset/basic5000/wav"

hann_window = {}
def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    # data, sampling_rate = librosa.load(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate

def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global hann_window
    dtype_device = str(y.dtype) + '_' + str(y.device)
    wnsize_dtype_device = str(win_size) + '_' + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[wnsize_dtype_device],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)

    spec = torch.view_as_real(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec

def get_audio(filename):
    max_wave_length = 32768.0
    filter_length = 1024
    hop_length = 256
    win_length = 1024
    audio, sampling_rate = load_wav_to_torch(filename)
    audio_norm = audio / max_wave_length
    audio_norm = audio_norm.unsqueeze(0)
    spec_filename = filename.replace(".wav", ".spec.pt")
    spec = spectrogram_torch(audio_norm, filter_length,
        sampling_rate, hop_length, win_length,
        center=False)
    spec = torch.squeeze(spec, 0)
    torch.save(spec, spec_filename)

if __name__=="__main__":
    waves = []
    batch_size = 16
    for wav_name in os.listdir(base):
        wav_path = os.path.join(base, wav_name)
        if wav_path.endswith(".wav"):
            waves.append(wav_path)
    with Pool(batch_size) as p:
        print(list((tqdm(p.imap(get_audio, waves), total=len(waves)))))
