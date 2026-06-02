import soundfile as sf
import torch
import torchaudio
import torch.nn.functional as F

import config as CONFIG

from audio_dataset import LABEL_MAP


mel_transform = torchaudio.transforms.MelSpectrogram( # computes STFT internally
    sample_rate=CONFIG.SAMPLE_RATE,
    n_fft=CONFIG.N_FFT,
    hop_length=CONFIG.HOP,
    n_mels=CONFIG.N_MELS
)

to_db = torchaudio.transforms.AmplitudeToDB()


class STFT:
    def __init__(self):
        self.window = torch.hann_window(CONFIG.N_FFT)

    def forward(self, wav):
        return torch.stft(
            wav,
            n_fft=CONFIG.N_FFT,
            hop_length=CONFIG.HOP,
            window=self.window,
            return_complex=True
        )

    def inverse(self, spec):
        return torch.istft(
            spec,
            n_fft=CONFIG.N_FFT,
            hop_length=CONFIG.HOP,
            window=self.window
        )

def mel_to_stft(explanation, stft_spec):
    return F.interpolate(
        input=torch.from_numpy(explanation).squeeze(-1).unsqueeze(0).unsqueeze(0), # (1, 1, n_mels, time)
        size=stft_spec.shape,
        mode="bilinear",
        align_corners=False
    )

def apply_explanation(explanation, wav):
    stft = STFT()
    stft_spec = stft.forward(wav).squeeze(0)

    mask = mel_to_stft(explanation["explanation"], stft_spec)
    mask = mask.squeeze(0).squeeze(0)
    
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8) # normalize to [0, 1]
    mask = mask ** 2.5 # sharpen the mask to make important regions more distinct
    # mask = torch.exp(mask) - 1
    # mask = mask / mask.max()

    mag = stft_spec.abs() # strength of each time-frequency bin

    threshold = torch.quantile(mask, 0.85)
    binary_mask = (mask >= threshold).float()
    mag_weighted = mag * binary_mask # only keep the top % most important bins according to the explanation
    #mag_weighted = mag * (mask > mask.mean()).float()
    
    phase = stft_spec.angle() # where in the waveform cycle each bin is (important for reconstructing the audio)
    alpha = 3 # Adjust this to control the strength of the explanation

    mag_masked = mag_weighted * (1.0 + alpha * mask)
    spec_masked = mag_masked * torch.exp(1j * phase)

    #spec_masked = stft_spec * mask
    wav_out = stft.inverse(spec_masked.unsqueeze(0))

    print(f"Mask stats - Min: {mask.min()}, Max: {mask.max()}, Mean: {mask.mean()}")
    print(f"Mean absolute difference: {(mag_masked - mag).abs().mean()}")

    return wav_out


def explain_audio(model, path):
    model.eval()
    device = next(model.parameters()).device

    emotion = path.split("/")[-1].split("_")[0]
    #print(f"Emotion: {emotion} ({EMOTION_MAP[emotion]})")
    # y = torch.tensor(EMOTION_MAP[emotion]).unsqueeze(0).to(device)

    #wav, sr = torchaudio.load(path)

    wav, sr = sf.read(path, dtype="float32")
    wav = torch.from_numpy(wav)

    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    else:
        wav = wav.T

    # Resample to model SR
    if sr != CONFIG.SAMPLE_RATE:
         wav = torchaudio.functional.resample(wav, sr, CONFIG.SAMPLE_RATE)
    # Convert stereo -> mono
    wav = wav.mean(dim=0, keepdim=True)
    # Pad/crop to fixed length
    if wav.shape[1] > CONFIG.NUM_SAMPLES:
        wav = wav[:, :CONFIG.NUM_SAMPLES]
    else:
        wav = F.pad(wav, (0, CONFIG.NUM_SAMPLES - wav.shape[1]))

    # stft = STFT()
    # stft_spec = stft.forward(wav).squeeze(0)
    x = to_db(mel_transform(wav)).unsqueeze(1).to(device)

    #explanation = get_attribution(model, x, y)
    explanation = model.explain(x)

    # print(f"Explanation shape: {torch.from_numpy(explanation['explanation']).squeeze(-1).unsqueeze(0).unsqueeze(0).shape}")
    # print(f"STFT spec shape: {stft_spec.shape}")

    # plt.imshow(x.squeeze(0).squeeze(0).cpu().detach().numpy(), aspect="auto", origin="lower")
    # plt.show()

    # plt.imshow(explanation["explanation"].squeeze(), aspect="auto", origin="lower")
    # plt.show()

    wav_out = apply_explanation(explanation, wav)

    return LABEL_MAP[emotion], explanation["prediction"], explanation["explanation"], wav, wav_out
