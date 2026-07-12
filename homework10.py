import numpy as np
import torch
import torch.nn as nn


def get_features(waveform, Fs):
    '''
    Get spectrogram features and frame labels.
    '''

    # ---------- Pre-emphasis ----------
    waveform = np.append(waveform[0], waveform[1:] - 0.97 * waveform[:-1])

    # ---------- Features ----------
    frame_length = int(0.004 * Fs)   # 4 ms
    step = int(0.002 * Fs)           # 2 ms

    frames = []
    for start in range(0, len(waveform) - frame_length + 1, step):
        frames.append(waveform[start:start + frame_length])

    frames = np.array(frames)

    spectrum = np.abs(np.fft.fft(frames, axis=1))
    features = spectrum[:, :frame_length // 2]

    # ---------- VAD ----------
    vad_length = int(0.025 * Fs)     # 25 ms
    vad_step = int(0.01 * Fs)        # 10 ms

    energies = []
    starts = []

    for start in range(0, len(waveform) - vad_length + 1, vad_step):
        frame = waveform[start:start + vad_length]
        energies.append(np.sum(frame ** 2))
        starts.append(start)

    energies = np.array(energies)
    threshold = 0.1 * np.max(energies)

    speech = energies > threshold

    labels = np.zeros(len(features), dtype=int)

    label = 1
    i = 0

    while i < len(speech):
        if speech[i]:
            start_sample = starts[i]

            while i < len(speech) and speech[i]:
                i += 1

            end_sample = starts[min(i - 1, len(starts) - 1)] + vad_length

            start_frame = start_sample // step
            end_frame = min(len(features), end_sample // step)

            labels[start_frame:end_frame] = label

            label += 1
        else:
            i += 1

    # Repeat each label five times
    labels = labels // 5

    return features.astype(np.float32), labels.astype(np.int64)


def train_neuralnet(features, labels, iterations):
    '''
    Train Sequential(LayerNorm, Linear).
    '''

    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    nfeats = features.shape[1]
    nlabels = int(np.max(labels)) + 1

    model = nn.Sequential(
        nn.LayerNorm(nfeats),
        nn.Linear(nfeats, nlabels)
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    lossvalues = np.zeros(iterations)

    for i in range(iterations):
        optimizer.zero_grad()

        outputs = model(x)

        loss = criterion(outputs, y)

        loss.backward()

        optimizer.step()

        lossvalues[i] = loss.item()

    return model, lossvalues


def test_neuralnet(model, features):
    '''
    Return posterior probabilities.
    '''

    x = torch.tensor(features, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(x)
        probabilities = torch.softmax(outputs, dim=1)

    return probabilities.detach().numpy()