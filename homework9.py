import numpy as np

def VAD(waveform, Fs):
    '''
    Extract waveform segments whose frame energy is greater than
    10% of the maximum frame energy.
    '''
    frame_length = int(0.025 * Fs)   # 25 ms
    step = int(0.01 * Fs)            # 10 ms

    energies = []
    starts = []

    for start in range(0, len(waveform) - frame_length + 1, step):
        frame = waveform[start:start + frame_length]
        energies.append(np.sum(frame ** 2))
        starts.append(start)

    energies = np.array(energies)
    threshold = 0.1 * np.max(energies)

    speech = energies > threshold

    segments = []
    i = 0
    while i < len(speech):
        if speech[i]:
            start = starts[i]
            while i < len(speech) and speech[i]:
                i += 1
            end = starts[min(i - 1, len(starts) - 1)] + frame_length
            segments.append(waveform[start:end])
        else:
            i += 1

    return segments


def segments_to_models(segments, Fs):
    '''
    Convert each speech segment into an average log spectrum.
    '''
    frame_length = int(0.004 * Fs)   # 4 ms
    step = int(0.002 * Fs)           # 2 ms

    models = []

    for seg in segments:

        # pre-emphasis
        pre = np.append(seg[0], seg[1:] - 0.97 * seg[:-1])

        frames = []
        for start in range(0, len(pre) - frame_length + 1, step):
            frames.append(pre[start:start + frame_length])

        frames = np.array(frames)

        spectrum = np.abs(np.fft.fft(frames, axis=1))

        spectrum = spectrum[:, :frame_length // 2]

        logspec = 20 * np.log10(np.maximum(1e-6, spectrum))

        model = np.mean(logspec, axis=0)

        models.append(model)

    return models


def recognize_speech(testspeech, Fs, models, labels):
    '''
    Recognize each speech segment using cosine similarity.
    '''
    test_segments = VAD(testspeech, Fs)
    test_models = segments_to_models(test_segments, Fs)

    sims = np.zeros((len(models), len(test_models)))

    for i, model in enumerate(models):
        for j, test in enumerate(test_models):
            sims[i, j] = np.dot(model, test) / (
                np.linalg.norm(model) * np.linalg.norm(test) + 1e-12
            )

    test_outputs = []

    for j in range(len(test_models)):
        best = np.argmax(sims[:, j])
        test_outputs.append(labels[best])

    return sims, test_outputs