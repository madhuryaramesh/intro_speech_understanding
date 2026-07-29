import numpy as np
import librosa

def lpc(speech, frame_length, frame_skip, order):
    '''
    Perform linear predictive analysis.
    '''

    nframes = int((len(speech) - frame_length) / frame_skip)

    A = np.zeros((nframes, order + 1))
    excitation = np.zeros((nframes, frame_length))

    for i in range(nframes):
        start = i * frame_skip
        frame = speech[start:start + frame_length]

        # LPC coefficients
        a = librosa.lpc(frame, order=order)
        A[i] = a

        # Compute excitation (prediction error)
        e = np.copy(frame)
        for n in range(order, frame_length):
            prediction = 0
            for k in range(1, order + 1):
                prediction -= a[k] * frame[n - k]
            e[n] = frame[n] - prediction

        excitation[i] = e

    return A, excitation


def synthesize(e, A, frame_skip):
    '''
    Synthesize speech from excitation and LPC coefficients.
    '''

    order = A.shape[1] - 1
    nframes = A.shape[0]

    synthesis = np.zeros(frame_skip * nframes)

    for i in range(nframes):
        a = A[i]
        frame_e = e[i * frame_skip:(i + 1) * frame_skip]

        y = np.zeros(frame_skip)

        for n in range(frame_skip):
            y[n] = frame_e[n]

            for k in range(1, order + 1):
                if n - k >= 0:
                    y[n] -= a[k] * y[n - k]

        synthesis[i * frame_skip:(i + 1) * frame_skip] = y

    return synthesis


def robot_voice(excitation, T0, frame_skip):
    '''
    Generate robot-voice excitation.
    '''

    nframes = excitation.shape[0]

    gain = np.sqrt(np.mean(excitation ** 2, axis=1))

    e_robot = np.zeros(nframes * frame_skip)

    for i in range(nframes):
        start = i * frame_skip

        for n in range(frame_skip):
            if n % T0 == 0:
                e_robot[start + n] = gain[i]

    return gain, e_robot