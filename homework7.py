import numpy as np

def major_chord(f, Fs):
    '''
    Generate a one-half-second major chord, based at frequency f, with sampling frequency Fs.

    @param:
    f (scalar): frequency of the root tone, in Hertz
    Fs (scalar): sampling frequency, in samples/second

    @return:
    x (array): a one-half-second waveform containing the chord

    A major chord is three notes, played at the same time:
    (1) The root tone (f)
    (2) A major third, i.e., four semitones above f
    (3) A major fifth, i.e., seven semitones above f
    '''
    duration = 0.5
    t = np.arange(0, duration, 1/Fs)

    # Frequencies
    f_major3 = f * (2 ** (4/12))
    f_fifth = f * (2 ** (7/12))

    # Chord waveform
    x = (
        np.cos(2 * np.pi * f * t) +
        np.cos(2 * np.pi * f_major3 * t) +
        np.cos(2 * np.pi * f_fifth * t)
    )

    return x


def dft_matrix(N):
    '''
    Create a DFT transform matrix, W, of size N.

    @param:
    N (scalar): number of columns in the transform matrix

    @result:
    W (NxN array): a matrix of dtype='complex' whose (k,n)^th element is:
           W[k,n] = cos(2*np.pi*k*n/N) - j*sin(2*np.pi*k*n/N)
    '''
    k = np.arange(N).reshape(N, 1)
    n = np.arange(N).reshape(1, N)

    W = np.cos(2 * np.pi * k * n / N) - 1j * np.sin(2 * np.pi * k * n / N)

    return W.astype(complex)


def spectral_analysis(x, Fs):
    '''
    Find the three loudest frequencies in x.

    @param:
    x (array): the waveform
    Fs (scalar): sampling frequency (samples/second)

    @return:
    f1, f2, f3: The three loudest frequencies (in Hertz)
      These should be sorted so f1 < f2 < f3.
    '''
    N = len(x)

    # FFT
    X = np.fft.fft(x)

    # Magnitude spectrum
    mag = np.abs(X)

    # Only positive frequencies
    mag = mag[:N // 2]
    freqs = np.arange(N // 2) * Fs / N

    # Find indices of three largest peaks
    indices = np.argsort(mag)[-3:]

    # Corresponding frequencies
    peak_freqs = np.sort(freqs[indices])

    return peak_freqs[0], peak_freqs[1], peak_freqs[2]