import numpy as np

from python_speech_features import mfcc
from sklearn.preprocessing import scale


def get_mfcc(sample_rate: int, audio: np.ndarray):
    """Extract MFCC from audio file.

    :param sample_rate: Audio file sample rate
    :type sample_rate: int
    :param audio: Array containing audio data
    :type audio: np.ndarray
    """

    features = mfcc(
        audio,
        samplerate=sample_rate,
        winlen=0.025,
        winstep=0.01,
        numcep=20,
        appendEnergy=True)
    
    features = scale(features)
    
    return features

def get_deltas(mfcc_features: np.ndarray, n_deltas: int=20) -> np.ndarray:
    """Calculate delta-features out of MFCC features.

    :param mfcc_features: MFCC features
    :type mfcc_features: np.ndarray
    :param n_deltas: Number of deltas to calculate, defaults to 20
    :type n_deltas: int, optional
    :return: Delta-features
    :rtype: np.ndarray
    """

    n_rows, _ = mfcc_features.shape

    deltas = np.zeros((n_rows, n_deltas))

    N = 2

    for i in range(n_rows):
        index = []

        j = 1
        while j <= N:
            if i - j < 0:
                first = 0
            else:
                first = i - j
            
            if i + j > n_rows - 1:
                second = n_rows - 1
            else:
                second = i + j
        
            index.append((second, first))

            j += 1
        
        deltas[i] = (mfcc_features[index[0][0]] - mfcc_features[index[0][1]] + \
            (2 * (mfcc_features[1][0] - mfcc_features[index[1][1]]))) / 10
    
    return deltas

def get_features(sample_rate: int, audio: np.ndarray):
    """Extract features from audio file.  This extracts MFCC and their
        derivatives.

    :param sample_rate: Audio file sample rate
    :type sample_rate: int
    :param audio: Array containing audio data
    :type audio: np.ndarray
    """

    mfcc_features = get_mfcc(sample_rate, audio)
    delta_features = get_deltas(mfcc_features)

    return np.hstack((mfcc_features, delta_features))
