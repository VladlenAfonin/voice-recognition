import os
import features
import ml
import numpy as np

from sklearn.mixture import GaussianMixture
from scipy.io.wavfile import read


DATASET_PATH = 'development_set'
TRAIN_FILE_NAMES = 'train_data_paths.txt'
TEST_FILE_NAMES = 'test_data_paths.txt'


def get_gmms_for_speakers(file_paths) -> list[(GaussianMixture, str)]:
    """Create and train GMMs for file paths given (typically  extracted from a
        text file).

    :param file_paths: File paths to search for data
    :return: List if trained GMMs
    :rtype: list[GaussianMixture]
    """
    
    gmms_with_labels = []

    n_vectors_extracted = 1
    for path in file_paths:
        path = path.strip()

        speaker_features = np.array([])

        sample_rate, audio = read(os.path.join(DATASET_PATH, path))
        speaker_vector = features.get_features(sample_rate, audio)

        if speaker_features.size != 0:
            speaker_features = np.vstack((speaker_features, speaker_vector))
        else:
            speaker_features = speaker_vector
        
        if n_vectors_extracted == 5:
            gmm = ml.create_gmm(speaker_features)
            gmms_with_labels.append((gmm, path.split('-')[0]))

            n_vectors_extracted = 0

        n_vectors_extracted += 1
    
    return gmms_with_labels

def test_gmms_for_speakers(
    file_paths,
    gmms: list[GaussianMixture],
    labels: list[str]):

    predictions = []

    for path in file_paths:
        path = path.strip()

        sample_rate, audio = read(os.path.join(DATASET_PATH, path))
        speaker_vector = features.get_features(sample_rate, audio)

        log_likelihoods = np.zeros(len(gmms))

        for i in range(len(gmms)):
            scores = gmms[i].score(speaker_vector)
            log_likelihoods[i] = scores.sum()
        
        winner_id = np.argmax(log_likelihoods)
    
        predictions.append(path.split('-')[0] == labels[winner_id])

    return np.array(predictions)


def main():
    with open(TRAIN_FILE_NAMES, 'r') as file_paths:
        gmms_with_labels = get_gmms_for_speakers(file_paths)
    
    gmms, labels = map(list, zip(*gmms_with_labels))

    with open(TEST_FILE_NAMES, 'r') as file_paths:
        results = test_gmms_for_speakers(file_paths, gmms, labels)
    
    print(f'Accuracy: {len(results[results == True]) / len(results)}')

if __name__ == '__main__':
    main()
