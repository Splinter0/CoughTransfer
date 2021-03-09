import os
import librosa
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import tensorflow_hub as hub

os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.compat.v1.Session(config=config)

MODEL_LINK = "https://tfhub.dev/google/yamnet/1"

SR = 16000
DATASET = "/home/splinter/Desktop/EE/CoughDetection/"
LABELS = ["cough", "not"]
SAMPLE_SIZE = int(np.ceil(0.96*SR)) #Taken from the documentation of Yamnet
PAD = SAMPLE_SIZE
TEST_SPLIT = 0.25

extractor = hub.load(MODEL_LINK)

def extract(signal):
    with tf.device("/gpu:0"):
        scores, embeddings, log_mel_spectrogram = extractor(signal)
    return embeddings

def make_process(path):
    data = []
    for i, label in enumerate(LABELS):
        print("Processing "+label+"...")
        for audio in tqdm(os.listdir(path+label)):
            if os.path.splitext(audio)[-1] != ".wav":
                continue
            try:
                signal, rate = librosa.load(path+label+"/"+audio, sr=SR)
            except:
                continue

            if len(signal) == 0:
                continue

            embeddings = extract(signal)
            for frame in embeddings:
                data.append([frame, i])

    return data

if __name__ == '__main__':
    data = make_process(DATASET)
    sp = int(np.ceil(TEST_SPLIT*len(data)))
    np.random.shuffle(data)
    test = data[:sp]
    train = data[sp:]
    print("Training set:", len(train))
    print("Testing set:", len(test))
    np.save("train.npy", train)
    np.save("test.npy", test)
