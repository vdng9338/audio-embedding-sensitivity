import time
import sys

if len(sys.argv) < 2:
    print(f"Usage: python3 {sys.argv[0]} <embedding>")
    print(f"<embedding> can be 'openl3', 'panns' or 'clap'")
    sys.exit(1)

starttime = time.perf_counter()

import librosa
import numpy as np

embedding_name = sys.argv[1]
if embedding_name == 'openl3':
    import openl3
    import openl3.models
    import soundfile as sf
elif embedding_name == "panns":
    from panns_inference import AudioTagging
elif embedding_name == "clap":
    from msclap import CLAP
else:
    print(f"Unknown embedding {embedding_name}")
    sys.exit(1)

import os
import shutil
import h5py


######################################################
# Set this variable to the directory containing all  #
# experiment files (~/audio-embedding-sensitivity in #
# the README)                                        #
######################################################
basedir = "/home/USER/audio-embedding-sensitivity"

ori_base = f'{basedir}/datasets/IRMAS-TrainingData/'
irmas_files = librosa.util.find_files(ori_base)

if embedding_name == "openl3":
    model = openl3.models.load_audio_embedding_model(input_repr="mel128", content_type="music", embedding_size=512)
elif embedding_name == "panns":
    model = AudioTagging(checkpoint_path=None, device="cuda") # change to "cpu" to use CPU
else:
    model = CLAP(version='2023', use_cuda=True) # change to False to use CPU

X = []
keys = []

print(f"Computing {embedding_name} embeddings", file=sys.stderr, flush=True)

for i, fn in enumerate(irmas_files):
    if embedding_name == "clap":
        embedding = model.get_audio_embeddings([fn]) # TODO Batch processing
        embedding = embedding.cpu().numpy()
    elif embedding_name == "openl3":
        audio, sr = sf.read(fn)
        embeddings, _ = openl3.get_audio_embedding(audio, sr, model=model, verbose=False, batch_size=32)
    else:
        audio, _ = librosa.core.load(fn, sr=32000, mono=True)
        audio = audio[None, :]
        _, embedding = model.inference(audio) # TODO Batch processing

    if embedding_name != "openl3":
        X.append(embedding[0])
        keys.append(os.path.splitext(os.path.basename(fn))[0])
    else:
        X.extend(embeddings)
        key = os.path.splitext(os.path.basename(fn))[0]
        keys.extend([key] * len(embeddings))

    if (i+1)%1000 == 0:
        print(f"{i+1}/{len(irmas_files)}", file=sys.stderr, flush=True)

X = np.asarray(X, dtype=np.float32 if embedding_name != "openl3_orig" else np.float16)

DATA = h5py.File(f'embeddings/embeddings.h5', mode='a')
features_path = f'irmas/{embedding_name}/features'
keys_path = f'irmas/{embedding_name}/keys'
if features_path in DATA:
    del DATA[features_path]
if keys_path in DATA:
    del DATA[keys_path]
DATA[f'irmas/{embedding_name}/features'] = X
DATA[f'irmas/{embedding_name}/keys'] = keys
DATA.close()



endtime = time.perf_counter()

print(f"Done! Took {endtime-starttime}s overall", file=sys.stderr, flush=True)
