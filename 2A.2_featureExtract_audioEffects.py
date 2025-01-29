import librosa
import numpy as np
import sys
from effect_params import effect_params_dict, effect_params_str_dict

if len(sys.argv) < 5:
    print(f"Usage: python3 {sys.argv[0]} <effect> <embedding> <from> <to>")
    print("<effect> can be one of bitcrush, gain, lowpass_cheby, reverb")
    print("<embedding> can be one of clap, openl3, panns")
    print("Set <from> to -1 to show the number of parameters in the grid")
    sys.exit(1)

effect = sys.argv[1]
embedding_name = sys.argv[2]
grid_from = int(sys.argv[3])
grid_to = int(sys.argv[4])

if effect in effect_params_dict:
    params = effect_params_dict[effect]
    params_str = effect_params_str_dict[effect]
else:
    print(f"Unknown effect {effect}")
    sys.exit(1)

if grid_from == -1:
    print(f"There are {len(params)} parameters in the grid")
    sys.exit(0)

###########################################################
# Replace this with the base directory containing all the #
# files related to the experiments                        #
###########################################################
basedir = "/home/user/audio-embedding-sensitivity"

ori_base = f'{basedir}/datasets/IRMAS-TrainingData/'
irmas_files = librosa.util.find_files(ori_base)

import time

starttime = time.perf_counter()

import os
import shutil
import h5py

if embedding_name == "clap":
    from msclap import CLAP
elif embedding_name == "openl3":
    import openl3
    import openl3.models
    import soundfile as sf
elif embedding_name == "panns":
    from panns_inference import AudioTagging
elif embedding_name == "none":
    pass
else:
    print(f"Unknown embedding {embedding_name}")
    sys.exit(1)

from pedalboard import Pedalboard, Bitcrush, Gain, Reverb
from pedalboard.io import AudioFile
import scipy.signal

for param, param_str in zip(params[grid_from:grid_to+1], params_str[grid_from:grid_to+1]):
    curr_starttime = time.perf_counter()
    audio_outpath = f'{basedir}/datasets/audio-embeddings/irmas-wav-{effect}_{param_str}'

    if effect == "bitcrush":
        board = Pedalboard([Bitcrush(param)])
        print(f"Applying {param_str}-bit bitcrushing to dataset", file=sys.stderr, flush=True)
    elif effect == "gain":
        board = Pedalboard([Gain(param)])
        print(f"Applying {param_str}dB gain to dataset", file=sys.stderr, flush=True)
    elif effect == "reverb":
        board = Pedalboard([Reverb(room_size=param)])
        print(f"Applying reverberation with room size {param_str} to dataset", file=sys.stderr, flush=True)
    elif effect == "lowpass_cheby":
        stopband_edge = param * 1.1
        passband_edge = param / 1.1
        order, Wn = scipy.signal.cheb2ord(passband_edge, stopband_edge, 3, 80, fs=44100)
        print(f"Applying order-{order} Chebyshev type II filter with stopband attenuation of 80dB "
              + f"and cutoff frequency {Wn} (wp={passband_edge}, ws={stopband_edge}, param={param}) to dataset", file=sys.stderr, flush=True)
        sos = scipy.signal.cheby2(order, 80, Wn, 'lowpass', output='sos', fs=44100)
    else:
        assert False, "Effect not recognized despite passing argument check"


    for i, fn in enumerate(irmas_files):
        with AudioFile(fn) as f:
            save_fn = os.path.join(audio_outpath, fn.replace(ori_base, ''))
            os.makedirs(os.path.dirname(save_fn), exist_ok=True)
    
            # Open an audio file to write to:
            with AudioFile(save_fn, 'w', f.samplerate, f.num_channels) as o:
            
                # Read one second of audio at a time, until the file is empty:
                if effect != "lowpass_cheby":
                    while f.tell() < f.frames:
                        chunk = f.read(f.samplerate)
                            
                        # Run the audio through our pedalboard:
                        effected = board(chunk, f.samplerate, reset=False)
                        
                        # Write the output to our output file:
                        o.write(effected)
                else:
                    all_audio = f.read(f.frames)
                    effected = scipy.signal.sosfilt(sos, all_audio)
                    o.write(effected)

        if (i+1)%1000 == 0:
            print(f"{i+1}/{len(irmas_files)}", file=sys.stderr, flush=True)

    if embedding_name == "clap":
        model = CLAP(version='2023', use_cuda=True) # change use_cuda to False to use CPU
    elif embedding_name == "openl3":
        model = openl3.models.load_audio_embedding_model(input_repr="mel128", content_type="music", embedding_size=512)
    elif embedding_name == "panns":
        model = AudioTagging(checkpoint_path=None, device="cuda") # device = "cpu" or "cuda"
    else:
        continue

    X = []
    keys = []

    effected_files = librosa.util.find_files(audio_outpath)

    print(f"Computing {embedding_name} embeddings", file=sys.stderr, flush=True)

    for i, fn in enumerate(effected_files):
        if embedding_name == "clap":
            embedding = model.get_audio_embeddings([fn])
            embedding = embedding.cpu().numpy()
        elif embedding_name == "openl3":
            audio, sr = sf.read(fn)
            embeddings, _ = openl3.get_audio_embedding(audio, sr, model=model, verbose=False, batch_size=32)
        else:
            audio, _ = librosa.core.load(fn, sr=32000, mono=True)
            audio = audio[None, :]
            _, embedding = model.inference(audio)
        
        if embedding_name != "openl3":
            X.append(embedding[0])
            keys.append(os.path.splitext(os.path.basename(fn))[0])
        else:
            X.extend(embeddings)
            key = os.path.splitext(os.path.basename(fn))[0]
            keys.extend([key] * len(embeddings))

        if (i+1)%1000 == 0:
            print(f"{i+1}/{len(effected_files)}", file=sys.stderr, flush=True)


    X = np.asarray(X, dtype=np.float32)
    # keys = np.asarray(keys, dtype='S31')

    DATA = h5py.File(f'embeddings/embeddings_{effect}_{param_str}.h5', mode='a')
    features_path = f'irmas/{embedding_name}/features'
    keys_path = f'irmas/{embedding_name}/keys'
    if features_path in DATA:
        del DATA[features_path]
    if keys_path in DATA:
        del DATA[keys_path]
    DATA[f'irmas/{embedding_name}/features'] = X
    DATA[f'irmas/{embedding_name}/keys'] = keys
    DATA.close()

    # Delete effected .wav files and individual embedding files to save space
    print(f"Deleting {audio_outpath}", file=sys.stderr, flush=True)
    shutil.rmtree(audio_outpath)

    curr_finishtime = time.perf_counter()
    print(f"Time taken for current parameter: {curr_finishtime-curr_starttime}s", file=sys.stderr, flush=True)
    print(f"Time elapsed since beginning: {curr_finishtime-starttime}s", file=sys.stderr, flush=True)
    print("", file=sys.stderr, flush=True)


endtime = time.perf_counter()

print(f"Done! Took {endtime-starttime}s overall", file=sys.stderr, flush=True)
