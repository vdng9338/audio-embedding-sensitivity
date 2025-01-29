from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import sys
import deem
import argparse
import multiprocessing
import h5py
from effect_params import effect_params_str_dict



def load_embedding_mp(iParam, B_feature_dir, embedding, meta_all, queue: multiprocessing.Queue):
    B_feature = deem.load_feature(B_feature_dir, embedding, meta_all)
    queue.put((iParam, B_feature))

def main():
    ######## SETTINGS ########
    parser = argparse.ArgumentParser(prog="combination_extract.py", description="Averages per-sample and groups embeddings corresponding to all audio effect parameters into one file for a given combination of audio embedding and audio effect.")
    parser.add_argument("-e", "--effect", choices=list(effect_params_str_dict), required=True)
    parser.add_argument("-j", "--num-threads", help="Number of parallel processes to load the embeddings into memory (default 2)", default=2, type=int)
    parser.add_argument("-b", "--embedding", help="Embedding to use", choices=["openl3", "panns", "clap"], required=True)
    args = parser.parse_args()

    effect = args.effect

    if effect in effect_params_str_dict:
        effect_params = effect_params_str_dict[effect]
    else:
        print(f"Error: unknown effect {effect}")
        sys.exit(1)

    if args.num_threads is None:
        num_threads = multiprocessing.cpu_count()
    else:
        num_threads = args.num_threads

    embedding = args.embedding

    ###### END SETTINGS ######


    meta_all = pd.read_csv("train_test_split.csv")


    ###### Load embeddings ######
    A_feature = deem.load_feature("embeddings/embeddings.h5", embedding, meta_all)
    (X_train_A, Y_train_A), (X_test_A, Y_test_A), _ = A_feature
    B_features = [None] * len(effect_params)
    B_load_procs: "list[multiprocessing.Process]" = []
    B_queue = multiprocessing.Queue()

    for iParam, param in enumerate(effect_params):
        if param is not None:
            B_feature_dir = f'embeddings/embeddings_{effect}_{param}.h5'
        else:
            B_feature_dir = 'embeddings/embeddings.h5'
        
        B_load_procs.append(multiprocessing.Process(target=load_embedding_mp, args=(iParam, B_feature_dir, embedding, meta_all, B_queue)))

    iProcessB = 0
    for i in range(min(len(B_load_procs), num_threads)):
        B_load_procs[i].start()
        iProcessB += 1

    for _ in tqdm(range(len(B_load_procs)), desc="Loading effected embeddings"):
        iParam, B_feature = B_queue.get()
        B_features[iParam] = B_feature
        if iProcessB < len(B_load_procs):
            B_load_procs[iProcessB].start()
            iProcessB += 1
    
    X_train_Bs = []
    X_test_Bs = []
    for (X_train_B, _), (X_test_B, _), _ in B_features:
        X_train_Bs.append(X_train_B)
        X_test_Bs.append(X_test_B)

    X_train_Bs = np.array(X_train_Bs)
    X_test_Bs = np.array(X_test_Bs)

    save_path = f"embeddings/grouped/{embedding}/{effect}.h5"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"Saving to {save_path}...")
    f = h5py.File(save_path, "w")
    f["train_A"] = X_train_A
    f["test_A"] = X_test_A
    f["train_Bs"] = X_train_Bs
    f["test_Bs"] = X_test_Bs
    f.close()


if __name__ == "__main__":
    main()