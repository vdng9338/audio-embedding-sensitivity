from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import sys
import deem
import argparse
import multiprocessing
from sklearn.decomposition import PCA
import json
from effect_params import effect_params_dict, effect_params_str_dict


def load_embedding_mp(iParam, B_feature_dir, embedding, meta_all, queue: multiprocessing.Queue):
    B_feature = deem.load_feature(B_feature_dir, embedding, meta_all)
    queue.put((iParam, B_feature))

def main():
    ######## SETTINGS ########
    parser = argparse.ArgumentParser(prog="desensitize_classif.py", description="Performs desensitization of audio effects using several methods.")
    parser.add_argument("-b", "--embedding", choices=["openl3", "panns", "clap"], required=True, help="Embedding to use.")
    parser.add_argument("-e", "--effect", choices=list(effect_params_dict), required=True, help="Audio effect to perform desensitization with.")
    parser.add_argument("-m", "--desensitize-methods", help="Comma-separated list of desensitization methods to use. Currently, "
                        + "choose from '',lda,avgdirproj,pcaproj_nonnorm,pcaproj_norm,cca,cca_samplewise_svd.",
                        default=",lda,avgdirproj,pcaproj_nonnorm,pcaproj_norm,cca,cca_samplewise_svd")
    parser.add_argument("-t", "--svd-thr", help="SVD threshold (in terms of proportion of highest singular value) for CCA samplewise SVD desensitization", type=float, default=0.3)
    parser.add_argument("-k", "--num-threads-classif", help="Number of parallel processes to run for classification (default as many as number of CPU cores)", default=None, type=int)
    args = parser.parse_args()

    embedding = args.embedding
    effect = args.effect

    if effect in effect_params_dict:
        effect_params_str = effect_params_str_dict[effect]
    else:
        print(f"Error: unknown effect {effect}")
        sys.exit(1)

    
    if args.num_threads_classif is None:
        num_threads_classif = multiprocessing.cpu_count()
    else:
        num_threads_classif = args.num_threads_classif

    desensitize_methods = ["-" + meth for meth in args.desensitize_methods.split(",")]
    has_nodesensitize = "-" in desensitize_methods
    if has_nodesensitize:
        new_desensitize_methods = [""]
    else:
        new_desensitize_methods = []
    desensitize_methods = new_desensitize_methods + [m for m in desensitize_methods if m != "-"]

    ###### END SETTINGS ######


    meta_all = pd.read_csv("train_test_split.csv")


    ###### Load embeddings ######
    A_feature = deem.load_feature("embeddings/embeddings.h5", embedding, meta_all)
    (X_train_A, Y_train_A), (X_test_A, Y_test_A), _ = A_feature
    B_features = [None] * len(effect_params_str)
    B_load_procs: "list[multiprocessing.Process]" = []
    B_queue = multiprocessing.Queue()

    for iParam, param in enumerate(effect_params_str):
        if param is not None:
            B_feature_dir = f'embeddings/embeddings_{effect}_{param}.h5'
        else:
            B_feature_dir = 'embeddings/embeddings.h5'
        
        B_load_procs.append(multiprocessing.Process(target=load_embedding_mp, args=(iParam, B_feature_dir, embedding, meta_all, B_queue)))

    iProcessB = 0
    for i in range(min(len(B_load_procs), num_threads_classif)):
        B_load_procs[i].start()
        iProcessB += 1

    for _ in tqdm(range(len(B_load_procs)), desc="Loading effected embeddings"):
        iParam, B_feature = B_queue.get()
        B_features[iParam] = B_feature
        if iProcessB < len(B_load_procs):
            B_load_procs[iProcessB].start()
            iProcessB += 1

    Y_train_Bs = [B_feature[0][1] for B_feature in B_features]
    Y_test_Bs = [B_feature[1][1] for B_feature in B_features]


    # Preparing to save results
    dir_info = effect
    save_dir = os.path.join(f'./results', dir_info)
    model_dir = os.path.join(f'./models', dir_info)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)


    results_deb = deem.deem()

    results_path = os.path.join(save_dir, f'result_{embedding}.csv')
    if os.path.isfile(results_path):
        results_deb.load_results(results_path)
    print(f"**** results_deb has {len(results_deb.result_all)} results ****")

    deformdir_path = os.path.join(model_dir, f"deformdir_{embedding}.pkl")
    if os.path.isfile(deformdir_path):
        results_deb.load_deformdirs(deformdir_path)
    clf_path = os.path.join(model_dir, f"models_{embedding}.pkl")
    if os.path.isfile(clf_path):
        results_deb.load_clfs(clf_path)


    # Extract positive samples from each dataset for each instrument
    X_inst_As = dict()
    X_inst_Bss = dict()
    for instrument in deem.instrument_map:
        X_inst_As[instrument] = X_train_A[Y_train_A==instrument]
        X_inst_Bs = []
        for (X_train_B, Y_train_B), _, _ in B_features:
            X_inst_Bs.append(X_train_B[Y_train_B == instrument])
        X_inst_Bss[instrument] = X_inst_Bs
    X_all_B = []
    for (X_train_B, _), (X_test_B, _), _ in B_features:
        X_all_B.extend([X_train_B, X_test_B])
    X_all = [X_train_A, X_test_A] + X_all_B

    # Compute PCA variances of dataset A
    pca_A = PCA()
    pca_A.fit(X_train_A)
    ref_variances = pca_A.explained_variance_

    for imethod, desensitize_method in enumerate(desensitize_methods):
        for iinst, instrument in enumerate(deem.instrument_map):
            print(f"------- Desensitization method '{desensitize_method}' ({imethod+1}/{len(desensitize_methods)}), instrument {instrument} ({iinst+1}/{len(deem.instrument_map)}) -------")

            
            ###### Desensitization ######
            print("Desensitizing...")
            desensitized_list, sensitize_dir = deem.desensitize(desensitize_method, X_inst_As[instrument], X_inst_Bss[instrument], X_all, ref_variances=ref_variances, svd_thr=args.svd_thr)
            print("Succeeded")
            
            if desensitize_method == "-cca_samplewise_svd":
                desensitize_method_key = f"-cca_samplewise_svd-{args.svd_thr}"
            else:
                desensitize_method_key = desensitize_method
            X_train_A_desensitized = desensitized_list[0]
            X_test_A_desensitized = desensitized_list[1]
            X_train_B_desensitized = []
            X_test_B_desensitized = []
            for j in range(1, len(B_features)+1):
                X_train_B_desensitized.append(desensitized_list[2*j])
                X_test_B_desensitized.append(desensitized_list[2*j+1])
            
            print("Storing deformation direction")
            results_deb.store_deformdir(desensitize_method_key, instrument, sensitize_dir)

            
            
            ###### Instrument classification ######

            process_list: "list[multiprocessing.Process]" = []
            queue = multiprocessing.Queue()

            if desensitize_method == "-cca_samplewise_svd":
                embedding_params = json.dumps({"svd_thr": args.svd_thr})
            else:
                embedding_params = None
            
            # One effect parameter at a time
            process_list.append(multiprocessing.Process(target=deem.instrument_classification_task, args=(instrument, X_train_A_desensitized, Y_train_A, X_test_A_desensitized, Y_test_A, desensitize_method, "orig", "orig", embedding, embedding_params, queue)))
            for i, param in enumerate(effect_params_str):
                # Train A, test B
                process_list.append(multiprocessing.Process(target=deem.instrument_classification_task, args=(instrument, X_train_A_desensitized, Y_train_A, X_test_B_desensitized[i], Y_test_Bs[i], desensitize_method, "orig", param, embedding, embedding_params, queue)))
                # Train B, test A
                process_list.append(multiprocessing.Process(target=deem.instrument_classification_task, args=(instrument, X_train_B_desensitized[i], Y_train_Bs[i], X_test_A_desensitized, Y_test_A, desensitize_method, param, "orig", embedding, embedding_params, queue)))
                # Train B, test B
                process_list.append(multiprocessing.Process(target=deem.instrument_classification_task, args=(instrument, X_train_B_desensitized[i], Y_train_Bs[i], X_test_B_desensitized[i], Y_test_Bs[i], desensitize_method, param, param, embedding, embedding_params, queue)))

            # A randomly sampled effect parameter per sample
            param_ids_train = np.random.randint(0, len(effect_params_str), size=len(X_train_B_desensitized[0]))
            param_ids_test = np.random.randint(0, len(effect_params_str), size=len(X_test_B_desensitized[0]))
            X_train_B = np.array([X_train_B_desensitized[p][i] for i, p in enumerate(param_ids_train)])
            X_test_B = np.array([X_test_B_desensitized[p][i] for i, p in enumerate(param_ids_test)])
            # Train A, test B
            # TODO Check that all the Y_test_Bs are actually equal
            process_list.append(multiprocessing.Process(target=deem.instrument_classification_task, args=(instrument, X_train_A_desensitized, Y_train_A, X_test_B, Y_test_Bs[0], desensitize_method, "orig", "randommix", embedding, embedding_params, queue)))
            # Train B, test A
            process_list.append(multiprocessing.Process(target=deem.instrument_classification_task, args=(instrument, X_train_B, Y_train_Bs[0], X_test_A_desensitized, Y_test_A, desensitize_method, "randommix", "orig", embedding, embedding_params, queue)))
            # Train B, test B
            process_list.append(multiprocessing.Process(target=deem.instrument_classification_task, args=(instrument, X_train_B, Y_train_Bs[0], X_test_B, Y_test_Bs[0], desensitize_method, "randommix", "randommix", embedding, embedding_params, queue)))

            deem.set_verbose(True)

            iProcess = 0
            for i in range(min(len(process_list), num_threads_classif)):
                process_list[i].start()
                iProcess += 1

            numFinished = 0
            while numFinished < len(process_list):
                inst, train_name, test_name, result, clf = queue.get()
                numFinished += 1
                if result is not None:
                    print(f"* Train {train_name}, test {test_name} finished: {numFinished}/{len(process_list)} *")
                    results_deb.store_result(result)
                    results_deb.store_clf(desensitize_method_key, inst, train_name, clf)
                    #print(f"**** results_deb has {len(results_deb.result_all)} results ****")
                else:
                    print(f"* Train {train_name}, test {test_name} failed ({numFinished}/{len(process_list)}) *")
                if iProcess < len(process_list):
                    process_list[iProcess].start()
                    iProcess += 1


    results_deb.results_to_csv(results_path)
    results_deb.save_deformdirs(deformdir_path)
    results_deb.save_clfs(clf_path)


if __name__ == "__main__":
    main()
