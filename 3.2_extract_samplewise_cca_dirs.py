import h5py
import numpy as np
import sys
from tqdm import tqdm
from sklearn.preprocessing import QuantileTransformer
from sklearn.cross_decomposition import CCA
from effect_params import effect_params_str_dict

def main():
    if len(sys.argv) < 3:
        print(f"Usage: python3 {sys.argv[0]} <embedding> <effect>")
        print('<embedding> can be one of "openl3", "panns" or "clap"')
        print('<effect> can be one of', ", ".join([f'"{eff}"' for eff in effect_params_str_dict]))
        sys.exit(1)
    
    embedding, effect = sys.argv[1:3]

    if embedding not in ("openl3", "panns", "clap"):
        print(f"Unknown embedding {embedding}")
        sys.exit(1)

    if effect in effect_params_str_dict:
        params = effect_params_str_dict[effect]
    else:
        print(f"Unknown effect {effect}")
        sys.exit(1)

    param_rank = np.arange(len(params))
    QT = QuantileTransformer(n_quantiles=len(params))
    param_rank_qt = QT.fit_transform(param_rank[None].T).squeeze()

    h5files = []
    for param in params:
        h5files.append(h5py.File(f"embeddings/averaged/{embedding}/embeddings_{effect}_{param}.h5", 'r'))

    out_dirs = []

    num_samples = len(h5files[0]["X_train"])

    for i in tqdm(range(num_samples), "Computing CCA directions"):
        embeddings = np.vstack([file["X_train"][i] for file in h5files])
        CCA_t = CCA(n_components=1, scale=False)
        CCA_t.fit(embeddings, param_rank_qt)
        proj_dir = CCA_t.x_weights_[:, 0]
        out_dirs.append(proj_dir)

    out = h5py.File(f"embeddings/averaged/{embedding}/ccadirs_{effect}.h5", "w")
    out["cca_dirs"] = np.array(out_dirs)
    out.close()

if __name__ == "__main__":
    main()
