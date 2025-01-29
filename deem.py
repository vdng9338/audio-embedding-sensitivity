import numpy as np
from tqdm import tqdm
import pandas as pd
import pickle
import h5py
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from sklearn.decomposition import PCA
import random
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from utils import equal_or_bothna
import json
from datetime import datetime
from typing import Optional, Iterable

from sklearn.preprocessing import QuantileTransformer
from sklearn.cross_decomposition import CCA

from multiprocessing import Queue
import traceback

param_grid = {'LR_param': {'C':[10**k for k in range(-8, 4, 1)]}, 
              'scoring': 'roc_auc', 'cv': 3, 'random_state': 42}

with open("instrument_map.json", "r") as f:
    instrument_map = json.load(f)

with open("genre_map.json", "r") as f:
    genre_map = json.load(f)

verbose = False

def set_verbose(v):
    global verbose
    verbose = v

class deem():
    """DEEM: DEsensitizing pre-trained audio EMbeddings
    
    This class manages saving the results, models and deformation directions."""
    
    def __init__(self):
        self.verbose = verbose
        self.ref_variances = None
        self.result_all = []
        self.deform_dirs = dict()
        """desensitizing_method -> instrument -> deformation dir"""
        self.clfs = dict()
        """desensitizing_method -> instrument -> effect param -> classifier"""
    
    def load_results(self, path):
        result_df = pd.read_csv(path)
        self.result_all = result_df.to_dict(orient='records')
    
    def cleanup_results(self, effect):
        new_results = []
        for result in self.result_all:
            if not (
                (effect == "bitcrush" and result["train_set"] in ["1", "2", "3"] + [str(k) for k in range(16, 32)]) or
                ("cca" in result["embedding"] and datetime.strptime(result["last_updated"], "%Y-%m-%d %H:%M:%S.%f") < datetime.strptime("2024-08-27 16:00:00.000", "%Y-%m-%d %H:%M:%S.%f"))
            ):
                new_results.append(result)
            else:
                print("Discarding result", result)
        self.result_all = new_results

    def load_deformdirs(self, path):
        with open(path, "rb") as f:
            self.deform_dirs = pickle.load(f)
    
    def load_clfs(self, path):
        with open(path, "rb") as f:
            self.clfs = pickle.load(f)
    
    def store_result(self, result):
        for i, old_result in enumerate(self.result_all):
            if old_result['instrument'] == result['instrument'] and old_result['train_set'] == result['train_set'] and \
                old_result['test_set'] == result['test_set'] and old_result['embedding'] == result['embedding'] and \
                equal_or_bothna(old_result['embedding_params'], result['embedding_params']):
                self.result_all[i] = result
                return
        self.result_all.append(result)
    
    def results_to_csv(self, path=None):
        result_df = pd.DataFrame(self.result_all, columns=['instrument', 'train_set', 'test_set', 'precision', 'recall', 'f1_score', 'support', 'accuracy', 'roc_auc', 'ap', 'extra_info', 'embedding', 'embedding_params', 'last_updated'])
        return result_df.to_csv(path, index=False)
    
    def store_deformdir(self, desensitizing_method, inst, dir):
        if desensitizing_method not in self.deform_dirs:
            self.deform_dirs[desensitizing_method] = dict()
        self.deform_dirs[desensitizing_method][inst] = dir

    def store_clf(self, desensitizing_method, inst, param, clf):
        if desensitizing_method not in self.clfs:
            self.clfs[desensitizing_method] = dict()
        if inst not in self.clfs[desensitizing_method]:
            self.clfs[desensitizing_method][inst] = dict()
        self.clfs[desensitizing_method][inst][param] = clf

    def save_deformdirs(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.deform_dirs, f)
    
    def save_clfs(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.clfs, f)



def create_result():
    new_result = {
        'instrument': None,
        'train_set': None,
        'test_set': None,
        'precision': None,
        'recall': None,
        'f1_score': None,
        'support': None,
        'accuracy': None,
        'roc_auc': None,
        'ap': None,
        'extra_info': None,
        'embedding': None,
        'embedding_params': None,
        'last_updated': None
    }
    return new_result

def train_classifier(X_train_desensitized: np.ndarray, Y_train: np.ndarray, instrument: str):
    X_train_inst_A, train_is_inst_A, _, _ = create_train_sets_one(instrument, X_train_desensitized, Y_train)
    return compute_classifier(X_train_inst_A, train_is_inst_A)

def test_classifier(clf: "GridSearchCV[LogisticRegression]", X_test_desensitized: np.ndarray, Y_test: np.ndarray, instrument: str):
    test_is_inst = Y_test==instrument
    # predict
    Y_pred = clf.predict(X_test_desensitized)
    # Get prediction scores for the positive class
    Y_pred_scores = clf.predict_proba(X_test_desensitized)[:, 1]
    
    model_auc = roc_auc_score(test_is_inst, Y_pred_scores)
    model_ap = average_precision_score(test_is_inst, Y_pred_scores)
    
    # record the result for each instrument
    classif_report = classification_report(test_is_inst, Y_pred, output_dict=True)
    report = pd.DataFrame(classif_report)['True']
    report_accuracy = classif_report['accuracy']
    
    result = create_result()
    result['precision'] = report['precision']
    result['recall'] = report['recall']
    result['f1_score'] = report['f1-score']
    result['support'] = report['support']
    result['accuracy'] = report_accuracy
    result['roc_auc'] = model_auc
    result['ap'] = model_ap
    result['last_updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    return result

def instrument_classification(instrument: str, X_train_desensitized: np.ndarray, Y_train: np.ndarray, X_test_desensitized: np.ndarray, Y_test: np.ndarray):
    clf = train_classifier(X_train_desensitized, Y_train, instrument)
    ret = test_classifier(clf, X_test_desensitized, Y_test, instrument)
    ret["instrument"] = instrument
    return ret, clf.best_estimator_

def instrument_classification_task(instrument: str, X_train_desensitized: np.ndarray, Y_train: np.ndarray, X_test_desensitized: np.ndarray, Y_test: np.ndarray, desensitizing_method: str, train_name: str, test_name: str, embedding: str, embedding_params: Optional[str], queue: Queue, extra_info: Optional[str] = None):
    try:
        result, clf = instrument_classification(instrument, X_train_desensitized, Y_train, X_test_desensitized, Y_test)
        result["train_set"] = train_name
        result["test_set"] = test_name
        result["embedding"] = embedding + desensitizing_method
        result["embedding_params"] = embedding_params
        result["extra_info"] = extra_info
        queue.put((instrument, train_name, test_name, result, clf))
    except BaseException as e:
        traceback.print_exception(e)
        queue.put((instrument, train_name, test_name, None, None))
        

def load_feature(file_path: Optional[str], embedding: str, meta: pd.DataFrame):
    """Load features from embedding file using given metadata.
    
    Args
    -----
    file_path: str, optional
        Path to embedding file. If None, will only return instruments and genres (no embeddings).
    embedding: str
        Which embedding to load ("openl3" or "vggish" for instance)
    meta: pandas.DataFrame
        Pandas DataFrame containing for each file desired a row with "instrument", "genre", "file_name" and "split" (="train"/"test").
        
    Returns
    -------
    (X_train, Y_train):
        X_train: numpy.ndarray
            numpy.ndarray of shape (N, D), where N is the number of samples and D the embedding dimension, containing for each file of the training
            set the average of the embeddings of that file
        Y_train: numpy.ndarray, dtype=str (??)
            numpy.ndarray of shape (N,) containing the dominant instrument of each file of the training set
    (X_test, Y_test):
        Same as (X_train, Y_train), but for the test set.
    (genre_train, genre_test):
        2-uple of `numpy.ndarray`s containing the genre of each sample of the training and the test set, respectively
    """

    if file_path is not None:
        embeddings = h5py.File(file_path, "r")
    else:
        embeddings = None
        X_train = None
        X_test = None

    ###### IRMAS data ######
    if embeddings is not None:
        feature = np.array(embeddings["irmas"][embedding]["features"])
        keys_ori = np.array(embeddings["irmas"][embedding]["keys"])
        try:
            # some machine may need the following line of code; please comment out if not the case for you
            keys_ori = np.array([str(k, 'utf-8') for k in keys_ori])  
        except:
            keys_ori = keys_ori

        key_clip = np.unique(keys_ori)

        feature_map = {}
        for key in key_clip:
            feature_map[key] = []
        for key, x in zip(keys_ori, feature):
            feature_map[key].append(x)
        
        feature_clip = np.zeros((len(key_clip), feature.shape[1]))
        for i, key in enumerate(key_clip):
            feature_clip[i] = np.mean(np.vstack(feature_map[key]), axis=0)

    key_train = list(meta.loc[(meta['split'] == 'train')]['file_name'])
    key_test = list(meta.loc[(meta['split'] == 'test')]['file_name'])

    if embeddings is not None:
        idx_map = dict([(key_clip[i], i) for i in range(len(key_clip))])
        idx_train = [idx_map[item] for item in key_train]
        idx_test = [idx_map[item] for item in key_test]

        # cast the idx_* arrays to numpy structures
        idx_train = np.asarray(idx_train)
        idx_test = np.asarray(idx_test)

        # use the split indices to partition the features, labels, and masks
        X_train = feature_clip[idx_train,:]
        X_test = feature_clip[idx_test]

    Y_train = meta.loc[(meta['split'] == 'train')]['instrument'].values
    Y_test = meta.loc[(meta['split'] == 'test')]['instrument'].values

    genre_train = meta.loc[(meta['split'] == 'train')]['genre'].values
    genre_test = meta.loc[(meta['split'] == 'test')]['genre'].values

    return (X_train, Y_train), (X_test, Y_test), (genre_train, genre_test)


def resample_data(feature, Y_train, instrument, num, return_indices=False):
    """
    Select "num" number of negative samples for a given positive instrument, balanced by instrument.

    Args
    ------
    feature: tensor
        Original pre-trained embedding features.
    Y_train: array-like
        Instrument of each sample of `feature`.
    num: int 
        Number of negative samples to return.
    return_indices: bool, optional
        Whether to return the list of indices of selected negative samples.
    """

    feature_all = np.zeros((0, feature.shape[1]), dtype=feature.dtype)   # (294, )
    indices_all = []
    other_instruments = list(instrument_map)
    other_instruments.remove(instrument)

    for inst in other_instruments:

        feature_inst =  feature[Y_train == inst]
        inst_indices = np.argwhere(Y_train == inst)[:, 0]
        
        random.seed(param_grid['random_state'])
        idx_sam = random.sample(list(np.arange(feature_inst.shape[0])), num // len(other_instruments))  
        idx_sam = np.array(idx_sam)

        feature_all = np.vstack((feature_all, feature_inst[idx_sam]))  # (83, )
        indices_all.extend(inst_indices[idx_sam])

    if return_indices:
        return feature_all, np.array(indices_all)
    else:
        return feature_all

    
def create_train_sets_one(instrument, X_train, Y_train, idx_false=None):
    """Create a training dataset for the given instrument from the given embeddings. The
    returned set will contain all the positive samples and as many negative samples as positive samples,
    balanced by instrument class.
    
    Args
    ------
    instrument: string
        The instrument to consider.
    X_train: tensor
        The features of the entire training set.
    Y_train: tensor[str]
        The instruments of each sample of the entire training set.

    Returns
    -------
    X_train_inst: tensor
        Features of the resulting training dataset.
    train_is_inst: tensor[bool]
        Whether each sample from X_train_inst belongs to the given instrument.
    X_train_true: tensor
        Features of the positive samples from X_train_inst.
    idx_false: tensor[int]
        Indices of the negative samples that were selected.
    """
    train_is_inst = Y_train==instrument
    X_train_true = X_train[train_is_inst]
    X_train_false = X_train[~train_is_inst]
    Y_train_false = Y_train[~train_is_inst]
    
    if idx_false is None:
        X_train_false, idx_false = \
            resample_data(X_train_false, Y_train_false, instrument, X_train_true.shape[0], return_indices=True)  # 290
    else:
        X_train_false = X_train_false[idx_false]

    X_train_inst = np.vstack((X_train_true, X_train_false))  # 624
    train_is_inst = np.array([[True] * len(X_train_true) + [False] * len(X_train_false)]).reshape(-1,) # why reshape?

    return X_train_inst, train_is_inst, X_train_true, idx_false


def create_train_sets(instrument, A_feature, B_features, same_negatives=True):
    """Create training datasets (one for A, one for each parameter in B) for the given instrument from the embeddings of A, B.
    
    Args
    ------
    instrument: string
        The instrument to consider.
    A_feature: (X_train_A, Y_train_A), (X_test_A, Y_test_A), (genre_train_A, genre_test_A)
        Pre-trained embedding features and labels on first dataset (uncorrupted audio usually)
    B_feature: (X_train_B, Y_train_B), (X_test_B, Y_test_B), (genre_train_B, genre_test_B)
        Pre-trained embedding features and labels on second datasets (corrupted audio usually)
    same_negatives: bool, optional
        Whether to use the same negative indices as for dataset A to sample the negative samples for dataset B. Default True.

    Returns
    -------
    (X_train_inst_A, train_is_inst_A, X_train_A_true):
        X_train_inst_A: tensor
            Features of the training dataset from A.
        train_is_inst_A: tensor[bool]
            Whether each sample from X_train_inst_A belongs to the given instrument.
        X_train_A_true: tensor
            Features of the positive samples from X_train_inst_A.
    train_sets_B: list[(X_train_inst_B, train_is_inst_B, X_train_B_true)]
        Same as three previous return values, but for each parameter in B.
    """
    # Here, Y = instrument
    (X_train_A, Y_train_A), _, _ = A_feature
    
    X_train_inst_A, train_is_inst_A, X_train_A_true, idx_A_false = create_train_sets_one(instrument, X_train_A, Y_train_A)
    ret_B = []
    for (X_train_B, Y_train_B), _, _ in B_features:
        X_train_inst_B, train_is_inst_B, X_train_B_true, _ = create_train_sets_one(instrument, X_train_B, Y_train_B, idx_A_false if same_negatives else None)
        ret_B.append((X_train_inst_B, train_is_inst_B, X_train_B_true))

    return (X_train_inst_A, train_is_inst_A, X_train_A_true), ret_B


def lda_transform(X_inst_A, X_inst_Bs, to_transform):
    """Perform LDA using given samples and project out the domain separation direction
    for all tensors in to_transform.
    
    Args
    ------
    X_lda_A: tensor
        Array of features of the first dataset.
    X_lda_B: tensor
        Array of features of the second dataset.
    to_transform: iterable[tensor]
        A collection of tensors from the features of which to project out the domain separation direction."""

    X_train_conca = np.concatenate([X_inst_A] + list(X_inst_Bs))
    Ylda_A = np.zeros(len(X_inst_A))
    Ylda_B = np.ones(len(X_inst_Bs)*len(X_inst_A))

    Ylda_conca = np.hstack((Ylda_A, Ylda_B))

    LDA = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')
    LDA.fit(X_train_conca, Ylda_conca)

    v = LDA.coef_.copy()
    v /= np.sqrt(np.sum(v**2))
    v = v[0]
    A = np.outer(v, v)

    return list(map(lambda arr: arr.copy().dot(np.eye(len(A)) - A), to_transform)), v


    
def avgdir_proj(X_inst_A, X_inst_Bs, to_transform):
    """Project out the direction that separates the centroids of the two given datasets.
    Returns the dataset unmodified if centroids are less than 1e-10 apart from each other (in terms of L2 norm).
    
    Args
    ------
    X_centroid_A: tensor
        Features of the first dataset. (Only used to compute its centroid.)
    X_centroid_B: tensor
        Features of the second dataset. (Only used to compute its centroid.)
    to_transform: iterable[tensor]
        Tensors containing features to transform.
    
    Returns
    -------
    transformed: list[tensor]
        List containing transformed tensors (in same order as yielded by the input iterable)."""
    
    centroidA = np.average(X_inst_A, axis=0)
    centroidB = np.average(np.vstack(X_inst_Bs), axis=0)
    dir = centroidB-centroidA
    if np.linalg.norm(dir) < 1e-10:
        return list(to_transform)
    dir /= np.linalg.norm(dir)

    proj_matrix = np.eye(len(dir)) - dir[:,None]@dir[None]
    return list(map(lambda X: np.dot(X, proj_matrix), to_transform)), dir


def nonnorm_pca_proj(X_inst_A: np.ndarray, X_inst_Bs: Iterable[np.ndarray], to_transform: Iterable[np.ndarray]):
    """Given samples from the original dataset and each effected dataset, retrieves the deformation direction by PCA
    of all the displacements and projects it out from the to_transform embeddings.
    
    Args
    ------
    X_inst_A: tensor
        Embeddings of the samples from the original dataset.
    X_inst_Bs: iterable[tensor]
        Embeddings of the effected samples for each effect parameter.
    to_transform: iterable[tensor]
        Tensors containing features to transform.
        
    Returns
    -------
    transformed: list[tensor]
        List containing transformed tensors (in same order as yielded by the input variable).
    deform_dir: numpy.ndarray
        deformation direction that was projected out (first principal component of PCA)."""
    dim = X_inst_A.shape[1]
    displacements = np.vstack([X_inst_B - X_inst_A for X_inst_B in X_inst_Bs])
    pca = PCA()
    pca.fit(displacements)
    proj_matrix = np.eye(dim) - pca.components_[0][:,None]@pca.components_[0][None]
    
    return list(map(lambda X: np.dot(X, proj_matrix), to_transform)), pca.components_[0]

def norm_pca_proj(X_inst_A: np.ndarray, X_inst_Bs: Iterable[np.ndarray], to_transform: Iterable[np.ndarray], ref_variances):
    dim = X_inst_A.shape[1]
    displacements = np.vstack([X_inst_B - X_inst_A for X_inst_B in X_inst_Bs])
    pca = PCA()
    pca.fit(displacements)
    projdim = 0
    maxratio = 0
    for i in range(len(pca.explained_variance_)):
        ratio = pca.explained_variance_[i] / ref_variances[i]
        if ratio > maxratio:
            projdim = i
            maxratio = ratio
    proj_matrix = np.eye(dim) - pca.components_[projdim][:,None]@pca.components_[projdim][None]
    
    return list(map(lambda X: np.dot(X, proj_matrix), to_transform)), pca.components_[projdim]


def cca_proj(X_inst_Bs: Iterable[np.ndarray], to_transform: Iterable[np.ndarray]):
    dim = X_inst_Bs[0].shape[1]
    paths = np.array(X_inst_Bs).swapaxes(0, 1)
    paths_flat = np.reshape(paths, (paths.shape[0] * paths.shape[1], paths.shape[2]))
    param_flat = np.tile(np.arange(len(X_inst_Bs)), len(paths))
    QT = QuantileTransformer(n_quantiles=len(X_inst_Bs))
    param_flat_rank = QT.fit_transform(param_flat[None].T).squeeze()
    CCA_t = CCA(n_components=1, scale=False)
    CCA_t.fit(paths_flat, param_flat_rank)
    proj_dir = CCA_t.x_weights_[:, 0]
    #assert np.isclose(np.linalg.norm(proj_dir), 1.0), "Norm of CCA projection direction is not 1"
    proj_matrix = np.eye(dim) - proj_dir[:,None]@proj_dir[None]
    return list(map(lambda X: np.dot(X, proj_matrix), to_transform)), proj_dir

def cca_samplewise_svd_proj(X_inst_Bs: Iterable[np.ndarray], to_transform: Iterable[np.ndarray], threshold: float = 0.3):
    dim = X_inst_Bs[0].shape[1]
    paths = np.array(X_inst_Bs).swapaxes(0, 1)
    param_rank = np.arange(len(X_inst_Bs))
    QT = QuantileTransformer(n_quantiles=len(X_inst_Bs))
    param_rank_qt = QT.fit_transform(param_rank[None].T).squeeze()

    cca_dirs = []
    for path in paths:
        CCA_t = CCA(n_components=1, scale=False)
        CCA_t.fit(path, param_rank_qt)
        cca_dir = CCA_t.x_weights_[:, 0]
        cca_dirs.append(cca_dir)
    
    _, S, vh = np.linalg.svd(np.array(cca_dirs), full_matrices=False)
    proj_matrix = np.eye(dim)
    proj_dirs = np.array([], dtype=vh.dtype)
    for i in range(len(S)):
        if S[i]/S[0] < threshold:
            proj_dirs = vh[:i]
            break
        proj_matrix -= vh[i, :, None]@vh[i, None, :]
    return list(map(lambda X: np.dot(X, proj_matrix), to_transform)), proj_dirs


def check_desensitizing_method(desensitizing_method):
    """Checks that the desensitization method is well-formed and makes sense.
    Useful to update older code and spot mistakes which would have resulted
    otherwise in falling back to no desensitizing."""
    
    if desensitizing_method not in ("-lda", "-avgdirproj", "-pcaproj_nonnorm", "-pcaproj_norm", "-cca_samplewise_svd", "-cca", ""):
        raise ValueError(f"Unknown desensitizing method {desensitizing_method}")
    

def desensitize(desensitizing_method, X_inst_A, X_inst_Bs, to_transform, ref_variances=None, svd_thr = 0.3):
    deform_dir = None # May contain multiple deformation directions (LDA multiple desensitizing and CCA samplewise SVD desensitizing)
    if 'lda' in desensitizing_method:
        transformed, deform_dir = lda_transform(X_inst_A, X_inst_Bs, to_transform)
    elif '-avgdirproj' in desensitizing_method:
        transformed, deform_dir = avgdir_proj(X_inst_A, X_inst_Bs, to_transform)
    elif "-pcaproj_nonnorm" in desensitizing_method:
        transformed, deform_dir = nonnorm_pca_proj(X_inst_A, X_inst_Bs, to_transform)
    elif "-pcaproj_norm" in desensitizing_method:
        if ref_variances is None:
            raise ValueError("ref_variances must be set for normalized PCA projection")
        transformed, deform_dir = norm_pca_proj(X_inst_A, X_inst_Bs, to_transform, ref_variances)
    elif "-cca_samplewise_svd" in desensitizing_method:
        transformed, deform_dir = cca_samplewise_svd_proj(X_inst_Bs, to_transform, threshold=svd_thr)
    elif "-cca" in desensitizing_method:
        transformed, deform_dir = cca_proj(X_inst_Bs, to_transform)
    else:
        transformed = to_transform
    return transformed, deform_dir

def desensitize_mp(desensitizing_method, instrument, X_inst_A, X_inst_Bs, to_transform, queue: Queue, proj_matrix = None, ref_variances=None):
    print(f"Desensitizing '{desensitizing_method}' {instrument}")
    try:
        queue.put((instrument, desensitize(desensitizing_method, X_inst_A, X_inst_Bs, to_transform, proj_matrix=proj_matrix, ref_variances=ref_variances)))
        print(f"Finished '{desensitizing_method}' {instrument}")
    except BaseException as e:
        traceback.print_exception(e)
        print(f"'{desensitizing_method}' {instrument} failed")
        queue.put((instrument, None))


def compute_classifier(X_train_clf, Y_train_clf):
    # initialize and a logistic regression model
    LRmodel = LogisticRegression(random_state=param_grid['random_state'], penalty='l2', 
                                    solver='liblinear', class_weight='balanced')
    
    # hyperparameter tunning for logistic regression model
    clf =  GridSearchCV(LRmodel, param_grid=param_grid['LR_param'], cv=param_grid['cv'],
                            scoring=param_grid['scoring'])    
    
    # fit the model
    clf.fit(X_train_clf, Y_train_clf)

    return clf


def instrument_classification_noreport(desensitizing_method, A_feature, B_features):
    """ 
    Binary classification of instruments. Returns the classification results on the test set.

    The negative samples for set A and set B are chosen to have the same indices.

    Trains on A, tests on A and B.
    
    Args
    ------
    A_feature: (X_train_A, Y_train_A), (X_test_A, Y_test_A), (genre_train_A, genre_test_A)
        Pre-trained embedding features and labels on first dataset (uncorrupted audio in most cases)
    B_features: list[(X_train_B, Y_train_B), (X_test_B, Y_test_B), (genre_train_B, genre_test_B)]
        Pre-trained embedding features and labels on second datasets (corrupted audio with one list item per effect parameter in most cases)
    same_negatives: bool, optional
        Whether to select the same negative samples for A and B (only makes sense when
        there is a one-to-one correspondence between the samples of A and B). Default False.

    Returns
    ------
    X_test_A_inst: dict[str, tensor]
        dict that stores for each instrument the features of the test samples of dataset A
    test_A_is_inst: dict[str, tensor]
        dict that stores for each instrument a tensor that states for each sample whether it is a positive or a negative sample (dataset A)
    pred_proba_A_inst: dict[str, tensor]
        dict that stores for each instrument the list of predicted probabilities for each sample to be positive (dataset A)
    results_B: list[X_test_B_inst, test_B_is_inst, pred_proba_B_inst]
        Same as three previous arguments, but for the second datasets
    """

    check_desensitizing_method(desensitizing_method)

    X_test_A_inst = {}
    test_A_is_inst = {}
    pred_proba_A_inst = {}
    X_test_B_insts = [{} for _ in B_features]
    test_B_is_insts = [{} for _ in B_features]
    pred_proba_B_insts = [{} for _ in B_features]

    for instrument in tqdm(instrument_map):

        (X_train_A, _), (X_test_A, Y_test_A), _ = A_feature
        test_is_inst_A = Y_test_A==instrument
        X_test_Bs = [x[1][0] for x in B_features]
        Y_test_Bs = [x[1][1] for x in B_features]
        test_is_inst_Bs = [Y_test_B==instrument for Y_test_B in Y_test_Bs]

        (X_train_inst_A, train_is_inst_A, X_train_A_true), train_B = \
            create_train_sets(instrument, A_feature, B_features)
        X_train_inst_Bs = [x[0] for x in train_B]
        train_is_inst_Bs = [x[1] for x in train_B]
        X_train_B_trues = [x[2] for x in train_B]

        if "-pcaproj_norm" in desensitizing_method:
            # Compute reference variances from A training data
            pca = PCA()
            pca.fit(X_train_A)
            ref_variances = pca.explained_variance_
        else:
            ref_variances = None

        X_inst_A = X_train_A_true
        X_inst_Bs = X_train_B_trues
        
        (X_train_inst_A, X_test_A, X_test_B), _ = desensitize(X_inst_A, X_inst_Bs, (X_train_inst_A, X_test_A, X_test_B), ref_variances=ref_variances)

        clf = compute_classifier(X_train_inst_A, train_is_inst_A)

        # Get prediction scores for the positive class
        pred_scores_test_A = clf.predict_proba(X_test_A)[:, 1]
        pred_scores_test_Bs = [clf.predict_proba(X_test_B)[:, 1] for X_test_B in X_test_Bs]

        X_test_A_inst[instrument] = X_test_A
        test_A_is_inst[instrument] = test_is_inst_A
        pred_proba_A_inst[instrument] = pred_scores_test_A

        for i in range(len(B_features)):
            X_test_B_insts[i][instrument] = X_test_Bs[i]
            test_B_is_insts[i][instrument] = test_is_inst_Bs[i]
            pred_proba_B_insts[i][instrument] = pred_scores_test_Bs[i]

    return (X_test_A_inst, test_A_is_inst, pred_proba_A_inst), (X_test_B_insts, test_B_is_insts, pred_proba_B_insts)
