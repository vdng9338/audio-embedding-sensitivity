import h5py
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import re
import random


# Regex matchers for instrument name, genre and group

instrument_matcher = re.compile(r'.*?\[(?P<class>\w+)\].*')

def instrument_name(x):
    """Returns instrument class of x."""
    res = re.match(instrument_matcher, x)
    return res.groups()[0]

genre_matcher = re.compile(r'.*?\]\[(?P<class>\w+)\]\d.*')

def genreof(x):
    res = re.match(genre_matcher, x)
    return res.groups()[0]

group_matcher = re.compile('.*(?P<id>\d\d\d\d)__.*')

def groupname(x):
    """Returns the group name of x (filename of the audio file that x comes from)."""
    res = re.match(group_matcher, x)
    return res.groups()[0]


# Instrument and genre alignment dictionaries
instrument_align = {'cla': 'clarinet', 
                   'org': 'organ',
                   'cel': 'cello',
                   'vio': 'violin',
                   'gac': 'guitar_acc',
                   'voi': 'voice',
                   'gel': 'guitar_ele',
                   'sax': 'saxophone',
                   'tru': 'trumpet',
                   'pia': 'piano',
                   'flu': 'flute'}

genre_align = {'pop_roc': 'pop_roc',
            'jaz_blu': 'jazz_blue',
            'cla': 'classical',
            'cou_fol': 'country_folk',
            'lat_sou': 'latin_soul'}



def main():
    # Retrieve and deduplicate filenames
    print("Retrieving and deduplicating filenames...")
    embeddings = h5py.File("embeddings/embeddings.h5", "r")
    filenames = list(embeddings['irmas/openl3/keys'][()])
    filenames = [str(f, "utf-8") for f in filenames]
    filenames = list(set(filenames))

    # Drop latin_soul from dataset
    print("Dropping latin_soul from dataset...")
    filenames = [f for f in filenames if genreof(f) != 'lat_sou']

    # Extract instrument class and group names from filenames
    print("Extracting instruments and groups from filenames...")
    instruments = list(map(instrument_name, filenames))
    instruments = [instrument_align[i] for i in instruments]
    groups = list(map(groupname, filenames))

    # Use StratifiedGroupKFold to split dataset into train and test
    print("Creating train/test split...")
    labelencoder = LabelEncoder()
    labelencoder.fit(instruments)
    y = labelencoder.transform(instruments)
    splits = 4
    splitter = StratifiedGroupKFold(splits, shuffle=True, random_state=20220419)
    train_indices, test_indices = next(splitter.split(filenames, y=y, groups=groups))
    all_files = pd.Series(filenames)
    train_files = all_files[train_indices]
    test_files = all_files[test_indices]

    # Balance instrument classes in train set
    print("Balancing instrument classes in train set...")
    train_instruments = list(map(instrument_name, train_files))
    train_instruments = [instrument_align[i] for i in train_instruments]

    train_filename_per_inst = dict([(inst, []) for inst in instrument_align.values()])
    for i, (inst, filename) in enumerate(zip(train_instruments, train_files)):
        train_filename_per_inst[inst].append(filename)
    
    min_number = min([len(l) for l in train_filename_per_inst.values()])
    train_files = []
    random.seed(42)
    for filenames_inst in train_filename_per_inst.values():
        random.shuffle(filenames_inst)
        train_files.extend(filenames_inst[:min_number])


    # Build CSV
    print("Building CSV...")
    instrument_col, genre_col, file_name_col, split_col = [], [], [], []
    for item in [train_files, test_files]:
        instruments = [instrument_align[instrument_name(x)] for x in item]
        genres = [genre_align[genreof(x)] for x in item]
        instrument_col.extend(instruments)
        genre_col.extend(genres)
        file_name_col.extend(item)
    split_col = ['train'] * len(train_files) + ['test'] * len(test_files)

    meta_all = pd.DataFrame(columns=['instrument', 'genre', 'file_name', 'split'])
    meta_all['instrument'] = instrument_col
    meta_all['genre'] = genre_col
    meta_all['file_name'] = file_name_col
    meta_all['split'] = split_col

    # Write CSV
    print("Writing CSV...")
    meta_all.to_csv('train_test_split.csv', index=None)


if __name__ == '__main__':
    main()
