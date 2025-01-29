import deem
import h5py
import sys
import pandas as pd
import os

def main():
    if len(sys.argv) < 5:
        print(f"Usage: python3 {sys.argv[0]} <path/to/train_test_split.csv> <path/to/input.h5> <embedding> <path/to/output.h5>")
        sys.exit(1)
    
    meta_path, input_path, embedding, output_path = sys.argv[1:5]

    meta_all = pd.read_csv(meta_path)
    (X_train, Y_train), (X_test, Y_test), (genre_train, genre_test) = deem.load_feature(input_path, embedding, meta_all)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    outfile = h5py.File(output_path, "w")
    outfile["X_train"] = X_train
    outfile["Y_train"] = Y_train
    outfile["X_test"] = X_test
    outfile["Y_test"] = Y_test
    outfile["genre_train"] = genre_train
    outfile["genre_test"] = genre_test
    outfile.close()

if __name__ == "__main__":
    main()