#!/bin/bash

echo "====== Bitcrushing ======"
for emb in openl3 panns clap; do
    echo $emb
    for b in {4..15}; do
        echo -n "$b "
        python3 average_embeddings.py train_test_split.csv embeddings/embeddings_bitcrush_$b.h5 $emb embeddings/averaged/$emb/embeddings_bitcrush_$b.h5
    done
    echo
done

echo "====== Gain ======"
for emb in openl3 panns clap; do
    echo $emb
    for g in -40.0 -38.5 -37.0 -35.5 -34.0 -32.5 -31.0 -29.5 -28.0 -26.5 -25.0 -23.5 -22.0 -20.5 -19.0 -17.5 -16.0 -14.5 -13.0 -11.5 -10.0 -8.5 -7.0 -5.5 -4.0 -2.5 -1.0 0.5 2.0 3.5 5.0; do
        echo -n "$g "
        python3 average_embeddings.py train_test_split.csv embeddings/embeddings_gain_$g.h5 $emb embeddings/averaged/$emb/embeddings_gain_$g.h5
    done
    echo
done

echo "====== Low-pass filtering ======"
for emb in openl3 panns clap; do
    echo $emb
    for c in 1600.000 1735.498 1882.470 2041.889 2214.808 2402.372 2605.819 2826.496 3065.860 3325.496 3607.119 3912.592 4243.933 4603.335 4993.174 5416.026 5874.687 6372.191 6911.827 7497.162 8132.067 8820.740 9567.733 10377.986 11256.857 12210.155 13244.185 14365.783 15582.364 16901.972 18333.333; do
        echo -n "$c "
        python3 average_embeddings.py train_test_split.csv embeddings/embeddings_lowpass_cheby_$c.h5 $emb embeddings/averaged/$emb/embeddings_lowpass_cheby_$c.h5
    done
    echo
done

echo "====== Reverberation ======"
for emb in openl3 panns clap; do
    echo $emb
    for r in 0.01 0.04 0.07 0.10 0.13 0.16 0.19 0.22 0.25 0.28 0.31 0.34 0.37 0.40 0.43 0.46 0.49 0.52 0.55 0.58 0.61 0.64 0.67 0.70 0.73 0.76 0.79 0.82 0.85 0.88 0.91 0.94 0.97 1.00; do
        echo -n "$r "
        python3 average_embeddings.py train_test_split.csv embeddings/embeddings_reverb_$r.h5 $emb embeddings/averaged/$emb/embeddings_reverb_$r.h5
    done
    echo
done