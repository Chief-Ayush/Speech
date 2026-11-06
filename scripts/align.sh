#!/bin/bash
# Step 1c: forced alignment with the Montreal Forced Aligner 3.4.3 (RECONSTRUCTION of the commands that were run on Kaggle).
# Install (no root needed):  Miniforge -> conda create -n mfa -c conda-forge montreal-forced-aligner=3.4.3
# Run from the directory that holds raw_data/ , english_combined.dict and telugu.dict.
set -e
export PATH=/tmp/mf/envs/mfa/bin:$PATH
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa
mfa model download g2p english_us_arpa

# English: pretrained acoustic model + extended dictionary (about 23 minutes on 4 CPUs)
mfa align --clean -j 4 raw_data/English english_combined.dict english_us_arpa \
    preprocessed_data/English/TextGrid --beam 100 --retry_beam 400

# Telugu: no pretrained model -> train a single-speaker model from scratch on the 2,830 selected clips (about 2 h on 4 CPUs)
mfa train --clean --single_speaker -j 4 raw_data/Telugu telugu.dict telugu_acoustic.zip \
    --output_directory preprocessed_data/Telugu/TextGrid
# The first pass wrote TextGrids for 2,530 of 2,830 clips. Re-align everything with the trained model and a wider beam,
# then copy the 300 missing TextGrids into the first-pass folder (all 2,830 were then available).
mfa align --clean --single_speaker -j 3 raw_data/Telugu telugu.dict telugu_acoustic.zip \
    preprocessed_data/Telugu/TextGrid_realign --beam 200 --retry_beam 800
