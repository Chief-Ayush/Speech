# FastSpeech 2 text-to-speech for English and Telugu

Two separate FastSpeech 2 models, trained from scratch for about 4.5 hours of speech each: **English** (Svarah, multi-speaker, Indian-accented) and **Telugu**
(IndicTTS Telugu, one male voice). The code is a modernised port of the reference implementation
[ming024/FastSpeech2](https://github.com/ming024/FastSpeech2) (MIT licence), following [the FastSpeech 2 paper](https://arxiv.org/abs/2006.04558).
The full write-up (data, modifications, loss analysis, per-sentence analysis, limitations) is in `Report/Technical_Report.pdf`.

## Results at a glance

Intelligibility, measured with off-the-shelf speech recognisers on 80 held-out validation sentences per language (lower is better; "real" = the same recogniser on the real recordings):

| | CER (real) | CER (model) | WER (real) | WER (model) |
|---|---|---|---|---|
| English (Whisper-small) | 3.15 % | 5.68 % | 6.25 % | 11.67 % |
| Telugu (Whisper Telugu large-v2) | 2.72 % | 4.21 % | 5.15 % | 6.92 % |

All five test sentences per language were synthesised (`Results/*/test_generated_audios`). A native Telugu speaker listened to the five Telugu sentences: every word was
pronounced as written; the voice sounds robotic (a known property of this model family at this data size).
These numbers are objective intelligibility checks, not a formal listening test (no MOS).

## Repository layout

```
Speech/
├── English/                 everything specific to English
│   ├── config.yaml            all hyper-parameters and paths
│   ├── preprocess.py          step 2: aligner output -> mel / pitch / energy / duration features
│   ├── train.py               step 3: training (resumes automatically)
│   ├── inference.py           step 4: the 5 test sentences -> audio + variance-adaptor plots
│   ├── frontend.py            text -> phoneme symbols (ARPAbet)
│   └── checkpoints/           model_final_fp16.pt (step 50,000)
├── Telugu/                  the same files for Telugu (frontend.py + telugu_text.py: Telugu script -> ISO 15919 symbols)
├── fs2/                     language-independent FastSpeech 2 library, shared by both languages
│   ├── model.py               encoder, variance adaptor, decoder, post-net, loss
│   ├── data.py  features.py   dataset and batching / mel, F0, energy and duration extraction
│   ├── train.py  synth.py     training loop / inference and plotting
│   └── vocoder/               HiFi-GAN code (weights are downloaded on first use)
├── scripts/                 everything outside the model: data selection, aligner scripts, evaluation, plotting
├── Results/{English,Telugu}/  training_loss.png, validation_loss.png, test_generated_audios/, variance_adaptor_plots/{duration,pitch,energy}/, logs/, ASR results
├── Report/Technical_Report.pdf
├── requirements.txt
└── README.md
```

`English/` and `Telugu/` contain exactly the files the assignment asks for and are the only entry points. They are thin: each one loads its `config.yaml` and its `frontend.py` and calls the shared code in `fs2/`.
Results are in `Results/`; `Results/*/extra_validation_best_checkpoint/` holds the same outputs from the best-validation checkpoint, for comparison only.

## Quick start: generate the test sentences (CPU is enough)

```bash
pip install -r requirements.txt
python English/inference.py        # -> Results/English/test_generated_audios/*.wav and variance_adaptor_plots/
python Telugu/inference.py         # -> Results/Telugu/...
```

The checkpoints are in the repository, and the vocoder weights are downloaded automatically from `ming024/FastSpeech2` on first use (if that is blocked, unzip `generator_universal.pth.tar.zip` from that
repository into `fs2/vocoder/`). English also needs the NLTK data `averaged_perceptron_tagger_eng` and `cmudict`.

## Full pipeline (data -> trained model)

Run every command from the repository root. Data folders (`raw_data/`, `prep/`, `preprocessed_data/`, `output/`) are created next to the code and are not committed.
Environment used: Kaggle, 2 x Tesla T4, Python 3, PyTorch 2.x. The forced aligner (Montreal Forced Aligner 3.4.3) is installed with conda, not pip.

```bash
# 1. Select and split the data, export 22.05 kHz mono wav + transcripts (needs a Hugging Face token)
python scripts/prepare_data.py            # -> raw_data/<Lang>/<speaker>/<id>.wav|.lab, prep/<lang>_manifest.csv

# 2. Pronunciation dictionaries and forced alignment
python scripts/build_dictionaries.py
bash   scripts/align.sh                   # -> preprocessed_data/<Lang>/TextGrid

# 3. Features, training, test sentences (one GPU per language was used)
python English/preprocess.py  &&  python Telugu/preprocess.py
CUDA_VISIBLE_DEVICES=0 python English/train.py        # 50,000 steps, resumes automatically from output/ckpt/English/last.pt
CUDA_VISIBLE_DEVICES=1 python Telugu/train.py
python English/inference.py  &&  python Telugu/inference.py

# 4. Figures and the optional ASR intelligibility check
python scripts/plot_losses.py --log output/log/English --out Results/English --title English
python scripts/plot_losses.py --log output/log/Telugu  --out Results/Telugu  --title Telugu
python scripts/evaluate_asr.py --lang English --asr openai/whisper-small --asr_lang en --ckpts English/checkpoints/model_final_fp16.pt --out english_asr.json
python scripts/evaluate_asr.py --lang Telugu  --asr vasista22/whisper-telugu-large-v2 --asr_lang te --ckpts Telugu/checkpoints/model_final_fp16.pt --out telugu_asr.json
```

Each checkpoint file is self-contained (weights, config, symbols, statistics), about 69 MB (fp16), so no Git LFS is needed.

## What differs from the reference implementation

* Ported to current libraries (PyTorch 2.x `scaled_dot_product_attention`, librosa 0.10, no custom STFT, vectorised length regulator); fp16 autocast; length-bucketed batches;
  resumable training with best-validation tracking; fp16 inference checkpoints.
* Feature extraction reads MFA 3.x TextGrids directly (silence is an empty label there, the reference code expects `sil`/`sp`/`spn`); pitch/energy statistics come from the **training** utterances only.
* **English:** multi-speaker model (speaker-embedding table, 114 pseudo-speakers), ARPAbet symbols with stress, text front-end = number/time expansion + CMUdict/g2p_en run phrase by phrase
  (so that context decides e.g. noun vs verb "conduct"), spelled-out AM/PM.
* **Telugu:** no pretrained aligner or lexicon exists, so the pipeline is graphemic: Telugu script -> number expansion -> ISO 15919 romanisation (Aksharamukha) -> 50 symbols
  (49 letters with aspirates as single symbols, plus `sp` for punctuation); the forced aligner was trained from scratch on the 2,830 selected clips.
* Training budget: 50,000 steps at batch size 32 (the reference recipe uses 900,000 steps at batch size 16 on much larger corpora).

## Data selection and split (details in the report)

* **English (ai4bharat/Svarah):** clips of 1.5-15 s without digits or unusual characters, at least 3 words, 5-22 characters per second, not clipped or near-silent; the
  lowest-SNR clips were dropped to reach 6.1 h; then split 5.07 h / 1.04 h (4.71 h / 0.96 h after trimming leading/trailing silence). Svarah has no speaker id, so a
  pseudo-speaker is the unique combination of its 8 speaker-metadata fields (115 combinations for 117 real speakers; a few real speakers may be merged).
* **Telugu (SPRINGLab/IndicTTS_Telugu):** the male voice only, 1.5-15 s, no digits, Telugu script only, 8-16.5 characters per second, random order until 6.1 h; split
  5.10 h / 1.00 h (4.35 h / 0.85 h after trimming).
* **Split:** utterance-level and stratified by speaker, with all clips of the same sentence on the same side: zero text overlap between training and validation (verified).

## Honest notes and known limitations

* Validation loss flattens after about 9,000 (English) / 13,000 (Telugu) steps while the training loss keeps falling (small data, pitch/energy contours are partly unpredictable);
  the final checkpoint is nevertheless more intelligible than the best-validation one in both languages, so the final checkpoint is the submitted one (results of the best-validation checkpoints are in `Results/*/extra_validation_best_checkpoint/`).
* English test sentence E3 contains a typo ("rhyth m"); it was synthesised exactly as given, and a corrected version is included as `E3_corrected_extra`.
  The Telugu sentences were taken from the assignment PDF, whose text extraction contained stray spaces inside words; the intended orthography was used (word spaces have no effect on the model input).
* The Telugu number reading ("2026", "12వ", "10:45") follows rules we chose; the recogniser confirms the audio matches the words we chose, a single native speaker confirmed the pronunciation of every word.
* The ASR-based scores say nothing about naturalness; both voices sound synthetic, the Telugu one clearly robotic.
* `scripts/build_dictionaries.py` and `scripts/align.sh` were rewritten from the Kaggle notebook cells that were actually run (same commands and logic). The training itself ran in Kaggle notebooks that call the `English/` and `Telugu/` scripts below.

## Credits

FastSpeech 2: Ren et al., 2021. Code base and vocoder: ming024/FastSpeech2 (MIT), HiFi-GAN (MIT). Datasets: Svarah (AI4Bharat), IndicTTS Telugu (IIT Madras; as hosted by SPRING Lab). Aligner: Montreal Forced Aligner.
