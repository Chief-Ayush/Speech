`model_final_fp16.pt` - FastSpeech 2 (English, multi-speaker), step 50,000, fp16 weights (about 69 MB). It contains the weights, config, symbol list,
pitch/energy statistics and the speaker table, so `English/inference.py` needs nothing else (the HiFi-GAN vocoder is downloaded automatically).
