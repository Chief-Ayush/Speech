"""Step 4: synthesise the assignment's test sentences and draw the variance-adaptor plots (Telugu).
python Telugu/inference.py   (defaults: Telugu/checkpoints/model_final_fp16.pt -> Results/Telugu)"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.dirname(HERE), HERE]
import torch
from fs2 import load_cfg, synth
from frontend import text_to_phones

# Test sentences exactly as in the assignment sheet (see report for the notes on E3 and the Telugu typography).
TESTS = {
    "T1": "న్యూరోఫార్మకోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు.",
    "T2a": "తాడుపై అతని పట్టు చాలా బలంగా ఉంది.",
    "T2b": "ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చింది.",
    "T3": "పులి అడ విలో నిశ్శబ్దంగా నడిచింది.",
    "T4": "ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి.",
}


if __name__ == "__main__":
    cfg, a = load_cfg(HERE)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = a.ckpt or os.path.join(HERE, "checkpoints", "model_final_fp16.pt")
    out = a.out or os.path.join(os.path.dirname(HERE), "Results", "Telugu")
    model, ck = synth.load_model(ckpt, dev)
    voc = synth.load_vocoder(dev)
    spk = a.speaker
    if spk is None:
        counts = ck.get("speaker_counts") or {}
        spk = max(counts, key=counts.get) if counts else 0      # most frequent training speaker
    print("checkpoint step", ck["step"], "| speaker id", spk)
    os.makedirs(os.path.join(out, "test_generated_audios"), exist_ok=True)
    log = {}
    for tid, text in TESTS.items():
        ph = text_to_phones(text)
        r = synth.synthesize(model, ck, voc, ph, speaker=spk, dev=dev)
        synth.save_wav(os.path.join(out, "test_generated_audios", tid + ".wav"), r["wav"])
        synth.variance_plots(r, ph, tid + " (Telugu; symbols are ISO 15919 letters)", os.path.join(out, "variance_adaptor_plots"), tid)
        log[tid] = dict(text=text, phones=ph, duration_frames=r["dur"].tolist(), pitch_hz=[round(float(x), 1) for x in r["pitch"]],
                        energy=[round(float(x), 3) for x in r["energy"]], seconds=round(len(r["wav"]) / 22050, 2))
        print(tid, "%.2f s" % log[tid]["seconds"], " ".join(ph))
    json.dump(log, open(os.path.join(out, "test_synthesis_log.json"), "w"), ensure_ascii=False, indent=1)
