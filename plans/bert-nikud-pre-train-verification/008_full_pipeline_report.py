"""Generate a full visual pipeline report as markdown."""

import sys
import time
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "bert-nikud" / "src"))

import torch
from model import HebrewNikudModel, count_parameters
from dataset import (
    NikudDataset, load_dataset_from_file, collate_fn,
    prepare_training_data, VOWEL_TO_ID, ID_TO_VOWEL,
)
from encode import extract_nikud_labels
from decode import reconstruct_text_from_predictions
from tokenizer_utils import load_tokenizer
from normalize import normalize
from evaluate import calculate_cer, calculate_wer
from constants import LETTERS, CAN_HAVE_DAGESH, CAN_HAVE_SIN, CAN_NOT_HAVE_NIKUD

BASE = Path(__file__).resolve().parent.parent.parent / "bert-nikud"
TOKENIZER_PATH = BASE / "tokenizer" / "dictabert-large-char-menaked"
DATASET_PATH = BASE / "dataset" / "train.txt"
VAL_PATH = BASE / "dataset" / "val.txt"
OUT = Path(__file__).resolve().parent / "008_full_pipeline_report.md"

# Helpers
VOWEL_NAMES = {0: "none", 1: "patah ַ", 2: "tsere ֵ", 3: "hirik ִ", 4: "holam ֹ", 5: "qubut ֻ", 6: "shva ְ", 7: "vocal_shva ֽ"}

def char_desc(ch):
    """Short description of a character."""
    cp = f"U+{ord(ch):04X}"
    name = unicodedata.name(ch, "?")
    return f"`{ch}` {cp} {name}"

def label_str(label):
    """Format a label dict as a short string."""
    if label["vowel"] == -100:
        return "IGNORE"
    parts = [f"v={VOWEL_NAMES[label['vowel']]}"]
    if label["dagesh"]: parts.append("dagesh")
    if label["sin"]: parts.append("sin")
    if label["stress"]: parts.append("stress")
    if label["prefix"]: parts.append("prefix")
    return ", ".join(parts)


lines = []
def w(s=""):
    lines.append(s)

# ============================================================
w("# Full Pipeline Verification Report")
w()
w(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
w()

# --- 1. Environment ---
w("## 1. Environment")
w()
w(f"- Python: `{sys.version.split()[0]}`")
w(f"- PyTorch: `{torch.__version__}`")
w(f"- CUDA: `{torch.cuda.is_available()}` — {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
w()

# --- 2. Tokenizer ---
w("## 2. Tokenizer")
w()
tokenizer = load_tokenizer(str(TOKENIZER_PATH))
w(f"- Path: `{TOKENIZER_PATH}`")
w(f"- Vocab size: {tokenizer.vocab_size}")
w(f"- Special tokens: CLS={tokenizer.cls_token_id}, SEP={tokenizer.sep_token_id}, PAD={tokenizer.pad_token_id}, UNK={tokenizer.unk_token_id}")
w()

w("### Hebrew letter tokenization")
w()
w("| Letter | Token ID | Decoded | OK |")
w("|--------|----------|---------|-----|")
for letter in LETTERS:
    enc = tokenizer(letter, add_special_tokens=False)
    tid = enc["input_ids"][0]
    decoded = tokenizer.decode([tid])
    ok = "✓" if len(enc["input_ids"]) == 1 and tid != tokenizer.unk_token_id else "✗"
    w(f"| {letter} | {tid} | {decoded} | {ok} |")
w()

# --- 3. Dataset ---
w("## 3. Dataset")
w()
train_texts = load_dataset_from_file(str(DATASET_PATH))
val_texts = load_dataset_from_file(str(VAL_PATH))
w(f"- Train: `{DATASET_PATH}` — {len(train_texts)} lines")
w(f"- Val: `{VAL_PATH}` — {len(val_texts)} lines")
w()

w("### First 20 raw lines (train)")
w()
for i, t in enumerate(train_texts[:20]):
    w(f"{i+1}. `{t}`")
w()

# --- 4. Encoding: nikud text -> plain + labels ---
w("## 4. Encoding (nikud → plain text + labels)")
w()
w("Showing 5 sentences: how nikud text is decomposed into plain characters and per-character labels.")
w()

for idx in range(5):
    nikud_text = train_texts[idx]
    plain_text, labels_list = extract_nikud_labels(nikud_text)
    w(f"### Sentence {idx+1}")
    w()
    w(f"**Nikud input:** `{normalize(nikud_text)}`")
    w()
    w(f"**Plain text:** `{plain_text}`")
    w()
    w("| # | Char | Unicode | Hebrew? | Label |")
    w("|---|------|---------|---------|-------|")
    for ci, (ch, lab) in enumerate(zip(plain_text, labels_list)):
        is_heb = "✓" if ch in LETTERS else ""
        cp = f"U+{ord(ch):04X}"
        w(f"| {ci} | `{ch}` | {cp} | {is_heb} | {label_str(lab)} |")
    w()

# --- 5. Tokenization + alignment ---
w("## 5. Tokenization & Token-Label Alignment")
w()
w("Showing how tokenizer maps characters to token IDs, offset mapping, and how labels align.")
w()

for idx in range(5):
    nikud_text = train_texts[idx]
    data = prepare_training_data(nikud_text, tokenizer)
    input_ids = data["input_ids"]
    offsets = data["offset_mapping"]
    plain = data["plain_text"]

    w(f"### Sentence {idx+1}")
    w()
    w(f"**Plain:** `{plain}`")
    w()
    w(f"**Tokens:** {len(input_ids)} (incl. CLS/SEP)")
    w()
    w("| Token# | ID | Decoded | Offset | Vowel | Dagesh | Sin | Stress | Prefix |")
    w("|--------|-----|---------|--------|-------|--------|-----|--------|--------|")
    for ti in range(len(input_ids)):
        tid = input_ids[ti].item()
        decoded = tokenizer.decode([tid]).replace("|", "\\|")
        start, end = offsets[ti].tolist()
        vl = data["vowel_labels"][ti].item()
        dl = data["dagesh_labels"][ti].item()
        sl = data["sin_labels"][ti].item()
        stl = data["stress_labels"][ti].item()
        pl = data["prefix_labels"][ti].item()
        vowel_s = VOWEL_NAMES.get(vl, str(vl)) if vl != -100 else "—"
        dagesh_s = str(dl) if dl != -100 else "—"
        sin_s = str(sl) if sl != -100 else "—"
        stress_s = str(stl) if stl != -100 else "—"
        prefix_s = str(pl) if pl != -100 else "—"
        w(f"| {ti} | {tid} | `{decoded}` | [{start},{end}) | {vowel_s} | {dagesh_s} | {sin_s} | {stress_s} | {prefix_s} |")
    w()

# --- 6. Encode→Decode roundtrip ---
w("## 6. Encode → Decode Roundtrip (20 sentences)")
w()
w("Using ground-truth labels as predictions to verify perfect reconstruction.")
w()

roundtrip_ok = 0
roundtrip_total = 20
w("| # | Match | Target (first 80 chars) |")
w("|---|-------|------------------------|")
for idx in range(roundtrip_total):
    nikud_text = train_texts[idx]
    data = prepare_training_data(nikud_text, tokenizer)
    vowel_preds = data["vowel_labels"].clone(); vowel_preds[vowel_preds == -100] = 0
    dagesh_preds = data["dagesh_labels"].clone(); dagesh_preds[dagesh_preds == -100] = 0
    sin_preds = data["sin_labels"].clone(); sin_preds[sin_preds == -100] = 0
    stress_preds = data["stress_labels"].clone(); stress_preds[stress_preds == -100] = 0
    prefix_preds = data["prefix_labels"].clone(); prefix_preds[prefix_preds == -100] = 0
    reconstructed = reconstruct_text_from_predictions(
        data["input_ids"], data["offset_mapping"],
        vowel_preds, dagesh_preds, sin_preds, stress_preds, prefix_preds, tokenizer,
    )
    expected = normalize(nikud_text)
    ok = reconstructed == expected
    if ok: roundtrip_ok += 1
    w(f"| {idx+1} | {'✓' if ok else '✗'} | `{expected[:80]}` |")
w()
w(f"**Result: {roundtrip_ok}/{roundtrip_total} exact matches**")
w()

# --- 7. Model ---
w("## 7. Model Architecture")
w()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = HebrewNikudModel()
model.to(device)
total_p, train_p = count_parameters(model)
w(f"- Base: `dicta-il/dictabert-large-char`")
w(f"- Parameters: {total_p:,} total, {train_p:,} trainable")
w(f"- Hidden size: {model.hidden_size}")
w()
w("### Classification heads")
w()
w("| Head | Type | Output dim | Loss |")
w("|------|------|-----------|------|")
w("| Vowel | Multi-class | 8 | CrossEntropy (label_smoothing=0.1) |")
w("| Dagesh | Binary | 1 | BCEWithLogits |")
w("| Sin | Binary | 1 | BCEWithLogits |")
w("| Stress | Binary | 1 | BCEWithLogits |")
w("| Prefix | Binary | 1 | BCEWithLogits |")
w()

# --- 8. Forward pass ---
w("## 8. Forward Pass Verification")
w()
texts_small = train_texts[:4]
ds = NikudDataset(texts_small, tokenizer, use_cache=False)
from torch.utils.data import DataLoader
loader = DataLoader(ds, batch_size=4, collate_fn=collate_fn)
batch = next(iter(loader))
input_ids = batch["input_ids"].to(device)
attention_mask = batch["attention_mask"].to(device)

model.train()
outputs = model(
    input_ids=input_ids, attention_mask=attention_mask,
    vowel_labels=batch["vowel_labels"].to(device),
    dagesh_labels=batch["dagesh_labels"].to(device),
    sin_labels=batch["sin_labels"].to(device),
    stress_labels=batch["stress_labels"].to(device),
    prefix_labels=batch["prefix_labels"].to(device),
)
w(f"- Batch shape: {input_ids.shape}")
w(f"- Vowel logits: {outputs['vowel_logits'].shape}")
w(f"- Total loss: {outputs['loss'].item():.4f}")
w(f"  - Vowel: {outputs['vowel_loss'].item():.4f}")
w(f"  - Dagesh: {outputs['dagesh_loss'].item():.4f}")
w(f"  - Sin: {outputs['sin_loss'].item():.4f}")
w(f"  - Stress: {outputs['stress_loss'].item():.4f}")
w(f"  - Prefix: {outputs['prefix_loss'].item():.4f}")
w()

# --- 9. 1-min training ---
w("## 9. Training (1 minute, 200 samples)")
w()
train_sub = train_texts[:200]
train_ds = NikudDataset(train_sub, tokenizer, use_cache=False)
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_fn)

model2 = HebrewNikudModel()
model2.to(device)
optimizer = torch.optim.AdamW(model2.parameters(), lr=1e-4)

start = time.time()
step = 0
loss_log = []
model2.train()

while time.time() - start < 60:
    for b in train_loader:
        if time.time() - start >= 60:
            break
        optimizer.zero_grad()
        out = model2(
            input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device),
            vowel_labels=b["vowel_labels"].to(device), dagesh_labels=b["dagesh_labels"].to(device),
            sin_labels=b["sin_labels"].to(device), stress_labels=b["stress_labels"].to(device),
            prefix_labels=b["prefix_labels"].to(device),
        )
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model2.parameters(), 1.0)
        optimizer.step()
        step += 1
        loss_log.append(out["loss"].item())

w(f"- Steps: {step}")
w(f"- First loss: {loss_log[0]:.4f}")
w(f"- Last loss: {loss_log[-1]:.4f}")
w(f"- Loss decreased: {'✓' if loss_log[-1] < loss_log[0] else '✗'}")
w()

# --- 10. Inference samples ---
w("## 10. Inference After Training (20 eval samples)")
w()
eval_texts_sub = val_texts[:20]
eval_ds = NikudDataset(eval_texts_sub, tokenizer, use_cache=False)
eval_loader = DataLoader(eval_ds, batch_size=20, collate_fn=collate_fn)
eval_batch = next(iter(eval_loader))

model2.eval()
with torch.no_grad():
    preds = model2.predict(
        eval_batch["input_ids"].to(device),
        eval_batch["attention_mask"].to(device),
    )

w("| # | CER | Target | Predicted |")
w("|---|-----|--------|-----------|")
total_cer = 0.0
for i in range(len(eval_texts_sub)):
    predicted = reconstruct_text_from_predictions(
        eval_batch["input_ids"][i].to(device), eval_batch["offset_mapping"][i],
        preds["vowel"][i], preds["dagesh"][i],
        preds["sin"][i], preds["stress"][i], preds["prefix"][i],
        tokenizer,
    )
    target = normalize(eval_batch["original_text"][i])
    cer = calculate_cer(predicted, target)
    total_cer += cer
    # Escape pipes for markdown
    t_esc = target.replace("|", "\\|")
    p_esc = predicted.replace("|", "\\|")
    w(f"| {i+1} | {cer:.4f} | {t_esc} | {p_esc} |")
avg_cer = total_cer / len(eval_texts_sub)
w()
w(f"**Average CER: {avg_cer:.4f}**")
w()

# --- 11. Constants summary ---
w("## 11. Constants & Rules")
w()
w(f"- Hebrew letters: `{LETTERS}`")
w(f"- Can have dagesh: `{CAN_HAVE_DAGESH}`")
w(f"- Can have sin: `{CAN_HAVE_SIN}`")
w(f"- Cannot have nikud (finals): `{CAN_NOT_HAVE_NIKUD}`")
w()
w("### Vowel mapping")
w()
w("| ID | Name | Unicode |")
w("|----|------|---------|")
for vid, vchar in sorted(ID_TO_VOWEL.items()):
    if vchar is None:
        w(f"| {vid} | none | — |")
    else:
        w(f"| {vid} | {VOWEL_NAMES[vid]} | U+{ord(vchar):04X} |")
w()

# --- Write ---
OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"Report written to {OUT}")
print(f"Sections: Environment, Tokenizer, Dataset, Encoding, Alignment, Roundtrip, Model, Forward, Training, Inference, Constants")
