import evaluate

WER_METRIC = evaluate.load("wer")
CER_METRIC = evaluate.load("cer")


def compute_metrics(pred, tokenizer):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # convert -100 (ignored) tokens into pad tokens for decoding
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer_score = 100 * WER_METRIC.compute(predictions=pred_str, references=label_str)
    cer_score = 100 * CER_METRIC.compute(predictions=pred_str, references=label_str)

    return {"wer": wer_score, "cer": cer_score}
