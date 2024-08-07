import evaluate
from pythainlp import word_tokenize
from functools import partial

""" Generation metrics """
bleu = evaluate.load('bleu')
rouge = evaluate.load('rouge')
sacrebleu = evaluate.load('sacrebleu')
chrf = evaluate.load('chrf')
meteor = evaluate.load('meteor')
tokenizer = partial(word_tokenize, engine='newmm')

def generation_metrics_fn(list_hyp, list_label):
    # hyp and label are both list of string
    # Check if the lists are empty
    if not list_hyp or not list_label or len(list_hyp) != len(list_label):
        raise ValueError("Input lists must be non-empty and of the same length")
    
    list_hyp = [hyp if hyp is not None else "" for hyp in list_hyp]
    list_label = [label if label is not None else "" for label in list_label]
    list_label_sacrebleu = list(map(lambda x: [x], list_label))

    metrics = {}

    # Compute BLEU score
    try:
        metrics["BLEU"] = bleu.compute(predictions=list_hyp, references=list_label_sacrebleu, tokenizer=tokenizer)['bleu'] * 100
    except ZeroDivisionError:
        metrics["BLEU"] = 0.0

    # Compute SacreBLEU score
    try:
        metrics["SacreBLEU"] = sacrebleu.compute(predictions=list_hyp, references=list_label_sacrebleu, tokenize='flores200')['score']
    except ZeroDivisionError:
        metrics["SacreBLEU"] = 0.0

    # Compute chrF++ score
    try:
        metrics["chrF++"] = chrf.compute(predictions=list_hyp, references=list_label_sacrebleu)['score']
    except ZeroDivisionError:
        metrics["chrF++"] = 0.0

    # Compute METEOR score
    try:
        metrics["meteor"] = meteor.compute(predictions=list_hyp, references=list_label)['meteor'] * 100
    except ZeroDivisionError:
        metrics["meteor"] = 0.0

    # Compute ROUGE scores
    try:
        rouge_score = rouge.compute(predictions=list_hyp, references=list_label, tokenizer=tokenizer)
        metrics["ROUGE1"] = rouge_score['rouge1'] * 100
        metrics["ROUGE2"] = rouge_score['rouge2'] * 100
        metrics["ROUGEL"] = rouge_score['rougeL'] * 100
        metrics["ROUGELsum"] = rouge_score['rougeLsum'] * 100
    except ZeroDivisionError:
        metrics["ROUGE1"] = 0.0
        metrics["ROUGE2"] = 0.0
        metrics["ROUGEL"] = 0.0
        metrics["ROUGELsum"] = 0.0

    return metrics