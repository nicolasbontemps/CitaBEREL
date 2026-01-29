#!/usr/bin/env python3
# scripts/Cita_Berel_inference.py

# Auteur : Nicolas Bontemps
# Date : 2025-11-20
# Description : script for token-level citation identification in rabbinic litterature usinf CitaBEREL Bert model
# in : csv file with Id, token columns (tag optionnals for evaluation)
# out : csv file with predicted tags

import argparse
from pathlib import Path

import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification
from sklearn.metrics import classification_report
from tqdm import tqdm


# ---------- Hebrew cleaning ----------
def clean_hebrew(token: str) -> str:
    if not isinstance(token, str):
        token = "" if pd.isna(token) else str(token)
    token = token.replace("\uFB4F", "אל")  # aleph-lamed ligature
    return "".join([ch for ch in token if ("\u0590" <= ch <= "\u05FF") or (ch == "'")])


def remove_non_hebrew(text: str) -> str:
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)
    return "".join([ch for ch in text if ("\u0590" <= ch <= "\u05FF") or (ch == "'")])


# ---------- Token merge (BERT wordpieces) ----------
def merge_split_tokens(results):
    merged = []
    for token, pred in results:
        if token.startswith("##") and merged:
            prev_token, prev_pred = merged.pop()
            merged.append((prev_token + token[2:], prev_pred))
        else:
            merged.append((token, pred))
    return merged


def test_model(sentence, model, tokenizer, device):
    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        padding=True,
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())
    labels = [model.config.id2label[p.item()] for p in predictions[0]]
    return list(zip(tokens, labels))


def run_inference(
    csv_input: Path,
    csv_output: Path,
    model_name_or_path: str,
    text_col: str = "merge_norm2_and_abbrev",
    tag_col: str = "tag",
    id_col: str = "id",
    segment_size: int = 50,
    do_report: bool = True,
    min_seq_warn: bool = True,
):
    df = pd.read_csv(csv_input)

    # Ensure text column exists
    if text_col not in df.columns:
        raise ValueError(f"Colonne texte '{text_col}' absente du CSV.")

    # Clean + keep Hebrew only
    df[text_col] = df[text_col].fillna("").astype(str)
    df[text_col] = df[text_col].apply(clean_hebrew).apply(remove_non_hebrew)

    # Prepare labels for evaluation (optional)
    has_tag = tag_col in df.columns
    if has_tag:
        df[tag_col] = pd.to_numeric(df[tag_col], errors="coerce")
        df["__new_tag"] = np.where(df[tag_col] > 0, "LABEL_2", "LABEL_0")

    # Ensure ID column
    if id_col not in df.columns:
        df[id_col] = df.index

    # Mask empty/placeholder lines
    mask_exclu = df[text_col].str.strip().isin(["", "˙"])
    df_use = df[~mask_exclu].copy().reset_index(drop=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForTokenClassification.from_pretrained(model_name_or_path)
    model.to(device).eval()

    # Check model max length
    if min_seq_warn and hasattr(tokenizer, "model_max_length"):
        _ = tokenizer.model_max_length

    # Build segments of words (not subwords)
    lines = df_use[text_col].tolist()
    ids = df_use[id_col].tolist()
    indexed_words = [(i, w, ids[i]) for i, w in enumerate(lines)]
    segments = [
        indexed_words[i : i + segment_size]
        for i in range(0, len(indexed_words), segment_size)
    ]

    predictions = {}
    for segment in tqdm(segments, unit="seg"):
        _, words, orig_ids = zip(*segment)
        sentence = " ".join(words)
        raw = test_model(sentence, model, tokenizer, device)
        merged = merge_split_tokens(raw)

        j = 0
        for token, label in merged:
            if token in ["[CLS]", "[SEP]", "[UNK]"]:
                continue
            if j < len(orig_ids):
                predictions[orig_ids[j]] = (token, label)
                j += 1
        # Note: si la phrase est TRONQUÉE, on peut perdre l’alignement en fin de segment.

    # Reconstruct final rows
    final_rows = []
    for _, row in df.iterrows():
        rid = row[id_col]
        original = row[text_col]
        token, pred = ("", "IGNORÉ")
        if rid in predictions:
            token, pred = predictions[rid]
        final_rows.append([rid, original, token, pred, row.get("__new_tag", "")])

    cols = ["id", "Original", "Token", "Prediction", "new_tag"]
    final_df = pd.DataFrame(final_rows, columns=cols)
    final_df = final_df[final_df["Prediction"] != "IGNORÉ"]

    # Save
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(csv_output, index=False)

    # Optional classification report
    if do_report and has_tag and not final_df.empty:
        y_true = final_df["new_tag"].tolist()
        y_pred = final_df["Prediction"].tolist()
        print("\nClassification report\n")
        print(classification_report(y_true, y_pred))


def parse_args():
    p = argparse.ArgumentParser(description="Run token classification inference on a CSV.")
    p.add_argument("--input", required=True, type=Path, help="Chemin du CSV d'entrée")
    p.add_argument("--output", required=True, type=Path, help="Chemin du CSV de sortie")
    p.add_argument("--model", required=True, help="Nom ou chemin du modèle (HF hub ou local)")
    p.add_argument("--text-col", default="merge_norm2_and_abbrev", help="Colonne texte")
    p.add_argument("--tag-col", default="tag", help="Colonne des tags (0/1, optionnelle)")
    p.add_argument("--id-col", default="id", help="Colonne ID (auto-créée si absente)")
    p.add_argument("--segment-size", type=int, default=50, help="Taille du segment en mots")
    p.add_argument("--no-report", action="store_true", help="Désactiver le classification report")
    return p.parse_args()


def main():
    args = parse_args()
    run_inference(
        csv_input=args.input,
        csv_output=args.output,
        model_name_or_path=args.model,
        text_col=args.text_col,
        tag_col=args.tag_col,
        id_col=args.id_col,
        segment_size=args.segment_size,
        do_report=not args.no_report,
    )


if __name__ == "__main__":
    main()
