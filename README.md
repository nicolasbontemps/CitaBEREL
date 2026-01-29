# CitaBEREL - Hebrew Biblical Citation Detection – Inference Script
CitaBEREL is a finetuned BERT model for detecting Biblical quotations in Rabbinic Litterature. This repository provides a command-line script to run **token-level inference**

The model is applied to CSV files where each row corresponds to a word (or token) in the original text.

👉 **Model weights** are hosted on Hugging Face:  
https://huggingface.co/nbontemps/CitaBEREL

---

## Installation

Create a virtual environment (recommended), then install dependencies:

```bash
pip install -r requirements.txt

## Input CSV format

The input CSV must contain a text column (default: merge_norm2_and_abbrev).

Optional columns:

id – unique row identifier (auto-created if missing)

tag – gold labels (0/1) for evaluation

| id | merge_norm2_and_abbrev | tag |
| -: | ---------------------- | --: |
|  0 | ויאמר                 |   1 |
|  1 | ייי                    |   1 |
|  2 | אל                  |   1 |
|  3 | משה                  |   1 |



## usage

python scripts/Cita_Berel_inference.py \
  --input input.csv \
  --output output.csv \
  --model nbontemps/CitaBEREL \
  --text-col column_name

Optional arguments

| Argument         | Description                                          |
| ---------------- | ---------------------------------------------------- |
| `--tag-col`      | Gold tag column (default: `tag`)                     |
| `--id-col`       | ID column (default: `id`)                            |
| `--segment-size` | Number of words per inference segment (default: 50)  |
| `--no-report`    | Disable the classification report                    |

## Output csv

| id | Original | Token  | Prediction | new_tag |
| -: | -------- | ------ | ---------- | ------- |
|  0 | ויאמר   | ויאמר | LABEL_2    | LABEL_2 |
|  1 | ייי      | ייי    | LABEL_2    | LABEL_2 |
|  2 | אל    | אל  | LABEL_2    | LABEL_2 |





