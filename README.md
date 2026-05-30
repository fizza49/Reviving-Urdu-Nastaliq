# Reviving Urdu Nastaliq

Comparative handwritten Urdu text recognition notebook for the `NUST-UHWR` and `UNHD` datasets.

This project evaluates three OCR pipelines, measures their performance with CER/WER, and optionally applies Groq-powered LLM post-correction to refine OCR output.

## Overview

The notebook is designed as a standalone experimental pipeline for handwritten Urdu OCR. It:

- loads and validates the datasets,
- builds writer-aware train/validation/test splits,
- normalizes Urdu text and constructs a training vocabulary,
- runs three recognition pipelines with increasing model complexity,
- applies optional LLM post-correction through Groq,
- exports metrics, plots, predictions, and intermediate artifacts.

The goal is not only to compare model accuracy, but also to study where classical OCR methods break down and how sequence models and language-aware correction can improve results.

## Pipelines

### Pipeline 1 - HOG + SVM Baseline

This is the classical OCR baseline.

Functionality:

- extracts Histogram of Oriented Gradients (`HOG`) features from word images,
- classifies words with a linear SVM,
- evaluates a closed-vocabulary subset,
- optionally uses an n-gram language model for lightweight post-processing.

Purpose:

- provides a fast, interpretable baseline,
- shows how far handcrafted features can go on Urdu handwriting,
- serves as a reference point for the deep learning models.

### Pipeline 2 - CNN-BiLSTM-CTC

This is the main sequence-recognition pipeline.

Functionality:

- preprocesses line images for sequence modeling,
- uses a CNN encoder to extract visual features,
- models temporal structure with BiLSTM layers,
- decodes character sequences with CTC,
- reports CER/WER on validation and test splits.

Purpose:

- recognizes full Urdu text lines rather than isolated words,
- captures stroke-level and sequence-level structure,
- represents the core deep learning OCR approach in the notebook.

### Pipeline 3 - SwinHTR Style OCR + LLM Correction

This is the strongest recognition pipeline in the notebook, followed by optional correction.

Functionality:

- runs a stronger OCR model for handwritten Urdu transcription,
- supports beam-search and greedy decoding comparisons,
- can apply Groq LLM post-correction to fix OCR mistakes,
- evaluates raw OCR output and corrected output separately.

Purpose:

- improve recognition quality using a stronger model backbone,
- test whether language-aware correction helps Urdu OCR,
- separate the effect of OCR from the effect of post-processing.

## Datasets

The notebook uses two handwritten Urdu datasets:

- `NUST-UHWR`
- `UNHD`

The NUST split files are used to keep the evaluation writer-aware. UNHD is used primarily to expand the training vocabulary and provide additional training data.

## Evaluation

The notebook reports:

- `CER` - Character Error Rate
- `WER` - Word Error Rate
- validation accuracy for the baseline classifier
- raw OCR vs corrected OCR metrics for the LLM stage

It also exports plots and summary files so the results can be inspected outside the notebook.

## Groq LLM Correction

The notebook supports optional Groq-based post-correction for OCR output.

To enable it, add your key in Kaggle Secrets using the secret name:

- `GROQ_API_KEY`

If no key is found, the notebook skips the correction stage and continues with the OCR results only.

## Setup

### Kaggle

1. Add the `NUST-UHWR` and `UNHD` datasets to the notebook.
2. Add the Groq secret if you want to run LLM post-correction.
3. Run the notebook from top to bottom.

### Local Environment

Install the key dependencies used by the notebook:

- `torch`
- `transformers`
- `scikit-image`
- `albumentations`
- `jiwer`
- `editdistance`
- `nltk`
- `groq`
- `huggingface_hub`

## Notebook Flow

1. Environment setup and imports
2. Dataset loading and inspection
3. Writer-aware split creation
4. Text normalization and vocabulary building
5. Image preprocessing and augmentation
6. Pipeline 1 baseline training/evaluation
7. Pipeline 2 deep OCR training/evaluation
8. Pipeline 3 OCR inference
9. Groq LLM post-correction
10. Metrics export and result packaging

## Strengths Of The Notebook

- Compares classical OCR and deep sequence models in one place.
- Uses writer-aware evaluation to reduce leakage.
- Includes Urdu normalization and vocabulary handling.
- Separates raw OCR performance from LLM-corrected performance.
- Exports a full result bundle for analysis and presentation.

## Limitations

- Pipeline 1 is intentionally weak and should be treated as a baseline.
- Results are sensitive to preprocessing and label normalization.
- Groq correction depends on a valid secret being available in Kaggle.
- The dataset split logic and vocabulary should be rerun whenever the data changes.


## Citation / Usage

If you use this notebook or its experimental setup in your own work, please cite the dataset sources and describe the pipeline configuration you used, especially the preprocessing and post-correction stages.

## Contributors

- [Fizza Kashif](https://github.com/fizza49)
- [Tamkeen Sara](https://github.com/Tamkeen-Sara)
- [Muhammad Furqan Raza](https://github.com/frqnrza)
