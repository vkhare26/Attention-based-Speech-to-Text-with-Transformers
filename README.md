# Attention-based-Speech-to-Text-with-Transformers

This repository contains the implementation of an attention-based end-to-end speech-to-text deep neural network using a Transformer architecture. The project focuses on building an automatic speech recognition (ASR) system that converts audio features (filterbank or MFCC) into character-level transcriptions, evaluated using Character Error Rate (CER) on a Kaggle competition.

# Project Overview
The goal of this project is to implement a Transformer-based model for speech-to-text transcription, following an encoder-decoder architecture with multi-head attention mechanisms. The system processes audio features (filterbank or MFCC) and generates character sequences, handling variable-length sequences with padding and masking. Key features include:

Dataset: LibriSpeech (train-clean-100, dev-clean, test-clean) with 100 hours of audio and text-for-LM for pre-training.
Model: Transformer with configurable encoder and decoder layers, multi-head attention, and positional encoding.
Training Strategies:
Pretraining the decoder as a conditional language model (dec_cond_lm mode).
Warming up the encoder while freezing embeddings and decoder.
Full Transformer training with combined CTC and Cross-Entropy loss.


Inference: Supports greedy and beam search decoding.
Evaluation Metrics: Character Error Rate (CER), Word Error Rate (WER), and Levenshtein Distance (LD).
SpecAugment: Applied for data augmentation during training.

The implementation achieves a CER of ≤60% on the Kaggle leaderboard, meeting the early submission requirement.

# Repository Structure

config.yaml: Configuration file specifying dataset paths, model hyperparameters, training settings, and SpecAugment parameters.
hw4p2.py: Main script containing the Transformer implementation, dataset handling, training, validation, and testing logic.
submission.csv: Generated file for Kaggle submission with test predictions.
README.md: This file, providing project documentation.

# Prerequisites
To run this project, ensure you have the following:
Software Requirements

Python 3.8 or higher
PyTorch 2.0 or higher (with CUDA support for GPU acceleration)
Additional libraries (install via requirements.txt):
torchaudio
transformers
numpy
pandas
tqdm
scipy
seaborn
matplotlib
wandb (for experiment tracking)
torchinfo (for model summary)



# Hardware Requirements

GPU (recommended for faster training; e.g., NVIDIA GPU with CUDA support)
At least 16GB of RAM (for loading datasets and training)
Disk space for the LibriSpeech dataset (~30GB for train-clean-100)

# Dataset

Download the LibriSpeech dataset partitions (train-clean-100, dev-clean, test-clean, text-for-LM) and place them in the directory specified in config.yaml (default: /home/name/data/hw4p2).
Ensure the dataset is preprocessed into filterbank (FBank) features and text transcriptions as provided by the course.


# Configuration
The config.yaml file controls the dataset paths, model architecture, training hyperparameters, and SpecAugment settings. Key parameters include:

# Dataset:
root: Path to the dataset directory.
feat_type: "fbank" or "mfcc".
num_feats: Number of features (e.g., 80 for FBank).
batch_size: 16 (adjust based on GPU memory).


# SpecAugment:
specaug: true: Enables SpecAugment.
freq_mask_width_range: 10.
num_freq_mask: 6.
time_mask_width_range: 50.
num_time_mask: 8.


# Model:
d_model: 384 (model dimension).
d_ff: 1536 (feedforward dimension).
enc_num_layers: 4 (encoder layers).
enc_num_heads: 8 (encoder attention heads).
dec_num_layers: 4 (decoder layers).
dec_num_heads: 8 (decoder attention heads).


# Training:
optimizer: "AdamW".
learning_rate: 2e-4.
scheduler: "ReduceLR".
epochs: 60 (total epochs across training stages).
use_ctc: true (enables CTC loss with weight 0.5).



Modify config.yaml to adjust these settings based on your hardware or experimental needs.
Usage
Training the Model
The script implements a multi-stage training pipeline:

Warm-up the entire model (mode="full", 4 epochs).
Pretrain the decoder as a conditional language model (mode="dec_cond_lm", 24 epochs).
Warm-up the encoder while freezing embeddings and decoder (mode="full", 4 epochs).
Train the full Transformer with all layers unfrozen (mode="full", remaining epochs).

To train the model, run:
python hw4p2.py


The script automatically logs training metrics (loss, perplexity, CER, WER, Levenshtein Distance) to W&B if use_wandb: true in config.yaml.
Checkpoints are saved in the checkpoints/ directory under the experiment root (run_name in hw4p2.py).
Attention weight plots are saved in the attention_imgs/ directory.
Validation outputs are saved as JSON files in the out_text/ directory.

Testing and Kaggle Submission
After training, the script loads the best model (based on Levenshtein Distance) and generates predictions on the test-clean partition. The predictions are saved to submission.csv for Kaggle submission.
To generate test predictions manually (after training):

Ensure the best model checkpoint is available at /home/name/{run_name}/checkpoints/checkpoint-best-loss-modelfull.pth.
Run the test step (already included at the end of hw4p2.py):python hw4p2.py


The submission.csv file will be created in the working directory with the format:Index,Labels
0,"transcription 1"
1,"transcription 2"
...



Submit submission.csv to the Kaggle competition for evaluation.
Model Architecture
The Transformer model consists of the following components:

SpeechEmbedding:
Conv2DSubsampling for time and feature downsampling.
Optional BiLSTM for conditional LM pretraining.


Encoder:
Positional encoding.
Stack of enc_num_layers encoder layers (self-attention, FFN, residual connections, layer normalization).
CTC head for auxiliary training.


Decoder:
Target embedding and positional encoding.
Stack of dec_num_layers decoder layers (masked self-attention, cross-attention, FFN, residual connections, layer normalization).
Final linear layer for token prediction.


Inference:
Greedy search (default).
Beam search (beam width = 5, implemented in Decoder.recognize_beam_search).



The model is summarized using torchinfo during initialization, providing a detailed breakdown of parameters and layers.
Results
The model achieves the following performance on the validation set (dev-clean) after training:

Levenshtein Distance: (As reported in the final validation step).
Word Error Rate (WER): (As reported in the final validation step).
Character Error Rate (CER): (As reported in the final validation step).
