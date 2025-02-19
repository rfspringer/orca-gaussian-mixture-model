# Orca Vocalization Clustering Using Gaussian Mixture Models

This project implements Gibbs sampling on a Gaussian mixture model to analyze and cluster orca vocalizations from the Orcasound dataset. The implementation uses YAMNet embeddings to process audio features and identifies distinct vocalization patterns.

## Overview

The project analyzes a dataset from Orcasound containing:
- 5,537 training samples
- 828 test samples
- Human-confirmed orca vocalizations
- Audio features extracted using YAMNet embeddings (1024-dimensional)

## Technical Implementation

### Feature Extraction
- Parsed TSV annotation files to extract relevant audio segments
- Utilized YAMNet (pretrained audio-classification neural network)
- Generated 1024-dimensional embeddings for each audio file

### Clustering Method
- Implemented Gibbs sampling for Gaussian mixture models
- Parameters:
  - sigma: 0.01
  - alpha_0: 1
  - eta: 0.01
  - K: 3 (number of clusters)
  - max_iterations: 80

### Model Evaluation
- Evaluated held-out log likelihood on test dataset
- Monitored convergence through log joint probability
- Validated results through visual analysis of feature dimensions
- Performed qualitative assessment of audio samples from each cluster

## Results

The model identified three distinct clusters of orca vocalizations:
1. High-pitched squeals
2. Chirpy varied squeals
3. Whistly noises

Example audio files for each cluster:
- Cluster 1: Rows 2, 4, 4024
- Cluster 2: Rows 4167, 4954, 5017
- Cluster 3: Rows 48, 1776, 5256

Audio samples can be found in the [Orcasound dataset](https://github.com/orcasound/orcadata/wiki/Orca-training-data).

## Key Findings

- Model converged quickly with 3 clusters showing optimal performance
- Clustering remained effective despite varying background noise levels
- Some dimensions showed clearer 2-cluster patterns
- Gaussian mixture model may not be optimal for all vocalization patterns

## Future Improvements

1. Further clustering to identify orca-linguistic patterns within vocalization styles
2. Exploration of alternative embedding methods more suited to subtle audio differences
3. Investigation of non-Gaussian mixture models for better pattern recognition
4. More granular analysis of vocalization subcategories

## Dependencies

- NumPy
- SciPy
- Matplotlib
- Pandas
- YAMNet (for audio feature extraction)

## Usage

The main script performs the following operations:
1. Loads training and test embeddings from CSV files
2. Runs Gibbs sampling with specified parameters
3. Generates visualization plots
4. Evaluates model performance using held-out likelihood
5. Outputs cluster assignments and example files

Course project for STAT 6701, Columbia University
