# Content-Based Image Retrieval (CBIR)

This project implements a **query-by-example image retrieval system** designed to find paintings in the *Can Framis Museum* image collection based on their **visual content**.  

Developed by **Team 5** as part of the **C1 – Content Based Image Retrieval** course assignment at the *Master’s in Computer Vision (UPC-CVC)*, academic year 2025-2026.


## Overview
This project implements a Content-Based Image Retrieval (CBIR) system designed to search paintings in the Can Framis Museum dataset based on color, texture, and local descriptors.
The goal is to explore classical and modern feature extraction techniques for visual similarity search.

The following diagram illustrates the CBIR workflow implemented in this project.
<p align="center"> <img src="reports/figures/image_retrieval.png" alt="Overview of the Content-Based Image Retrieval pipeline" width="600"/> </p>

## Week 1

The first milestone focuses on global color-based image retrieval, using single-resolution color histograms as visual descriptors.

Color histograms are one of the simplest and most intuitive ways to represent an image, summarizing the distribution of pixel colors and enabling comparisons through distance metrics

### Steps implemented:

1. **Indexing the database** Compute and store color histograms for all database images (performed offline).

2. **Feature extraction for query images**
Compute the same descriptor type for each query image.

3. **Similarity computation**
Compare query descriptors with the database using distance metrics.

4. **Ranking and retrieval**
Sort database images according to similarity and return the top-k most visually similar paintings.

### Features

- **Multiple Image Descriptors:**
  - Grayscale Histogram
  - RGB Histogram
  - HSV Histogram
  - YCbCr Histogram
  - LAB Histogram

- **Multiple Distance Metrics:**
  - Euclidean Distance
  - L1 (Manhattan) Distance
  - Chi-Squared Distance
  - Histogram Intersection
  - Hellinger Distance
  - Cosine Distance
  - Canberra Distance
  - Bhattacharyya Distance
  - Jensen-Shannon Divergence
  - Correlation Distance

- **Evaluation Metrics:**
  - Mean Average Precision at K (mAP@K)
  - Top-K retrieval accuracy

- **Visualization:**
  - Heatmaps for comprehensive evaluation
  - Query-retrieval result visualizations


## Project Organization

The project follows a modular and reproducible structure, inspired by the cookiecutter data science template. Each folder has a clear purpose to ensure scalability and team collaboration.

```
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── raw            <- The original, immutable data dump.
│   ├── descriptors    <- Descriptors extracted from images ready to use for retrieval.
│   └── results        <- Results obtained from executing the retrieval.
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks/         <- Jupyter notebooks.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         team5 and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│
└── src   <- Source code for use in this project.
    ├── __init__.py             <- Makes src a Python module
    ├── config.py               <- Store useful variables and configuration
    ├── data/                   <- Scripts to extract/load data
    ├── descriptors/            <- Scripts to compute image descriptors
    ├── distances/              <- Scripts to compute distance measures
    ├── metrics/                <- Scripts to compute metrics
    ├── models/                 <- Scripts to compute image retrieval and generate deliverables
    ├── tools/                  <- Helper functions
    └── visualization/          <- Code to create visualizations
```

--------

## Installation

#### Clone the repository
``` bash
git clone https://github.com/MCV-2025-C1-Project/Team5.git
cd Team5
```

#### Create and activate virtual environment
It’s recommended to use a virtual environment to avoid dependency conflicts.
```bash
python -m venv .venv
source .venv/bin/activate
```

#### Install dependencies
```bash
pip install -r requirements.txt
```

#### Add dataset files

Place the datasets in the `data/raw/` folder as follows:
```
├── data
    └── raw
        ├──BBDD/            <- Database of paintings
        ├──qsd1_w1/         <- Query set for week 1 development
        └──qst1_w1/         <- Query set for week 1 testing
```


## Usage

Once the environment and data are set up, you can execute the pipeline to perform image retrieval.

### Index the database

The descriptor computator module automates the extraction of global color descriptors for all images in a given dataset folder. It forms the first stage of the CBIR pipeline, generating a numerical representation for each image based on its color distribution.

#### Key Features

- Supports multiple color spaces: Grayscale, RGB, HSV, Lab, and YCbCr.
- Supports agregation of consecutive values of the histogram in the same bin
- Allows configurable histogram binning through the values_per_bin argument.
- Automatically loads all .jpg or .jpeg images from the input folder.
- Stores computed descriptors and metadata in a single .pkl file for later use in retrieval and evaluation.

#### Workflow

1. **Load images** from the specified dataset folder (--input).
2. **Compute histograms** using the chosen descriptor (--descriptor).
3. **Aggregate and serialize** all image descriptors into a single .pkl file under data/descriptors/.

#### Example command

``` bash
python src/descriptors/compute_descriptors.py \
    --descriptor hsv \
    --input data/raw/BBDD \
    --outdir data/descriptors \
    --values_per_bin 2
```

#### Command-Line Arguments

Run `python main.py --help` to see all available options:

```
usage: compute_descriptors.py [-h] --descriptor {grayscale,hsv,lab,rgb,ycbcr} --input INPUT [--outdir OUTDIR] [--values_per_bin VALUES_PER_BIN]

Compute 1D image descriptors.

options:
  -h, --help            show this help message and exit
  --descriptor {grayscale,hsv,lab,rgb,ycbcr}
                        Which descriptor to run: grayscale | hsv | lab | rgb | ycbcr
  --input INPUT         Folder with images (BBDD, QSD1, QST1)
  --outdir OUTDIR       Output folder
  --values_per_bin VALUES_PER_BIN
                        Intensity values per bin
```

### Evaluate descriptors and distances

#TODO

#### Hyperparameters

Key hyperparameters can be configured:

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `descriptor` | Image descriptor type | `hsv` | rgb, hsv, ycbcr, lab, grayscale |
| `distance_metric` | Similarity metric | `chi_squared` | euclidean, l1, chi_squared, histogram_intersection, hellinger, cosine, canberra, bhattacharyya, jeffrey, correlation |
| `values_per_bin` | Histogram bin size | `1` | 1-256 (1=256 bins, 2=128 bins, etc.) |
| `k` | Number of top results | `5` | Any positive integer |


### Basic Usage

Run with default settings (HSV descriptor + Chi-Squared distance):
```bash
python #TODO
```


This will:
- Test all 50 descriptor-distance combinations (5 descriptors × 10 distance metrics)
- Generate heatmaps showing performance
- Find the best configuration
- Create visualizations with the best configuration


## Team members:

- OREGI LAUZIRIKA, Lore - loreoregi@gmail.com
- ROSELL MURILLO, Marina - marrosmur@gmail.com
- ARTERO PONS, Marc - marteropons@gmail.com
- PURKAYASTHA, Kunal - kunalpurkayastha09@gmail.com
