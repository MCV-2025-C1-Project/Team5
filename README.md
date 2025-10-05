# Team5

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A short description of the project.

## Project Organization

```
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
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
│                         generated with `pip freeze > requirements.txt`
│
└── src   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes src a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── data                    <- Scripts to extract/load data
    │   └──extract.py
    │
    ├── descriptors             <- Scripts to compute image descriptors
    │   ├──descriptor_1.py
    │   └──descriptor_n.py
    │
    ├── distances               <- Scripts to compute distance measures
    │   ├── measure_1.py
    │   └── measure_n.py
    │
    ├── metrics                 <- Scripts to compute metrics
    │   ├── metric_1.py
    │   └── metric_n.py
    │
    ├── models                  <- Scripts to compute image retrieval and generate deliverables
    │   ├── method_1.py
    │   └── method_n.py
    │
    ├── tools                   <- Helper functions
    │   └── startup.py             <- Load parameters from yaml configuration file
    │
    └── visualization
        └──plots.py                <- Code to create visualizations
```

--------

