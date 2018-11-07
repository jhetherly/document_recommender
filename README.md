# document_recommender
Creates graph of related Wikipedia pages using a very simple [tf-idf](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) similarity criterion.

* [Installation](#installation)
* [Usage](#usage)
* [Output](#output)

<a name="installation"/>

## Installation
This package relies solely on [Anaconda](https://www.anaconda.com/) to manage it's python package dependencies.
To create the `conda` environment, follow the instructions listed on this [page](https://conda.io/docs/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) using the `dr.yml` file.

<a name="usage"/>

## Usage
Once the `conda` environment is created, activate the environment by following the system-specific instructions on this [page](https://conda.io/docs/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).
The steps to create the similarity graph is broken down into four steps:
1. Download raw pages
2. Make a vocabulary from these raw pages
3. Compute the most similar pages using a TF-IDF criterion
4. Create a similarity plot
All the scripts are configurable through JSON file location in the `settings` directory.
The scripts are generally "self-documenting" and follow a simple procedural structure.

### Download Wiki Data
```bash
python download_wiki_data.py --download_settings <settings file>
```
The download settings file has the following options:
* topic: the Wikipedia Category to start scraping from
* min_pages: the minimum number of pages to scrape
* buffer: additional percentage by which to increase the `min_pages` (to "buffer" against empty pages)
* save_dir: directory in which to save the raw pages

### Make vocab
```bash
python make_vocab.py --download_settings <settings file> --parse_settings <settings file>
```
The parse settings file has the following options:
* exclude_vocab: list of strings to exclude from vocabulary
* min_page_vocab: pages must contain a minimum number of vocabulary words to be included in the analysis
* plot_top_k: plot the top-k page vocabulary frequency or cumulative distribution
* plot_cumulative: boolean that indicates either cumulative or frequency distributions
* save_dir: directory in which to save the JSON vocab and PDF plots

### Compute page similarity
```bash
python compute_tf-idf.py --download_settings <settings file> --parse_settings <settings file> --similarity_settings <settings file>
```
The similarity settings file has the following options:
* vocab_top_k: only consider the top-k vocab words
* graph_top_k: take the top-k similar pages
* metric: scikit-learn matric to use in the similarity calculation
* save_dir: directory in which to save the JSON similarity results (and graph in next step)

### Create similarity graph
```bash
python compute_tf-idf.py --download_settings <settings file> --parse_settings <settings file> --similarity_settings <settings file>
```

<a name="output"/>

## Outputs
* the downloaded raw pages are stored in `data/wiki/<topic>/*.html` by default
* the vocabulary JSON files are stored in `artifacts/wiki/<topic>/vocab/*.json` by default
* the vocabulary PDF plot files are stored in `artifacts/wiki/<topic>/vocab/*.pdf` by default
* the similarity results are stored in `artifacts/wiki/<topic>/graph/*.json` by default
* the similarity graph is stored in `artifacts/wiki/<topic>/graph/*.pdf` by default