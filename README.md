# HGLMA
## Overview

```HGLMA``` is a computational framework designed for predicting missing reactions (gap-filling) in genome-scale metabolic models (GEMs). It leverages a combination of a Hypergraph and a multi-head attention mechanism to learn complex dependencies within the metabolite-reaction hypergraph. The framework computes prediction scores for candidate reactions, which are then integrated with a calculated biochemical similarity score for final ranking and selection of reactions to be added to the model.

## Description
Here we have three folders: \
```HGLMA``` is for the evaluation of reaction prediction performance.\
```HGLMA_recovery``` is for the evaluation of reaction recovery.\
```HGLMA_gapfilling``` is dedicated to the gap-filling of draft GEMs.
The ```data``` folder contains all the experimental results, which are also available in the supplementary materials.


## System Requirements
### Dependencies
The package depends on the Python scientific stack:

```
torch: 2.1.0
dgl: 2.4.0 + cu121
dhg: 0.9.5
numpy: 1.23.5
pandas: 1.1.1
cobra: 0.22.1
optlang: 1.5.2
python-libsbml: 5.19.0
networkx: 2.8.8
scipy: 1.10.1
scikit-learn: 0.23.1
matplotlib: 3.3.4
tqdm: 4.66.5
rdkit: 2022.3.5
node2vec: 0.4.6
openpyxl: 3.1.
```

Users are required to additionally install the ```cplex``` solver (https://www.ibm.com/analytics/cplex-optimizer) from IBM to run the package. Note that cplex only works with certain python versions (e.g., CPLEX_Studio12.10 has APIs for python3.6 an python3.7).

## Usage
- We extracted metabolite features from metabolite SMILES and molecular structures using the pre-trained models ```ChemBERTa``` and ```GraphMVP```, respectively. The pre-generated metabolite features are available at the link below. 
- The link to the pre-trained model ```ChemBERTa``` is as follows: https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1
- The link to the pre-trained model ```GraphMVP``` is as follows: https://github.com/chao1224/GraphMVP
- If you wang to use ```/HGLMA```, please just download pre-trained node embedding files for all metabolites **`metabolite_emb_2816.pkl`** ([download link](https://drive.google.com/file/d/1-21en06Ds1Bo11ljRYP-1qelxAb91QJb/view?usp=sharing)) and place it in the `data` folder.
- If you wang to use ```/HGLMA_recovery``` or ```/HGLMA_gapfilling```, download pre-trained node embedding files for all metabolites **`metabolite_emb_2816.pkl`** ([download link](https://drive.google.com/file/d/1-21en06Ds1Bo11ljRYP-1qelxAb91QJb/view?usp=sharing)) and place it in the `data` folder, and download bigg reactions pool files for all metabolites **`bigg_universe.xml`** ([download link](https://drive.google.com/file/d/1jORDo7qQt3pnjS2ZTqmFMrKckzDeZM4c/view?usp=sharing)) and place it in the `data/pools` folder.
- To run the demonstration, you need to navigate to the corresponding folder first, then type "python3 main.py" in your terminal.


### Prepare your input files

1. The folder ```HGLMA_gapfilling/BiGG Models``` contains 1 GEM from Zimmermann et al. as examples. GEM is a xml file.

The draft GEM is derived from: Zimmermann, J., Kaleta, C. & Waschina, S. gapseq: informed prediction of bacterial metabolic pathways and reconstruction of accurate metabolic models. Genome Biol 22, 81 (2021). https://doi.org/10.1186/s13059-021-02295-1

2. The folder ```HGLMA_gapfilling/data/pools``` need to contain a reaction pool under name ```bigg_universe.xml```. Each pool is a GEM that has the extension ```.xml```. To use your own pool, remember to rename it to ```universe.xml```. Also remember to edit ```EX_SUFFIX``` and ```NAMESPACE``` in the input_parameters.txt to specify the suffix of exchange reactions and which namespace of biochemical reaction database is used. For ```NAMESPACE```, we currently only support ```bigg```.

The reaction pool is derived from: Chen C, Liao C, Liu Y Y. Teasing out missing reactions in genome-scale metabolic networks through hypergraph learning. Nature Communications, 2023, 14(1): 2375.
