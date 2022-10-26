# FEGS
Functional Embedding of Gene Signatures

## Dependencies
- python 3.7.3</br>
- numpy 1.17.3</br>
- pandas 1.3.3</br>
- scikit-learn 1.0.1</br>
- tensorflow 2.0.0</br>

## Training models for predicting compound targets using L1000 gene signatures
This section describes how to use pretrained gene embeddings to train a model that predict target genes of compounds from L1000 perturbagen gene signatures. Example data are provided in the data folder. To do this, run the `l1000_model.py` script. Cross validation will be performed for a list of compounds that have target annotations. There are a number of arguments that need to be specified for the script. Run the command to see these options:
```
python l1000_model.py -h
```
### Input format
Before using the script, the following files need to be prepared.
1. Path to a file that lists the compounds for cross validation (`--cpdlist_file`)
2. Path to a file containing target annotations of compounds (`--target_file`)
3. Path to a file containing L1000 gene signatures (`--sig_file`)
4. Perturbagen type of gene signatures to use (`--perttype`)

  Choose target gene signatures of `shRNA` perturbation or `cDNA` perturbation to use.

5. Gene embeddings learned from GO annotations (`--emb_go`)

  By default, this will be the gene embeddings already-trained from Gene Ontology annotations saved in the data folder.

6. Gene embeddings learned from archs4 gene expression experiments (`--emb_archs4`)

  By default, this will be the gene embeddings already-trained from gene lists derived from ARCHS4 gene expression experiments saved in the data folder.

7. Path to a directory to save the predicted target ranking lists (`--outdir`)
8. Path to a directory to save trained models (`--modeldir`)

### Output
1. Learned models weights will be saved in the specified `--modeldir`
2. For each perturbagen type (shRNA or cDNA), a target ranking list will be saved for each compound in each cell line, named as `compoundID@cellline_perturbagen.txt`

Each file contains three columns, the first column lists the ID of genes, the second column indicates the rankings of genes, the last column indicates the predicted scores of the model for the compound gene pair. The higher the ranking, the more likely the gene will be targeted by the compound in the cell line.

### Test pretrained models on other compounds


## Training gene embeddings and generate signature embeddings
Use the `gene_vec_model.py` script to train . Run the command to see the arguments need to be specified:
```
python gene_vec_model.py -h
```
### Input format

### Output
1.
