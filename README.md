# FRoGS
We introduced a form of "word2vec" for bioinformatics named Functional Representation of Gene Signature (FRoGS), where FRoGS vectors encode known human genes' functions. FRoGS has pretrained gene embeddings and can generate embeddings of gene signatures based on the embeddings of individual genes.

## Dependencies
- python 3.7.3</br>
- numpy 1.17.3</br>
- pandas 1.3.3</br>
- scikit-learn 1.0.1</br>
- tensorflow 2.0.0</br>

## Operating system
The software has been tested on the Rocky Linux 8.8 system.

## Installation
Run the command to download the package:
```
git clone https://github.com/chenhcs/FRoGS.git
```
## Using FRoGS gene embeddings to classify tissue specific genes
In this demonstration, we showcase a simple application of FRoGS gene embeddings for the classification of tissue-specific genes. Run the command within the `demo/` directory to initiate the demo:
```
python classifier.py
```
Within the `demo/data/` directory, we have provided three gene lists, each containing tissue-specific genes (Entrez Gene IDs) associated with a specific tissue. In the script `classifier.py`, we assign a vector representation to each gene using our pre-trained FRoGS gene embeddings. Then a t-SNE plot is generated to visualize the clustering patterns of genes based on their vector representations.
![alt text](https://github.com/chenhcs/FRoGS/blob/main/demo/tsne.png)

Finally in this example, we build a random forest classifier to predict the tissue specificity of genes based on their FRoGS vector representations.


## Training models for predicting compound targets using L1000 gene signatures
This section describes how to use pretrained gene embeddings to train a model that predict target genes of compounds from L1000 perturbagen gene signatures. Example data are provided in the data folder. To do this, run the command `python l1000_model.py` within the `src/` directory. Cross validation will be performed for a list of compounds that have target annotations. There are a number of arguments that need to be specified for the script. Run the command under the src directory to see these options:
```
python l1000_model.py -h
```
### Input format
Before using the script, the directories of following files and parameters need to be specified.
1. Path to a file that lists the compounds for cross validation (`--cpdlist_file`)
2. Path to a file containing target annotations of compounds (`--target_file`)

    By default, we will use Broad's compound target annotations.

3. Path to a file containing L1000 gene signatures (`--sig_file`)
4. Perturbagen type of gene signatures to use (`--perttype`)

    Choose target gene signatures of `shRNA` perturbation or `cDNA` perturbation to use.

5. Gene embeddings learned from GO annotations (`--emb_go`)

    By default, this will be the gene embeddings already-trained from Gene Ontology annotations saved in the data folder.

6. Gene embeddings learned from archs4 gene expression experiments (`--emb_archs4`)

    By default, this will be the gene embeddings already-trained from gene lists derived from ARCHS4 gene expression experiments saved in the data folder.
7. Number of training epochs (`--epochs`)
8. Path to a directory to save the predicted target ranking lists (`--outdir`)
9. Path to a directory to save trained models (`--modeldir`)

### Output
1. Learned model weights will be saved in the specified `--modeldir`
2. For each perturbagen type (shRNA or cDNA), a target ranking list will be saved in the specified `--outdir` for each compound in each cell line, named as `compoundID@cellline_perturbagen.txt`

    Each file contains three columns, the first column lists gene IDs, the second column indicates the rankings of genes, the last column indicates the predicted scores of the model for the compound gene pair. The higher the ranking, the more likely the gene will be targeted by the compound in the cell line.

### Test pretrained models on other compounds
For each perturbagen type (shRNA or cDNA), three models trained on all the Broad's target annotations are provided in the `saved_model` directory. Run the following script under the src directory to reproduce the predictions for compounds without Broad's target annotations combining the three models.
```
python l1000_inference.py
```

## Training gene embeddings and generate signature embeddings
This section describes how to train gene embeddings using your own data. To do this, run the `gene_vec_model.py` script. Again, example data are provided in the data folder. Run the command under the src directory to see the arguments that need to be specified:
```
python gene_vec_model.py -h
```
### Input format
1. Type of data from which the gene embeddings learned (`--datatype`)

    Here we consider either `go` or `archs4` data.

2. Path to a file containing GO term gene associations or ARCHS4 experiments gene associations (`--association_file`)

    A TAB separated file where each line starts with a GO/ARCHS4 category ID, following by the number of genes in the category, following by the genes associated with the category separated by comma.

3. Path to save the learned embeddings (`--outfile`)

### Output
1. A gene embedding file will be saved as the specified `--outfile`.

    A comma separated file where each line indicates a gene embedding. The first item in the line is the gene ID, following by a list of float numbers which is the embedding representation of the gene.

### Generate gene signature embeddings
Use the following command to run examples of generating gene signature embeddings from embeddings of individual genes:
```
python signature_embedding.py
```

## Run logistic regression (LR) models for aggregating multiple predictions
The code for the LR model and the trained LR model for Model L, Q, LQ, NCI60, and PSP are provided in the `LR_model` directory.
