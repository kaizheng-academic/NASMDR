# NASMDR
a framework for miRNA-drug resistance prediction through efficient neural architecture search and graph isomorphism networks (NASMDR).




# Requirements
* requirements.txt



# Files:

1.data

drug_M2V_64.csv: M2V-based descriptor D<sub>mol2vec</sub>.

drug_GIN_64.csv:  Graph Isomorphism Networks-based descriptor D<sub>gin</sub>.

miRNA_kmer.csv:  Kmer-based miRNA sequence feature descriptor M<sub>Kmer</sub>.

miRNA_kmerSparseMatrix_pca.csv: k-mer Sparse Nonnegative Matrix Factorization-based miRNA sequence feature descriptor M<sub>KSNMF</sub>.

data_3000.csv: The benchmark dataset of miRNA-drug resistance associ-ations. 

drug+smiles.csv: Drug ID and structure information.

miRNA+seq.csv: miRNA ID and sequence information.






2.Code



train.py: This function can test the predictive performance of our model under two,five,ten-fold cross-validation.

KSNMF.ipynb: This function can get the k-mer Sparse Nonnegative Matrix Factorization-based miRNA sequence feature descriptor M<sub>KSNMF</sub>.

# Train and test folds

python train.py 

All files of Data and Code should be stored in the same folder to run the model.


# Contact 
If you have any questions or suggestions with the code, please let us know. Contact Kai Zheng at kaizheng@csu.edu.cn
