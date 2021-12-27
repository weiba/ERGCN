#Identifying Cancer Subtypes Using a Residual Graph Convolution Model on a Sample Similarity Network

## Dependencies
- CUDA 10.2
- python 3.6.9
- pytorch 1.8.0
- torch-scatter 2.0.7
- torch-sparse 0.6.10
- torch-geometric 1.7.2
- networkx 2.1
- scikit-learn

## Datasets

The coad contains three datasets(BRCA,GBM,LUNG).We use produce_adjacent_matrix.py to generate adjacency matrix, and then use produce _data.py to generate edge pairs.

## Results
The `data` folder contains results

- Running examples
```
python train.py
```



