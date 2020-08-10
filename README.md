# GraphER
The code of our AAAI'20 paper "GraphER: Token-Centric Entity Resolution with Graph Convolutional Neural Networks"

# Usage:

* Firstly, download the embedding file "glove.6B.200d.txt" from https://nlp.stanford.edu/projects/glove/, and put it into Embedding folder

* Secondly, initialize the ER-graph, e.g., for Amazon-Google dataset
python graph_att.py Amazon-Google

* Finally, run the training:
python train.py
