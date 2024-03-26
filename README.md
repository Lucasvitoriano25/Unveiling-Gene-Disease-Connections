# Unveiling Gene-Disease Connections: A Graph Neural Network Approach for Biological Knowledge Graphs
[Training Data Download](https://drive.google.com/uc?export=download&id=1UxNXfc8ZvkP6F3M54cXbXpBRA_s-krJ4) | [Original Data](https://snap.stanford.edu/biodata/)

## Authors
- Joao Pedro Volpi, CentraleSupelec, `joao-pedro.monteiro-volpi@student-cs.fr`
- Lucas Vitoriano, CentraleSupelec, `lucasvitoriano.queirozlira@student-cs.fr`
- Lucas Jose Veloso, CentraleSupelec, `lucasjose.velosodesouza@student-cs.fr`

## Approach
This study explores the use of Graph Neural Networks (GNNs) for identifying gene-disease associations through a Knowledge Graph derived from the Stanford Biomedical Network Dataset Collection, which includes over 21,000 connections. By representing genes and diseases as nodes, and their relationships as edges, we enable GNNs to uncover complex patterns and predict novel associations. This approach not only reveals previously unknown gene-disease links but also offers fresh insights into the molecular basis of diseases.

## Knowledge graph representation 

![Graph](https://github.com/Lucasvitoriano25/Unveiling-Gene-Disease-Connections/assets/52925699/1d0770f4-9905-4ed1-ace9-a05ed093bcc5)
The image depicts several instances from our knowledge graph, illustrating the critical challenge we address in this work. Specifically, we focus on utilizing the existing relations (highlighted in orange) and attributes within our incomplete knowledge graph to predict possible missing relationships (indicated in pink) between genes and diseases.

## Results

Our experimental setup evaluated three different Graph Neural Network (GNN) architectures: SAGEConv, GraphConv, and GATv2. The performance of each model was measured in terms of accuracy and F1 score across epochs during training and testing phases. 

![Accuracy](https://github.com/Lucasvitoriano25/Unveiling-Gene-Disease-Connections/assets/52925699/d4fe99fd-a040-41ca-a988-5a9c38500846)
![F1_score](https://github.com/Lucasvitoriano25/Unveiling-Gene-Disease-Connections/assets/52925699/59c3384b-1189-4ee9-ac26-44b5e070e805)
