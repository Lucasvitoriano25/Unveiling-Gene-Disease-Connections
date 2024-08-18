# Deep Learning for Genetic Analysis: A Graph Neural Network Approach for Biological Knowledge Graphs
[Training Data Download](https://drive.google.com/uc?export=download&id=1UxNXfc8ZvkP6F3M54cXbXpBRA_s-krJ4) | [Original Data](https://snap.stanford.edu/biodata/)



## Introduction
The intersection of genetics and disease pathology represents a critical area of biomedical research, with the potential to significantly advance our understanding of disease mechanisms and inform the development of targeted therapies. The Stanford Biomedical Network Dataset Collection provides a comprehensive resource for exploring this intersection, offering detailed datasets on gene-disease associations, disease classifications, and gene annotations. This work utilizes Graph Neural Networks (GNNs) to delve into the intricate relationships captured within these datasets and the _Knowledge Graph_ we constructed from them. Our primary focus is on the disease-gene association network, aiming to perform **Knowledge Graph Completion**. Through this process, we strive to discover new connections between diseases and genes, enriching our understanding of their relationships.

The Graph Neural Networks (GNNs), a class of deep learning models designed for graph data, are well-suited to model the intricate, non-linear relationships present in biological data. By representing genes and diseases as nodes and their associations as edges, GNNs can learn from the topology of the network as well as node and edge attributes, enabling the identification of novel gene-disease links and the prediction of disease phenotypes based on genetic data.

This study aims to harness the disease-gene association network dataset, containing information on 7,813 nodes and 21,357 edges, to uncover patterns and associations that may not be immediately apparent through traditional analysis methods.

## The Dataset

<p align="center">
  <img src="https://github.com/user-attachments/assets/532bdfca-4c5b-4ebc-a88a-27f52032ca27" alt="SNAP Logo" height="200px" />
</p>

The **[Stanford Biomedical Network Dataset Collection](http://snap.stanford.edu/biodata)**[1] encompasses a rich compilation of biomedical datasets, containing information of drugs, proteins, cells, tissues, etc. We will focus on gene-disease connections. This section elaborates on the composition of the disease and gene datasets within the collection.

### Disease Datasets
The disease datasets are comprehensive, incorporating disease names, definitions, synonyms, and mappings to disease categories. These datasets serve as a fundamental resource for understanding the molecular basis and classification of various diseases:

- **Disease Descriptions and Synonyms**: Derived from the Disease Ontology, this dataset (1.7MB, `D-DoMiner_miner-diseaseDOID`) contains detailed descriptions and features of diseases. It is an essential tool for annotating disease-related information with standardized ontology terms.
- **Disease Descriptions**: Focusing on molecular diseases and environmentally influenced medical conditions, this dataset (3.3MB, `D-MeshMiner_miner-disease`) provides synopses, including names, definitions, and synonyms of diseases. It draws references from the Comparative Toxicogenomics Database and the MINER network, highlighting its utility in understanding disease mechanisms and associations.
- **Classification of Diseases into Disease Categories**: This dataset (15KB, `D-DoPathways_diseaseclasses`) offers a mapping of diseases to their respective categories, based on etiology and anatomical location. It includes diverse examples such as Marfan syndrome (monogenic diseases), rheumatoid arthritis (musculoskeletal system diseases), and liver neoplasms (cancers), facilitating a structured approach to disease classification.

### Gene Datasets
- **Gene Names, Descriptions, and Synonyms**: This dataset (12MB, `G-SynMiner_miner-geneHUGO`) comprises information on **35,000** human genes, including their names, descriptions, synonyms, familial relationships, and chromosomal locations. It forms the backbone of our gene-centric analyses, enabling the exploration of gene functions, relationships, and their roles in diseases.

### Disease-gene Association Network
This dataset (`DG-AssocMiner_miner-disease-gene`, 818KB) represents a disease-gene association network that encapsulates information on genes associated with diseases. In the first sketch of our Knowledge Graph (KG), we define that genes and disease correspond to nodes, and edges denote the associations between them. It provides a structured means to analyze the network of disease-gene interactions.

#### Dataset Statistics
- **Number of Nodes: 7,813**
  - Disease Nodes: 519
  - Gene Nodes: 7,294
- **Number of Edges: 21,357**

This collection provides a comprehensive framework for our study. The disease datasets, with their rich descriptive and classificatory information, combined with the gene datasets' extensive coverage of gene attributes, offer an opportunity to explore and predict gene-disease relationships.

## Modeling as a Knowledge Graph

We've transformed the Stanford Biomedical Network Dataset Collection into a Knowledge Graph (KG) using the Web Ontology Language (OWL). This setup provides a semantic framework to map entities and relationships in the biomedical field, enhancing our ability to analyze and infer data. Key components include:

### Defining Classes
The KG structure is based on two main classes:
- **Disease**: Stores information on individual diseases such as names, classifications, and symptoms.
- **Gene**: Represents genes with details on their functions and chromosomal locations.

### Specifying Properties
Properties help define class attributes and the relationships between them, including:
- **Datatype Properties**: Describe disease and gene attributes such as risk factors and gene family.
- **Object Property**: `isAssociatedWith` links genes directly to diseases, crucial for analyzing their associations.

### Instantiating Individuals
We populate the KG by creating instances of diseases and genes, assigning relevant properties, and establishing relationships. The focus is on using these relationships to perform **Knowledge Graph Completion**, aiming to predict missing links between genes and diseases:

<p align="center">
  <img src="https://github.com/Lucasvitoriano25/Unveiling-Gene-Disease-Connections/assets/52925699/1d0770f4-9905-4ed1-ace9-a05ed093bcc5" alt="Graph" height="800px" />
</p>
The image depicts several instances from our knowledge graph, illustrating the critical challenge we address in this work. Specifically, we focus on utilizing the existing relations (highlighted in orange) and attributes within our incomplete knowledge graph to predict possible missing relationships (indicated in pink) between genes and diseases.

### Graph Neural Networks (GNN)

Graph Neural Networks (GNNs) represent a cutting-edge class of deep learning models designed to perform inference on data structured as graphs. Unlike traditional neural networks, GNNs can capture the complex relationships and interdependencies between nodes (e.g., genes, diseases) within a graph, making them particularly suited for analyzing biological networks. By leveraging node features and edge information, GNNs can learn to predict not only the properties of individual nodes but also the strength and significance of the connections between them. This capability is important for our work, as it allows us to model the intricate gene-disease associations within the Knowledge Graph, potentially uncovering novel insights and predictive markers for diseases.

<p align="center">
  <img src="https://github.com/user-attachments/assets/2a112039-d4a3-474d-b955-c907e90289f5" alt="GNN" height="350px" />
</p>


## Preprocessing the Dataset

#### Disease Feature Standardization
- Harmonized disease identifiers across datasets with differing IDs and naming conventions.
- Excluded diseases lacking comprehensive information on definitions, categories, and synonyms.

#### Disease Feature Augmentation with LLMs

We leveraged the capabilities of OpenAI's GPT 3.5 Turbo [2] to generate contextualized information on diseases. This process allowed us to enrich our dataset by extracting additional features in JSON format, such as main symptoms, risk factors, disease classes, and main systems affected. By relying on existing definitions to guide the generation process, we aimed to ensure the accuracy of the information and minimize the risk of generating incorrect data. The augmentation prompt was:

```text
The {disease} has the following definition: {definition}
Based on your available knowledge and in the definition provided, give me information about {disease} only in JSON format on:
{
  "main symptom": "",
  "risk factors": "",
  "disease class": "",
  "main system affected": ""
}
Go straight to the point: only list the important terms in the JSON format above.
```

After this preprocessing with GPT 3.5 Turbo[2] to enrich the dataset with contextualized disease information, we utilize BioClinicalBERT[3] to generate embeddings for the text data. These embeddings, which encode complex biomedical information into numerical form, serve as input features for the Graph Neural Network (GNN). By transforming rich textual descriptions into a format understandable by machine learning models, BioClinicalBERT[3] ensures that the GNN can effectively integrate and leverage semantic insights from the disease and gene descriptions. This step is crucial for enhancing the model's ability to discern and predict nuanced relationships within the biomedical data, contributing to the accuracy of the link predictions made by the GNN.

#### Gene Feature Enhancement

We addressed significant challenges in analyzing the "location" features of genes (position of the gene in the human chromosome sequence)) by applying regular expressions to dissect and transform the data into detailed components like Start Chromosome, Start Chromosome Arm, and specific locational markers. This detailed breakdown not only enhanced our data granularity but also enabled a more profound exploration of gene distribution across chromosomes. It significantly improved our ability to analyze and predict potential associations between specific genes and diseases, providing valuable insights into the genetic factors influencing disease mechanisms. Gene Location Data Transformation:


<div align="center">

| Feature          | Original Value | Refined Value |
|------------------|----------------|---------------|
| Start Chromosome | 5q14.2-q14.3   | 5             |
| Start Arm        | 5q14.2-q14.3   | q             |
| Start Loc        | 5q14.2-q14.3   | 14            |
| Start SubLoc     | 5q14.2-q14.3   | 2             |
| End Arm          | 5q14.2-q14.3   | q             |
| End Loc          | 5q14.2-q14.3   | 14            |
| End SubLoc       | 5q14.2-q14.3   | 3             |

</div>

## Implementation Details

### Model Architecture Selection: A Comparative Overview

We explored various Graph Neural Network (GNN) models to identify the most effective architecture for our dataset. Here's a summary of the architectures considered:

- **SAGEConv (GraphSAGE)**: Utilizes a sampling-based approach for neighbor aggregation, which is effective in large graphs by reducing computational load and memory usage, although it risks missing important neighbors which can affect learning outcomes.

- **GCNConv (Graph Convolutional Network)**: Employs a spectral method for neighbor information aggregation, offering good generalization across different graph structures but may face challenges with highly irregular connectivity patterns.

- **GATv2Conv (Graph Attention Network V2)**: Improves upon the original GAT model by using dynamic attention mechanisms to adjust the model’s focus based on graph structure, enhancing the ability to capture complex node interdependencies at the cost of increased computational demand.

These models were carefully evaluated to understand their efficacy in encoding the complex relationships in gene-disease interaction data, aiming to maximize both predictive accuracy and model interpretability.

### Training Pipeline Overview

The training pipeline begins with the extraction and enhancement of disease and gene features from the Stanford Bio-Database using natural language models like ChatGPT and BioClinicalBERT. These models preprocess the raw data, enriching it with additional medical insights and preparing it for integration into a Knowledge Graph (KG). This preprocessing step is critical as it ensures the data is normalized and enriched with relevant biomedical context, which enhances the accuracy and effectiveness of subsequent analyses.

<p align="center">
  <img src="https://github.com/user-attachments/assets/692346bd-aa43-44e8-8633-2cbf9c0560dd" alt="Pipeline" height="550px" />
</p>

After preprocessing, the enriched data feeds into a Graph Neural Network (GNN), which models the complex relationships between genes and diseases as a network of nodes (representing genes and diseases) and edges (representing their associations). Although not originally designed for biological data, the GNN effectively adapts to such datasets, leveraging its capacity to learn from the graph structure. This capability allows the GNN to perform deep learning-based predictions of potential links, known as link prediction. This process not only predicts new, plausible gene-disease associations but also enhances our understanding of the underlying biological processes, contributing to research in genomic medicine and personalized treatments.


## References

[1] Sagar Maheshwari Marinka Zitnik, Rok Sosic and Jure Leskovec. BioSNAP Datasets: Stanford biomedical network dataset collection. [http://snap.stanford.edu/biodata](http://snap.stanford.edu/biodata), August 2018

[2] OpenAI. 2022. "Introducing ChatGPT." Accessed 2023. [https://openai.com/blog/chatgpt](https://openai.com/blog/chatgpt).

[3] Emily Alsentzer, John R. Murphy, Willie Boag, Wei-Hung Weng, Di Jin, Tristan Naumann, and Matthew B. A. McDermott. 2019. "Publicly Available Clinical BERT Embeddings." Accessed [Year]. [https://arxiv.org/abs/1904.03323](https://arxiv.org/abs/1904.03323).



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
