import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, to_hetero

class GNNEncoder(nn.Module):
  def __init__(
      self,
      num_blocks: int = 1,
      activation:nn = nn.ReLU,
      RGNN:nn = GATv2Conv,
      hidden_layer:int = 512,
      heads:int = 4,
      attention_based: bool = False,
      dropout:float = 0.2
    ):
    """
    Encodes node features using multiple graph neural network blocks.

    This class constructs a multi-layer graph neural network (GNN) encoder, where each layer consists of a GNN layer
    followed by batch normalization and an activation function. It supports various GNN architectures and activations.

    **Parameters:**

    - num_blocks (int, optional): The number of GNN blocks to stack. Defaults to 1.
    - activation (nn.Module, optional): The activation function to apply after each block. Defaults to nn.PReLU.
    - RGNN (nn.Module, optional): The type of GNN layer to use. Defaults to GATv2Conv.
    - hidden_layer (int, optional): The dimensionality of the hidden layer in each GNN block. Defaults to 64.
    - heads (int, optional): The number of attention heads for GAT-based GNN layers. Defaults to 4.
    - dropout (float, optional): The dropout probability for regularization. Defaults to 0.2.

    **Returns:**

    - torch.Tensor: The encoded node features, with shape (num_nodes, hidden_layer * heads).
    """

    super().__init__()

    self.num_blocks = num_blocks
    self.gnn_layers = nn.ModuleList()
    self.norm_layers = nn.ModuleList()
    self.activation = activation()
    self.dropout = nn.Dropout(dropout)
    self.attention_based = attention_based
    print(self.attention_based)
    for _ in range(self.num_blocks):
      if self.attention_based:
        self.gnn_layers.append(RGNN((-1, -1),hidden_layer,heads = heads, add_self_loops = False))
        self.norm_layers.append(nn.BatchNorm1d(hidden_layer*heads))
      else:
        self.gnn_layers.append(RGNN((-1, -1),hidden_layer))
        self.norm_layers.append(nn.BatchNorm1d(hidden_layer))
      #self.gnn_layers.append(SAGEConv((-1, -1), out_channels=3072))

  def forward(self, nodes_features, edge_index):

    for gnn,norm in zip(self.gnn_layers,self.norm_layers):
      nodes_features = gnn(nodes_features,edge_index)
      nodes_features = norm(nodes_features)
      nodes_features = self.activation(nodes_features)
      nodes_features = self.dropout(nodes_features)
    return nodes_features

class GNN_FeedFoward(nn.Module):
  def __init__(
        self,
        data_sample,
        num_blocks: int = 2,
        activation:nn = nn.ReLU,
        RGNN:nn = GATv2Conv,
        dropout:float = 0.1,
        hidden_layer:int = 128,
        heads:int = 4,
        num_hidden_layers: int = 3,
        attention_based: bool = False
    ):

    """
    Constructs a graph neural network for link prediction in heterogeneous graphs.

    This class combines a multi-layer GNN encoder for node feature learning with a feedforward network
    to predict link existence scores. It's designed for heterogeneous graphs where nodes have different types.

    **Parameters:**

    - data_sample (PyG Data object): A sample of the heterogeneous graph data used for metadata extraction.
    - num_blocks (int, optional): The number of GNN blocks in the encoder. Defaults to 2.
    - activation (nn.Module, optional): The activation function used in GNN blocks and feedforward layers. Defaults to nn.PReLU.
    - RGNN (nn.Module, optional): The type of GNN layer to use. Defaults to GATv2Conv.
    - dropout (float, optional): The dropout probability for regularization. Defaults to 0.2.
    - hidden_layer (int, optional): The dimensionality of the hidden layer in GNN blocks. Defaults to 64.
    - heads (int, optional): The number of attention heads for GAT-based GNN layers. Defaults to 4.
    - num_hidden_layers (int, optional): The number of hidden layers in the feedforward network. Defaults to 3.
    """

    super().__init__()
    if attention_based:
        self.encoder = GNNEncoder(
            num_blocks=num_blocks,
            activation=activation,
            RGNN=RGNN,
            hidden_layer=hidden_layer,
            heads=heads,
            attention_based=True,
            dropout=dropout
        )
        encoder_output_channels_per_node = hidden_layer * heads
    else:
        self.encoder = GNNEncoder(
            num_blocks=num_blocks,
            activation=activation,
            RGNN=RGNN,
            hidden_layer=hidden_layer,
            heads=1,  
            attention_based=False,
            dropout=dropout
        )
        encoder_output_channels_per_node = hidden_layer
    self.encoder = to_hetero(self.encoder, data_sample.metadata(), aggr='mean')

    self.feedforward = nn.ModuleList()
    self.final_activation_function = nn.Sigmoid()
    self.feedforward_input_size = 2 * encoder_output_channels_per_node
    self.feedforward_hidden_size = self.feedforward_input_size
    # add input layer
    self.feedforward.append(nn.Linear(self.feedforward_input_size,self.feedforward_hidden_size))
    self.feedforward.append(activation())
    self.feedforward.append(nn.Dropout(dropout))

    # add hidden layers
    for _ in range(num_hidden_layers):
      self.feedforward.append(nn.Linear(self.feedforward_hidden_size, self.feedforward_hidden_size))
      self.feedforward.append(activation())
      self.feedforward.append(nn.Dropout(dropout))

    # add output layer
    self.feedforward.append(nn.Linear(self.feedforward_hidden_size,1))
    # self.feedforward.append(self.final_activation_function)

  def forward(self, nodes_features, edge_index, edge_label_index):
    nodes_embs = self.encoder(nodes_features, edge_index)
    gene_embs = nodes_embs['gene'][edge_label_index[1]]
    disease_embs = nodes_embs['disease'][edge_label_index[0]]
    concat = torch.cat((gene_embs,disease_embs), dim=1)
    for layer in self.feedforward:
      concat = layer(concat)
    return concat.view(-1)

class GNN_dot_product(nn.Module):
  """
  Constructs a graph neural network for link prediction in heterogeneous graphs using a dot product approach.

  This class combines a multi-layer GNN encoder for node feature learning with a dot product operation
  to predict link existence scores. It's designed for heterogeneous graphs where nodes have different types.

  **Parameters:**

  - data_sample (PyG Data object): A sample of the heterogeneous graph data used for metadata extraction.
  - num_blocks (int, optional): The number of GNN blocks in the encoder. Defaults to 2.
  - activation (nn.Module, optional): The activation function used in GNN blocks (defaults to nn.ReLU).
  - RGNN (nn.Module, optional): The type of GNN layer to use (defaults to GATv2Conv).
  - dropout (float, optional): The dropout probability for regularization (defaults to 0.1).
  - hidden_layer (int, optional): The dimensionality of the hidden layer in GNN blocks (defaults to 512).
  - heads (int, optional): The number of attention heads for GAT-based GNN layers (defaults to 12).
  """

  def __init__(self, data_sample, num_blocks=1, activation=nn.ReLU, RGNN=GATv2Conv, dropout=0.2, hidden_layer=256, heads=8, attention_based: bool = False):
    super().__init__()

    self.encoder = GNNEncoder(num_blocks = num_blocks, activation = activation, RGNN = RGNN, hidden_layer = hidden_layer, heads = heads, attention_based = attention_based,dropout = dropout)
    self.encoder = to_hetero(self.encoder, data_sample.metadata(), aggr='mean')

  def forward(self, nodes_features, edge_index, edge_label_index):

    nodes_embs = self.encoder(nodes_features, edge_index)

    gene_embs = nodes_embs['gene'][edge_label_index[1]]
    disease_embs = nodes_embs['disease'][edge_label_index[0]]

    # Dot product for link prediction
    scores = torch.sum(gene_embs * disease_embs, dim=1)  # Element-wise multiplication and summation

    return scores
