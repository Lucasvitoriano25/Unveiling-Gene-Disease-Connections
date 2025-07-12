import torch
import torch.nn as nn
# Ensure you have the specific GNN layers you plan to use imported
from torch_geometric.nn import GATv2Conv, GraphConv, SAGEConv # Add others if needed
# to_hetero will be removed from the model definitions below

class GNNEncoder(nn.Module):
  def __init__(
      self,
      num_blocks: int = 1,
      activation: nn.Module = nn.ReLU, # Pass class, instantiate inside
      RGNN: nn.Module = GATv2Conv,    # Pass class
      hidden_layer: int = 512,
      heads: int = 4,
      attention_based: bool = False,
      dropout: float = 0.2
    ):
    """
    Encodes node features using multiple graph neural network blocks for homogeneous graphs.
    (Docstring updated for clarity)
    """
    super().__init__()

    self.num_blocks = num_blocks
    self.gnn_layers = nn.ModuleList()
    self.norm_layers = nn.ModuleList()
    self.activation = activation() # Instantiate activation function
    self.dropout = nn.Dropout(dropout)
    self.attention_based = attention_based

    current_in_channels = -1 # For lazy initialization of the first layer

    for i in range(self.num_blocks):
      if self.attention_based:
        # For attention layers like GATv2Conv, output is hidden_layer * heads
        # The input to the first layer is -1 (lazy), subsequent layers take hidden_layer * heads
        # PyG GATv2Conv signature: GATv2Conv(in_channels, out_channels, heads, ...)
        # Here, 'hidden_layer' is treated as out_channels_per_head
        self.gnn_layers.append(RGNN(current_in_channels, hidden_layer, heads=heads, add_self_loops=False))
        current_out_channels = hidden_layer * heads
        self.norm_layers.append(nn.BatchNorm1d(current_out_channels))
      else:
        # For non-attention layers like GraphConv, output is hidden_layer
        # PyG GraphConv signature: GraphConv(in_channels, out_channels, ...)
        self.gnn_layers.append(RGNN(current_in_channels, hidden_layer))
        current_out_channels = hidden_layer
        self.norm_layers.append(nn.BatchNorm1d(current_out_channels))
      
      current_in_channels = current_out_channels # Input for the next block

  def forward(self, x, edge_index): # Changed 'nodes_features' to 'x' for clarity
    """
    Forward pass for the GNNEncoder.
    x: Node feature tensor for the homogeneous graph [num_nodes, in_features]
    edge_index: Edge index tensor for the homogeneous graph [2, num_edges]
    """
    for gnn, norm in zip(self.gnn_layers, self.norm_layers):
      x = gnn(x, edge_index)
      x = norm(x)
      x = self.activation(x)
      x = self.dropout(x)
    return x

class GNN_FeedFoward(nn.Module): # Renamed for clarity
  def __init__(
        self,
        # data_sample removed
        in_channels: int, # Add in_channels for the first GNN layer if GNNEncoder doesn't use lazy init for it
                        # OR ensure GNNEncoder uses lazy init (-1) for its first layer.
                        # Based on your GNNEncoder, it uses lazy init, so in_channels isn't strictly
                        # needed here for GNNEncoder itself, but good for knowing input dim.
        num_blocks: int = 1, # Default from your example instantiation
        activation: nn.Module = nn.ReLU, # Pass class
        RGNN: nn.Module = GATv2Conv,    # Pass class
        dropout: float = 0.1,
        hidden_layer: int = 128,
        heads: int = 4,
        num_hidden_layers: int = 0, # Default from your example instantiation
        attention_based: bool = False
    ):
    super().__init__()
    
    self.is_encoder_attention_based = attention_based # Store the mode

    self.encoder = GNNEncoder(
        num_blocks=num_blocks,
        activation=activation, # Pass class
        RGNN=RGNN,             # Pass class
        hidden_layer=hidden_layer,
        heads=heads,
        attention_based=self.is_encoder_attention_based, # Pass to GNNEncoder
        dropout=dropout
    )
    # `to_hetero` is removed

    self.feedforward = nn.ModuleList()
    # self.final_activation_function = nn.Sigmoid() # Keep commented if using BCEWithLogitsLoss

    if self.is_encoder_attention_based:
        encoder_output_channels_per_node = hidden_layer * heads
    else:
        encoder_output_channels_per_node = hidden_layer
            
    self.feedforward_input_size = 2 * encoder_output_channels_per_node
    self.feedforward_hidden_size = self.feedforward_input_size 

    self.feedforward.append(nn.Linear(self.feedforward_input_size, self.feedforward_hidden_size))
    self.feedforward.append(activation()) # Instantiate activation
    self.feedforward.append(nn.Dropout(dropout))

    for _ in range(num_hidden_layers):
      self.feedforward.append(nn.Linear(self.feedforward_hidden_size, self.feedforward_hidden_size))
      self.feedforward.append(activation()) # Instantiate activation
      self.feedforward.append(nn.Dropout(dropout))

    self.feedforward.append(nn.Linear(self.feedforward_hidden_size, 1))

  def forward(self, x, edge_index, edge_label_index): # Changed parameters
    # x: Combined node features [total_num_nodes, num_features]
    # edge_index: Graph connectivity [2, num_edges]
    # edge_label_index: Links to predict [2, num_target_links]
    
    nodes_embs = self.encoder(x, edge_index) # Returns [total_num_nodes, encoder_output_channels_per_node]
    
    # edge_label_index contains global indices for the homogeneous graph
    # Assuming edge_label_index[0] are source nodes, edge_label_index[1] are target nodes
    source_embs = nodes_embs[edge_label_index[0]]
    target_embs = nodes_embs[edge_label_index[1]]
    
    concat = torch.cat((source_embs, target_embs), dim=1)
    for layer in self.feedforward:
      concat = layer(concat)
    return concat.view(-1)

class GNN_dot_product(nn.Module): # Renamed for clarity
  def __init__(self,
               # data_sample removed
               in_channels: int, # Add in_channels (similar to GNN_FeedFoward_Homo)
               num_blocks: int = 1, 
               activation: nn.Module = nn.ReLU, # Pass class
               RGNN: nn.Module = GATv2Conv,    # Pass class
               dropout: float = 0.2, 
               hidden_layer: int = 256, 
               heads: int = 8, 
               attention_based: bool = False):
    super().__init__()

    self.is_encoder_attention_based = attention_based # Store mode

    self.encoder = GNNEncoder(
        num_blocks=num_blocks, 
        activation=activation, # Pass class
        RGNN=RGNN,             # Pass class
        hidden_layer=hidden_layer, 
        heads=heads, 
        attention_based=self.is_encoder_attention_based, 
        dropout=dropout
    )
    # `to_hetero` is removed

  def forward(self, x, edge_index, edge_label_index): # Changed parameters
    nodes_embs = self.encoder(x, edge_index)

    source_embs = nodes_embs[edge_label_index[0]]
    target_embs = nodes_embs[edge_label_index[1]]

    scores = torch.sum(source_embs * target_embs, dim=1)
    return scores