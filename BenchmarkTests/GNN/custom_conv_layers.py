from torch_geometric.nn.conv import GCNConv, SAGEConv, GraphConv, GATConv, GATv2Conv, TransformerConv
from typing import Any, Callable, Dict, Final, List, Optional, Tuple, Union
from experimenter import get_model
from models.folds import SoftFold
import torch.nn as nn

class GCNFoldConv(GCNConv) :
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        out_channels: int,
        has_stretch: bool,
        crease: Optional[float] = None,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: Optional[bool] = None,
        normalize: bool = True,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            improved=improved,
            cached=cached,
            add_self_loops=add_self_loops,
            normalize=normalize,
            bias=bias,
            **kwargs
        )
        if out_channels >= in_channels :
            del self.lin
            self.lin = nn.Sequential(
                # Each "layer" will be a linear layer followed by a soft fold
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU(),
                SoftFold(hidden_channels, crease=crease, has_stretch=has_stretch)
            )
            

class SAGEFoldConv(SAGEConv):
    pass  

class GraphFoldConv(GraphConv):
    pass 

class GATFoldConv(GATConv):
    pass 

class GATv2FoldConv(GATv2Conv):
    pass

class TransformerFoldConv(TransformerConv) :
    pass 


# NNConv - pass in Origami Network instead of MLP
# same with GIN
