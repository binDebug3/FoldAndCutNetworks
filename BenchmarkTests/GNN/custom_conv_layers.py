from torch_geometric.nn.conv import GCNConv, SAGEConv, GraphConv, GATConv, GATv2Conv, TransformerConv
from typing import Any, Callable, Dict, Final, List, Optional, Tuple, Union
from BenchmarkTests.experimenter import get_model
from models.folds import SoftFold
import torch.nn as nn

class GCNFoldConv(GCNConv) :
    def __init__(
        self,
        in_channels: int,
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
                nn.Linear(in_channels, out_channels),
                nn.ReLU(),
                SoftFold(out_channels, crease=crease, has_stretch=has_stretch)
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
