from BenchmarkTests.GNN.custom_conv_layers import GCNFoldConv, SAGEFoldConv, GATFoldConv, GATv2FoldConv
from torch_geometric.nn.models import GCN, GraphSAGE, GAT, GIN
from torch_geometric.nn.conv import MessagePassing, GINConv
from typing import Any, Callable, Dict, Final, List, Optional, Tuple, Union
from torch_geometric.nn.pool import global_mean_pool, global_max_pool
from BenchmarkTests.experimenter import get_model
import torch.nn as nn

class FoldGCN(GCN) :
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        num_folds: int,
        dim_increase: List[float],
        has_stretch: bool,
        out_channels: Optional[int] = None,
        crease: Optional[float] = None,
        dropout: float = 0.0,
        act: Union[str, Callable, None] = "relu",
        act_first: bool = False,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        jk: Optional[str] = None,
        **kwargs,
    ):
        super.__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=out_channels,
            dropout=dropout,
            act=act,
            act_first=act_first,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
            jk=jk,
            **kwargs
        )
        self.num_folds = num_folds
        self.dim_increase = dim_increase
        self.has_stretch = has_stretch
        self.crease = crease
    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        return GCNFoldConv(in_channels, out_channels, self.num_folds, 
                           self.dim_increase, self.has_stretch, 
                           crease=self.crease, **kwargs)




class FoldGraphSAGE(GraphSAGE):
    def init_conv(self, in_channels: Union[int, Tuple[int, int]],
                  out_channels: int, **kwargs) -> MessagePassing:
        return SAGEFoldConv(in_channels, out_channels, **kwargs)
    
class FoldGIN(GIN):
    def __init__(
        self,
        model_name: str,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        act: Union[str, Callable, None] = "relu",
        act_first: bool = False,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        jk: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=out_channels,
            dropout=dropout,
            act=act,
            act_first=act_first,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
            jk=jk,
            **kwargs
        )
        self.model_name = model_name

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        fold_and_cut_net, lr = get_model(self.model_name, input_size=in_channels, output_size=out_channels)
        return GINConv(fold_and_cut_net, **kwargs)
    
class GINNetwork(nn.Module) :
    def __init__(self, in_channels, hidden_channels, num_layers, num_classes, graph_level_task:bool):
        super(GINNetwork, self).__init__()
        if graph_level_task :
            self.gin = GIN(in_channels, hidden_channels, num_layers)
            self.fc = nn.Linear(hidden_channels, num_classes)
        else : 
            self.gin = GIN(in_channels, hidden_channels, num_layers, out_channels=num_classes)
        self.graph_level_task = graph_level_task
    
    def forward(self, batch) :
        x = self.gin(batch.x, batch.edge_index)
        if self.graph_level_task :
            x = global_max_pool(x, batch.batch)
            x = self.fc(x)
        return x
    
    def reset_parameters(self) :
        self.gin.reset_parameters()

class FoldGINNetwork(nn.Module) :
    def __init__(self, model_name, in_channels, hidden_channels, num_layers, num_classes, graph_level_task:bool):
        super(FoldGINNetwork, self).__init__()
        if graph_level_task :
            self.gin = FoldGIN(model_name, in_channels, hidden_channels, num_layers)
            self.fc = nn.Linear(hidden_channels, num_classes)
        else : 
            self.gin = FoldGIN(model_name, in_channels, hidden_channels, num_layers, out_channels=num_classes)
        self.graph_level_task = graph_level_task
    
    def forward(self, batch) :
        x = self.gin(batch.x, batch.edge_index)
        if self.graph_level_task :
            x = global_max_pool(x, batch.batch)
            x = self.fc(x)
        return x

    
class FoldGAT(GAT) :
    def init_conv(self, in_channels: Union[int, Tuple[int, int]],
                  out_channels: int, **kwargs) -> MessagePassing:

        v2 = kwargs.pop('v2', False)
        heads = kwargs.pop('heads', 1)
        concat = kwargs.pop('concat', True)

        # Do not use concatenation in case the layer `GATConv` layer maps to
        # the desired output channels (out_channels != None and jk != None):
        if getattr(self, '_is_conv_to_out', False):
            concat = False

        if concat and out_channels % heads != 0:
            raise ValueError(f"Ensure that the number of output channels of "
                             f"'GATConv' (got '{out_channels}') is divisible "
                             f"by the number of heads (got '{heads}')")

        if concat:
            out_channels = out_channels // heads

        Conv = GATFoldConv if not v2 else GATv2FoldConv
        return Conv(in_channels, out_channels, heads=heads, concat=concat,
                    dropout=self.dropout.p, **kwargs) 