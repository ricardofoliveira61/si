from typing import Any
from si.base.model import Model
from si.base.transformer import Transformer
from si.statistics import euclidean_distance


class Kmeans(Transformer,Model):

    def __init__(self,k:int,max_iter:int=1000,distance:callable=euclidean_distance, **kwargs) -> Any:
        super().__init__(**kwargs)