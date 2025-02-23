from _typeshed import Incomplete
from numba.core import types as types
from numba.core.datamodel.registry import DataModelManager as DataModelManager, register as register
from numba.core.extending import models as models
from numba.cuda.types import CUDADispatcher as CUDADispatcher, Dim3 as Dim3, GridGroup as GridGroup

cuda_data_manager: Incomplete
register_model: Incomplete

class Dim3Model(models.StructModel):
    def __init__(self, dmm, fe_type) -> None: ...

class GridGroupModel(models.PrimitiveModel):
    def __init__(self, dmm, fe_type) -> None: ...

class FloatModel(models.PrimitiveModel):
    def __init__(self, dmm, fe_type) -> None: ...
