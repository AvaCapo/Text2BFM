from huggingface_hub import PyTorchModelHubMixin

from src.generator import GeneratorFBcprModelVAE as BaseGeneratorFBcprModelVAE
from src.mdm import MDMFBcprModel as BaseMDMFBcprModel


class GeneratorFBcprModelVAE(
    BaseGeneratorFBcprModelVAE,
    PyTorchModelHubMixin,
    library_name="metamotivo",
    tags=["facebook", "meta", "pytorch"],
    license="cc-by-nc-4.0",
    repo_url="https://github.com/facebookresearch/metamotivo",
    docs_url="https://metamotivo.metademolab.com/",
):
    ...


class MDMFBcprModel(
    BaseMDMFBcprModel,
    PyTorchModelHubMixin,
    library_name="metamotivo",
    tags=["facebook", "meta", "pytorch"],
    license="cc-by-nc-4.0",
    repo_url="https://github.com/facebookresearch/metamotivo",
    docs_url="https://metamotivo.metademolab.com/",
): ...

__all__ = ["GeneratorFBcprModelVAE"]
