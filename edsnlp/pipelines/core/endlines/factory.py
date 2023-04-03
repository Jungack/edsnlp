from typing import Optional

from edsnlp.core import PipelineProtocol, registry
from edsnlp.utils.deprecation import deprecated_factory
from .endlines import EndLines


@deprecated_factory("endlines", "eds.endlines")
@registry.factory.register("eds.endlines")
def create_component(
    nlp: PipelineProtocol,
    name: str,
    model_path: Optional[str],
):
    return EndLines(nlp, end_lines_model=model_path)
