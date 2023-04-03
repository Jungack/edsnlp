from typing import List

from edsnlp.core import PipelineProtocol, registry
from .context import ContextAdder

DEFAULT_CONFIG = dict(
    context=["note_id"],
)


@registry.factory.register(
    "eds.context",
    default_config=DEFAULT_CONFIG,
)
def create_component(
    nlp: PipelineProtocol,
    name: str,
    context: List[str],
):

    return ContextAdder(
        nlp,
        context=context,
    )
