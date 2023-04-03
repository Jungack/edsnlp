from typing import List, Optional, Union

from edsnlp.core import PipelineProtocol, registry
from .tnm import TNM

DEFAULT_CONFIG = dict(
    pattern=None,
    attr="TEXT",
)


@registry.factory.register(
    "eds.TNM",
    default_config=DEFAULT_CONFIG,
    assigns=["doc.ents", "doc.spans"],
)
def create_component(
    nlp: PipelineProtocol,
    name: str,
    pattern: Optional[Union[List[str], str]],
    attr: str,
):
    return TNM(
        nlp,
        pattern=pattern,
        attr=attr,
    )
