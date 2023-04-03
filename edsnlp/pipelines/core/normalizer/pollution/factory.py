from typing import Dict, List, Optional, Union

from edsnlp.core import PipelineProtocol, registry
from edsnlp.utils.deprecation import deprecated_factory
from . import Pollution

DEFAULT_CONFIG = dict(
    pollution=dict(
        information=True,
        bars=True,
        biology=False,
        doctors=True,
        web=True,
        coding=False,
        footer=True,
    ),
)


@deprecated_factory(
    "pollution", "eds.pollution", default_config=DEFAULT_CONFIG, assigns=["token.tag"]
)
@registry.factory.register(
    "eds.pollution", default_config=DEFAULT_CONFIG, assigns=["token.tag"]
)
def create_component(
    nlp: PipelineProtocol,
    name: str,
    pollution: Optional[Dict[str, Union[bool, str, List[str]]]],
):
    return Pollution(
        nlp,
        pollution=pollution,
    )
