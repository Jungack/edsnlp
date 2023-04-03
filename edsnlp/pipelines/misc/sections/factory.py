from typing import Dict, List, Optional

from edsnlp.core import PipelineProtocol, registry
from edsnlp.utils.deprecation import deprecated_factory
from . import Sections

DEFAULT_CONFIG = dict(
    sections=None,
    add_patterns=True,
    attr="NORM",
    ignore_excluded=True,
)


@deprecated_factory("sections", "eds.sections", default_config=DEFAULT_CONFIG)
@registry.factory.register("eds.sections", default_config=DEFAULT_CONFIG)
def create_component(
    nlp: PipelineProtocol,
    name: str,
    sections: Optional[Dict[str, List[str]]],
    add_patterns: bool,
    attr: str,
    ignore_excluded: bool,
):
    return Sections(
        nlp,
        sections=sections,
        add_patterns=add_patterns,
        attr=attr,
        ignore_excluded=ignore_excluded,
    )
