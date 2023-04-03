from typing import Dict, List, Optional, Union

from edsnlp.core import PipelineProtocol, registry
from edsnlp.utils.deprecation import deprecated_factory
from .reason import Reason

DEFAULT_CONFIG = dict(
    reasons=None,
    attr="TEXT",
    use_sections=False,
    ignore_excluded=False,
)


@deprecated_factory("reason", "eds.reason", default_config=DEFAULT_CONFIG)
@registry.factory.register("eds.reason", default_config=DEFAULT_CONFIG)
def create_component(
    nlp: PipelineProtocol,
    name: str,
    reasons: Optional[Dict[str, Union[List[str], str]]],
    attr: str,
    use_sections: bool,
    ignore_excluded: bool,
):
    return Reason(
        nlp,
        reasons=reasons,
        attr=attr,
        use_sections=use_sections,
        ignore_excluded=ignore_excluded,
    )
