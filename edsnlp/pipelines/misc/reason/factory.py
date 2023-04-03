from typing import Dict, List, Optional, Union

from edsnlp.core import PipelineProtocol, registry
from edsnlp.utils.deprecation import deprecated_factory
from .reason import Reason
from . import patterns

DEFAULT_CONFIG = dict(
    reasons=None,
    attr="TEXT",
    use_sections=False,
    ignore_excluded=False,
    reason_sections=None,
    excluded_sections=None,
)


@deprecated_factory("reason", "eds.reason", default_config=DEFAULT_CONFIG)
@registry.factory.register("eds.reason", default_config=DEFAULT_CONFIG)
def create_component(
    nlp: PipelineProtocol,
    name: str,
    reasons: Optional[Dict[str, Union[List[str], str]]],
    reason_sections: Optional[List[str]],
    excluded_sections: Optional[List[str]],
    attr: str,
    use_sections: bool,
    ignore_excluded: bool,
):
    if reasons is None:
        reasons = patterns.reasons
    if reason_sections is None:
        reason_sections = patterns.reason_sections
    if excluded_sections is None:
        excluded_sections = patterns.excluded_sections
    return Reason(
        nlp,
        reasons=reasons,
        attr=attr,
        use_sections=use_sections,
        ignore_excluded=ignore_excluded,
        reason_sections=reason_sections,
        excluded_sections=excluded_sections,
    )
