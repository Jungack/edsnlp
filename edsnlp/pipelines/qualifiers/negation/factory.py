from typing import List, Optional

from edsnlp.core import PipelineProtocol, registry
from edsnlp.utils.deprecation import deprecated_factory
from .negation import Negation

DEFAULT_CONFIG = dict(
    pseudo=None,
    preceding=None,
    following=None,
    termination=None,
    verbs=None,
    attr="NORM",
    on_ents_only=True,
    within_ents=False,
    explain=False,
)


@deprecated_factory(
    "negation",
    "eds.negation",
    default_config=DEFAULT_CONFIG,
    assigns=["span._.negation"],
)
@registry.factory.register(
    "eds.negation",
    default_config=DEFAULT_CONFIG,
    assigns=["span._.negation"],
)
def create_component(
    nlp: PipelineProtocol,
    name: str,
    attr: str,
    pseudo: Optional[List[str]],
    preceding: Optional[List[str]],
    following: Optional[List[str]],
    termination: Optional[List[str]],
    verbs: Optional[List[str]],
    on_ents_only: bool,
    within_ents: bool,
    explain: bool,
):

    return Negation(
        nlp=nlp,
        attr=attr,
        pseudo=pseudo,
        preceding=preceding,
        following=following,
        termination=termination,
        verbs=verbs,
        on_ents_only=on_ents_only,
        within_ents=within_ents,
        explain=explain,
    )
