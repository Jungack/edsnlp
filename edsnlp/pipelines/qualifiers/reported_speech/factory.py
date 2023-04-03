from typing import List, Optional

from edsnlp.core import PipelineProtocol, registry
from edsnlp.utils.deprecation import deprecated_factory
from .reported_speech import ReportedSpeech

DEFAULT_CONFIG = dict(
    pseudo=None,
    preceding=None,
    following=None,
    quotation=None,
    verbs=None,
    attr="NORM",
    on_ents_only=True,
    within_ents=False,
    explain=False,
)


@deprecated_factory(
    "rspeech",
    "eds.reported_speech",
    default_config=DEFAULT_CONFIG,
    assigns=["span._.reported_speech"],
)
@deprecated_factory(
    "reported_speech",
    "eds.reported_speech",
    default_config=DEFAULT_CONFIG,
    assigns=["span._.reported_speech"],
)
@registry.factory.register(
    "eds.reported_speech",
    default_config=DEFAULT_CONFIG,
    assigns=["span._.reported_speech"],
)
def create_component(
    nlp: PipelineProtocol,
    name: str,
    attr: str,
    pseudo: Optional[List[str]],
    preceding: Optional[List[str]],
    following: Optional[List[str]],
    quotation: Optional[List[str]],
    verbs: Optional[List[str]],
    on_ents_only: bool,
    within_ents: bool,
    explain: bool,
):
    return ReportedSpeech(
        nlp=nlp,
        attr=attr,
        pseudo=pseudo,
        preceding=preceding,
        following=following,
        quotation=quotation,
        verbs=verbs,
        on_ents_only=on_ents_only,
        within_ents=within_ents,
        explain=explain,
    )
