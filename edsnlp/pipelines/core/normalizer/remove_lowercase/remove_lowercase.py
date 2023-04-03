from typing import Callable

from spacy.tokens import Doc

from edsnlp.core import registry, PipelineProtocol
from edsnlp.utils.deprecation import deprecated_factory
from edsnlp.utils.factories import get_default_config


def remove_lowercase(doc: Doc):
    for token in doc:
        token.norm_ = token.text
    return doc


@deprecated_factory("remove_lowercase", "eds.remove_lowercase", assigns=["token.norm"])
@registry.factory.register("eds.remove_lowercase", assigns=["token.norm"])
def create_component(nlp: PipelineProtocol, name: str) -> Callable:
    """
    Reverts the norm_ attribute of tokens to the text attribute.

    Parameters
    ----------
    nlp : PipelineProtocol
        EDS-NLP pipeline object
    name : str
        Name of the component

    Returns
    -------
    Callable
    """
    return remove_lowercase


RemoveLowercase = create_component
DEFAULT_CONFIG = get_default_config(create_component)
