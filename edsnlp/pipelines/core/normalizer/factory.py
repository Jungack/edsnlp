from typing import Any, Dict, Union

from edsnlp.core import PipelineProtocol, registry
from edsnlp.utils.deprecation import deprecated_factory
from .accents.factory import (
    DEFAULT_CONFIG as accents_config,
    create_component as create_accents,
)
from .normalizer import Normalizer
from .pollution.factory import (
    DEFAULT_CONFIG as pollution_config,
    create_component as create_pollution,
)
from .quotes.factory import (
    DEFAULT_CONFIG as quotes_config,
    create_component as create_quotes,
)
from .remove_lowercase.factory import create_component as create_remove_lowercase
from .spaces.factory import (
    DEFAULT_CONFIG as spaces_config,
    create_component as create_spaces,
)

DEFAULT_CONFIG = dict(
    accents=True,
    lowercase=True,
    quotes=True,
    spaces=True,
    pollution=True,
)


@deprecated_factory(
    "normalizer",
    "eds.normalizer",
    default_config=DEFAULT_CONFIG,
    assigns=["token.norm", "token.tag"],
)
@registry.factory.register(
    "eds.normalizer", default_config=DEFAULT_CONFIG, assigns=["token.norm", "token.tag"]
)
def create_component(
    nlp: PipelineProtocol,
    name: str,
    lowercase: Union[bool, Dict[str, Any]] = True,
    accents: Union[bool, Dict[str, Any]] = True,
    quotes: Union[bool, Dict[str, Any]] = True,
    spaces: Union[bool, Dict[str, Any]] = True,
    pollution: Union[bool, Dict[str, Any], Dict[str, bool]] = True,
) -> Normalizer:
    """
    Normalisation pipeline. Modifies the `NORM` attribute,
    acting on five dimensions :

    - `lowercase`: using the `TEXT` attribute as a base for `NORM`.
    - `accents`: deterministic and fixed-length normalisation of accents.
    - `quotes`: deterministic and fixed-length normalisation of quotation marks.
    - `spaces`: "removal" of spaces tokens (via the tag_ attribute).
    - `pollution`: "removal" of pollutions (via the tag_ attribute).

    Parameters
    ----------
    lowercase : Union[bool, Dict[str, Any]]
        Whether to remove case.
    accents : Union[bool, Dict[str, Any]]
        `Accents` configuration object
    quotes : Union[bool, Dict[str, Any]]
        `Quotes` configuration object
    spaces : Union[bool, Dict[str, Any]]
        `Spaces` configuration object
    pollution : Union[bool, Dict[str, Any]]
        Optional `Pollution` configuration object.
    """

    remove_lowercase = None
    if not lowercase:
        remove_lowercase = create_remove_lowercase(nlp=nlp, name="eds.remove_lowercase")

    if accents:
        config = dict(**accents_config)
        if isinstance(accents, dict):
            config.update(accents)
        accents = create_accents(nlp=nlp, name="eds.accents", **config)

    if quotes:
        config = dict(**quotes_config)
        if isinstance(quotes, dict):
            config.update(quotes)
        quotes = create_quotes(nlp=nlp, name="eds.quotes", **config)

    if spaces:
        config = dict(**spaces_config)
        if isinstance(spaces, dict):
            config.update(spaces)
        spaces = create_spaces(nlp=nlp, name="eds.spaces", **config)

    if isinstance(pollution, dict) and pollution.get("pollution"):
        pollution = pollution["pollution"]
    if pollution:
        config = dict(**pollution_config["pollution"])
        if isinstance(pollution, dict):
            config.update(pollution)
        pollution = create_pollution(nlp=nlp, name="eds.pollution", pollution=config)

    normalizer = Normalizer(
        remove_lowercase=remove_lowercase or None,
        accents=accents or None,
        quotes=quotes or None,
        pollution=pollution or None,
        spaces=spaces or None,
    )

    return normalizer
