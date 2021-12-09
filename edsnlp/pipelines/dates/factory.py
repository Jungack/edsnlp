from typing import List, Union

from spacy.language import Language

from . import Dates, terms

default_config = dict(
    no_year=terms.no_year_pattern,
    absolute=terms.absolute_date_pattern,
    relative=terms.relative_date_pattern,
    full=terms.full_date_pattern,
    since=terms.since_pattern,
    false_positive=terms.false_positives,
)


# noinspection PyUnusedLocal
@Language.factory("dates", default_config=default_config)
def create_component(
    nlp: Language,
    name: str,
    no_year: Union[List[str], str],
    absolute: Union[List[str], str],
    full: Union[List[str], str],
    relative: Union[List[str], str],
    since: Union[List[str], str],
    false_positive: Union[List[str], str],
):
    return Dates(
        nlp,
        no_year=no_year,
        absolute=absolute,
        relative=relative,
        full=full,
        since=since,
        false_positive=false_positive,
    )
