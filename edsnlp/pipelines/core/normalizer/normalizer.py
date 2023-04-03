from spacy.tokens import Doc
from typing import Optional

from .accents import Accents
from .pollution import Pollution
from .quotes import Quotes
from .remove_lowercase import RemoveLowercase
from .spaces import Spaces


class Normalizer(object):
    """
    Normalisation pipeline. Modifies the `NORM` attribute,
    acting on five dimensions :

    - `remove_lowercase`: using the default `NORM`
    - `accents`: deterministic and fixed-length normalisation of accents.
    - `quotes`: deterministic and fixed-length normalisation of quotation marks.
    - `spaces`: "removal" of spaces tokens (via the tag_ attribute).
    - `pollution`: "removal" of pollutions (via the tag_ attribute).
    """

    def __init__(
        self,
        remove_lowercase: Optional[RemoveLowercase],
        accents: Optional[Accents],
        quotes: Optional[Quotes],
        spaces: Optional[Spaces],
        pollution: Optional[Pollution],
    ):
        """
        Parameters
        ----------
        remove_lowercase : bool
            Whether to remove case.
        accents : Optional[Accents]
            Optional `Accents` object.
        quotes : Optional[Quotes]
            Optional `Quotes` object.
        spaces : Optional[Spaces]
            Optional `Spaces` object.
        pollution : Optional[Pollution]
            Optional `Pollution` object.
        """
        self.remove_lowercase = remove_lowercase
        self.accents = accents
        self.quotes = quotes
        self.spaces = spaces
        self.pollution = pollution

    def __call__(self, doc: Doc) -> Doc:
        """
        Apply the normalisation pipeline, one component at a time.

        Parameters
        ----------
        doc : Doc
            spaCy `Doc` object

        Returns
        -------
        Doc
            Doc object with `NORM` attribute modified
        """
        if self.remove_lowercase is not None:
            self.remove_lowercase(doc)
        if self.accents is not None:
            self.accents(doc)
        if self.quotes is not None:
            self.quotes(doc)
        if self.spaces is not None:
            self.spaces(doc)
        if self.pollution is not None:
            self.pollution(doc)
        return doc
