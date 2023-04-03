from loguru import logger
from spacy.tokens import Doc, Span
from typing import Dict, List, Union

from edsnlp.core import PipelineProtocol
from edsnlp.pipelines.core.matcher import GenericMatcher
from edsnlp.utils.filter import get_spans
from edsnlp.utils.inclusion import check_inclusion


class Reason(GenericMatcher):
    """Pipeline to identify the reason of the hospitalisation.

    It declares a Span extension called `ents_reason` and adds
    the key `reasons` to doc.spans.

    It also declares the boolean extension `is_reason`.
    This extension is set to True for the Reason Spans but also
    for the entities that overlap the reason span.
    """

    def __init__(
        self,
        nlp: PipelineProtocol,
        reasons: Dict[str, Union[List[str], str]],
        reason_sections: List[str],
        excluded_sections: List[str],
        use_sections: bool = False,
        attr: Union[Dict[str, str], str] = "TEXT",
        ignore_excluded: bool = False,
    ):
        """
        Parameters
        ----------
        nlp : Pipeline
            EDS-NLP pipeline object
        reasons : Dict[str, Union[List[str], str]]
            The terminology of reasons.
        reason_sections: List[str]
            The list of sections that are considered as reasons.
        excluded_sections: List[str]
            The list of sections we don't search into.
        use_sections : bool
            Whether to use the `sections` pipeline to improve results.
        attr : str
            spaCy's attribute to use:
            a string with the value "TEXT" or "NORM", or a dict with
            the key 'term_attr'. We can also add a key for each regex.
        ignore_excluded : bool
            Whether to skip excluded tokens.
        """
        super().__init__(
            nlp,
            terms=None,
            regex=reasons,
            attr=attr,
            ignore_excluded=ignore_excluded,
        )

        assert not (
            use_sections and (reason_sections is None or excluded_sections is None)
        ), (
            "You must provide the `reason_sections` and `excluded_sections` "
            "parameters when enabling the `use_sections` option."
        )

        self.reason_sections = reason_sections
        self.excluded_sections = excluded_sections

        self.use_sections = use_sections and (
            "eds.sections" in nlp.pipe_names or "sections" in nlp.pipe_names
        )
        if use_sections and not self.use_sections:
            logger.warning(
                "You have requested that the pipeline use annotations "
                "provided by the `eds.section` pipeline, but it was not set. "
                "Skipping that step."
            )

        self.set_extensions()

    @classmethod
    def set_extensions(cls) -> None:

        if not Span.has_extension("ents_reason"):
            Span.set_extension("ents_reason", default=None)

        if not Span.has_extension("is_reason"):
            Span.set_extension("is_reason", default=False)

    def _enhance_with_sections(self, sections: List, reasons: List) -> List:
        """Enhance the list of reasons with the section information.
        If the reason overlaps with history, so it will be removed from the list

        Parameters
        ----------
        sections : List
            Spans of sections identified with the `sections` pipeline
        reasons : List
            Reasons list identified by the regex

        Returns
        -------
        List
            Updated list of spans reasons
        """

        for section in sections:
            if section.label_ in self.reason_sections:
                reasons.append(section)

            if section.label_ in self.excluded_sections:
                for reason in reasons:
                    if check_inclusion(reason, section.start, section.end):
                        reasons.remove(reason)

        return reasons

    def __call__(self, doc: Doc) -> Doc:
        """Find spans related to the reasons of the hospitalisation

        Parameters
        ----------
        doc : Doc

        Returns
        -------
        Doc
        """
        matches = self.process(doc)
        reasons = get_spans(matches, "reasons")

        if self.use_sections:
            sections = doc.spans["sections"]
            reasons = self._enhance_with_sections(sections=sections, reasons=reasons)

        doc.spans["reasons"] = reasons

        # Entities
        if len(doc.ents) > 0:
            for reason in reasons:  # TODO optimize this iteration
                ent_list = []
                for ent in doc.ents:
                    if check_inclusion(ent, reason.start, reason.end):
                        ent_list.append(ent)
                        ent._.is_reason = True

                reason._.ents_reason = ent_list
                reason._.is_reason = True

        return doc
