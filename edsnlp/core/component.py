import contextlib
import torch
import typing
from abc import ABCMeta
from enum import Enum
from functools import wraps
from spacy.tokens import Doc
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from edsnlp.utils.collections import batch_compress_dict, batchify, decompress_dict

BatchInput = typing.TypeVar("BatchInput", bound=Dict[str, Any])
BatchOutput = typing.TypeVar("BatchOutput", bound=Dict[str, Any])
Scorer = Callable[[Sequence[Tuple[Doc, Doc]]], Union[float, Dict[str, Any]]]


class CacheEnum(str, Enum):
    preprocess = "preprocess"
    collate = "collate"
    forward = "forward"


def hash_batch(batch):
    if isinstance(batch, list):
        return hash(tuple(id(item) for item in batch))
    elif not isinstance(batch, dict):
        return id(batch)
    return hash((tuple(batch.keys()), tuple(map(hash_batch, batch.values()))))


def cached_preprocess(fn):
    @wraps(fn)
    def wrapped(self: "TorchComponent", doc: Doc):
        if not self._do_cache:
            return fn(self, doc)
        cache_id = hash(id(doc))
        if cache_id in self._preprocess_cache:
            return self._preprocess_cache[cache_id]
        res = fn(self, doc)
        self._preprocess_cache[cache_id] = res
        return res

    return wrapped


def cached_preprocess_supervised(fn):
    @wraps(fn)
    def wrapped(self: "TorchComponent", doc: Doc):
        if not self._do_cache:
            return fn(self, doc)
        cache_id = hash(id(doc))
        if cache_id in self._preprocess_supervised_cache:
            return self._preprocess_supervised_cache[cache_id]
        res = fn(self, doc)
        self._preprocess_supervised_cache[cache_id] = res
        return res

    return wrapped


def cached_collate(fn):
    import torch

    @wraps(fn)
    def wrapped(self: "TorchComponent", batch: Dict, device: torch.device):
        cache_id = hash_batch(batch)
        if not self._do_cache or cache_id is None:
            return fn(self, batch, device)
        if cache_id in self._collate_cache:
            return self._collate_cache[cache_id]
        res = fn(self, batch, device)
        self._collate_cache[cache_id] = res
        res["cache_id"] = cache_id
        return res

    return wrapped


def cached_forward(fn):
    @wraps(fn)
    def wrapped(self: "TorchComponent", batch):
        # Convert args and kwargs to a dictionary matching fn signature
        cache_id = hash_batch(batch)
        if not self._do_cache or cache_id is None:
            return fn(self, batch)
        if cache_id in self._forward_cache:
            return self._forward_cache[cache_id]
        res = fn(self, batch)
        self._forward_cache[cache_id] = res
        return res

    return wrapped


class TorchComponentMeta(ABCMeta):
    def __new__(meta, name, bases, class_dict):
        if "preprocess" in class_dict:
            class_dict["preprocess"] = cached_preprocess(class_dict["preprocess"])
        if "preprocess_supervised" in class_dict:
            class_dict["preprocess_supervised"] = cached_preprocess_supervised(
                class_dict["preprocess_supervised"]
            )
        if "collate" in class_dict:
            class_dict["collate"] = cached_collate(class_dict["collate"])
        if "forward" in class_dict:
            class_dict["forward"] = cached_forward(class_dict["forward"])

        return super().__new__(meta, name, bases, class_dict)


class TorchComponent(
    torch.nn.Module,
    typing.Generic[BatchOutput, BatchInput],
    metaclass=TorchComponentMeta,
):
    """
    A TorchComponent is a Component that can be trained and inherits `torch.nn.Module`.
    You can use it either as a torch module inside a more complex neural network, or as
    a standalone component in a [Pipeline][edsnlp.core.pipeline.Pipeline].

    In addition to the methods of a torch module, a TorchComponent adds a few methods to
    handle preprocessing and collating features, as well as caching intermediate results
    for components that share a common subcomponent.
    """

    def __init__(self):
        super().__init__()
        self.cfg = {}
        self._preprocess_cache = {}
        self._preprocess_supervised_cache = {}
        self._collate_cache = {}
        self._forward_cache = {}
        self._do_cache = False

    @contextlib.contextmanager
    def cache(self):
        did_cache_before = self.enable_cache(True)
        yield
        self.enable_cache(did_cache_before)

    def post_init(self, gold_data: Iterable[Doc], **kwargs):
        """
        Initialize the missing properties of the module, such as its vocabulary,
        using the gold data and the provided keyword arguments.

        Parameters
        ----------
        gold_data: Iterable[Doc]
            Gold data to use for initialization
        kwargs: Any
            Additional keyword arguments to use for initialization
        """
        for name, value in kwargs.items():
            if value is None:
                continue
            current_value = getattr(self, name)
            if current_value is not None and current_value != value:
                raise ValueError(
                    "Cannot initialize module with different values for "
                    "attribute '{}': {} != {}".format(name, current_value, value)
                )
            setattr(self, name, value)

    def enable_cache(self, do_cache):
        saved = self._do_cache
        self._do_cache = do_cache

        for module in self.modules():
            if isinstance(module, TorchComponent) and module is not self:
                module.enable_cache(do_cache)

        return saved

    @property
    def device(self):
        return next((p.device for p in self.parameters()), torch.device("cpu"))

    def reset_cache(self, cache: Optional[CacheEnum] = None):
        def clear(module):
            try:
                assert cache is None or cache in (
                    CacheEnum.preprocess,
                    CacheEnum.collate,
                    CacheEnum.forward,
                )
                if cache is None or cache == CacheEnum.preprocess:
                    module._preprocess_cache.clear()
                if cache is None or cache == CacheEnum.preprocess:
                    module._preprocess_supervised.clear()
                if cache is None or cache == CacheEnum.collate:
                    module._collate_cache.clear()
                if cache is None or cache == CacheEnum.forward:
                    module._forward_cache.clear()
            except AttributeError:
                pass

        self.apply(clear)

    def preprocess(self, doc: Doc) -> Dict[str, Any]:
        """
        Preprocess the document to extract features that will be used by the
        neural network to perform its predictions.

        Parameters
        ----------
        doc: Doc
            Document to preprocess

        Returns
        -------
        Dict[str, Any]
            Dictionary (optionally nested) containing the features extracted from
            the document.
        """
        return {}

    def collate(
        self, batch: Dict[str, Sequence[Any]], device: torch.device
    ) -> BatchInput:
        """
        Collate the batch of features into a single batch of tensors that can be
        used by the forward method of the component.

        Parameters
        ----------
        batch: Dict[str, Sequence[Any]]
            Batch of features
        device: torch.device
            Device on which the tensors should be moved

        Returns
        -------
        BatchInput
            Dictionary (optionally nested) containing the collated tensors
        """
        return {}

    def forward(self, batch: BatchInput) -> BatchOutput:
        """
        Perform the forward pass of the neural network.

        Parameters
        ----------
        batch: BatchInput
            Batch of tensors (nested dictionary) computed by the collate method

        Returns
        -------
        BatchOutput
        """
        raise NotImplementedError()

    def module_forward(self, *args, **kwargs):
        """
        This is a wrapper around `torch.nn.Module.__call__` to avoid conflict
        with the [`Component.__call__`][edspdf.component.Component.__call__]
        method.
        """
        return torch.nn.Module.__call__(self, *args, **kwargs)

    def make_batch(self, docs: Sequence[Doc]) -> Dict[str, Sequence[Any]]:
        """
        Convenience method to preprocess a batch of documents and collate them
        Features corresponding to the same path are grouped together in a list,
        under the same key.

        Parameters
        ----------
        docs: Sequence[Doc]
            Batch of documents

        Returns
        -------
        Dict[str, Sequence[Any]]
        """
        batch = decompress_dict(
            list(
                batch_compress_dict([{self.name: self.preprocess(doc)} for doc in docs])
            )
        )
        return batch

    def batch_process(self, docs: Sequence[Doc]) -> Sequence[Doc]:
        """
        Process a batch of documents using the neural network.
        This differs from the `pipe` method in that it does not return an
        iterator, but executes the component on the whole batch at once.

        Parameters
        ----------
        docs: Sequence[Doc]
            Batch of documents

        Returns
        -------
        Sequence[Doc]
            Batch of updated documents
        """
        with torch.no_grad():
            batch = self.make_batch(docs)
            inputs = self.collate(batch[self.name], device=self.device)
            if hasattr(self, "compiled"):
                res = self.compiled(inputs)
            else:
                res = self.module_forward(inputs)
            docs = self.postprocess(docs, res)
            return docs

    def postprocess(self, docs: Sequence[Doc], batch: BatchOutput) -> Sequence[Doc]:
        """
        Update the documents with the predictions of the neural network.
        By default, this is a no-op.

        Parameters
        ----------
        docs: Sequence[Doc]
            Batch of documents
        batch: BatchOutput
            Batch of predictions, as returned by the forward method

        Returns
        -------
        Sequence[Doc]
        """
        return docs

    # Same as preprocess but with gold supervision data
    def preprocess_supervised(self, doc: Doc) -> Dict[str, Any]:
        """
        Preprocess the document to extract features that will be used by the
        neural network to perform its training.
        By default, this returns the same features as the `preprocess` method.

        Parameters
        ----------
        doc: Doc
            Document to preprocess

        Returns
        -------
        Dict[str, Any]
            Dictionary (optionally nested) containing the features extracted from
            the document.
        """
        return self.preprocess(doc)

    def clean_gold_for_evaluation(self, gold: Doc) -> Doc:
        """
        Clean the gold document before evaluation.
        Only the attributes that are predicted by the component should be removed.
        By default, this is a no-op.

        Parameters
        ----------
        gold: Doc
            Gold document

        Returns
        -------
        Doc
            The document without attributes that should be predicted
        """
        return gold

    def pipe(self, docs: Iterable[Doc], batch_size=1) -> Iterable[Doc]:
        """
        Applies the component on a collection of documents. It is recommended to use
        the [`Pipeline.pipe`][edsnlp.core.pipeline.Pipeline.pipe] method instead of this
        one to apply a pipeline on a collection of documents, to benefit from the
        caching of intermediate results.

        Parameters
        ----------
        docs: Iterable[Doc]
            Input docs
        batch_size: int
            Batch size to use when making batched to be process at once
        """
        for batch in batchify(docs, batch_size=batch_size):
            self.reset_cache()
            yield from self.batch_process(batch)
