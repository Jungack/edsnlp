import catalogue
import inspect
import spacy
from dataclasses import dataclass
from functools import wraps
from spacy import Language
from spacy.pipe_analysis import validate_attrs
from typing import Callable, Optional, Any, Sequence, Dict, Iterable, List
from weakref import WeakKeyDictionary

import edsnlp
from confit import Registry, Config, set_default_registry
from confit.registry import validate_arguments
from edsnlp.utils.collections import FrozenDict, FrozenList

PIPE_META = WeakKeyDictionary()


def accepted_arguments(
    func: Callable,
    args: Sequence[str],
) -> List[str]:
    """
    Checks that a function accepts a list of keyword arguments

    Parameters
    ----------
    func: Callable[..., T]
        Function to check
    args: Union[str, Sequence[str]]
        Argument or list of arguments to check

    Returns
    -------
    List[str]
    """
    if isinstance(args, str):
        args = [args]
    sig = inspect.signature(func)
    has_kwargs = any(
        param.kind == param.VAR_KEYWORD for param in sig.parameters.values()
    )
    if has_kwargs:
        return args
    return [arg for arg in args if arg in sig.parameters]


@dataclass
class FactoryMeta:
    assigns: Iterable[str]
    requires: Iterable[str]
    retokenizes: bool
    default_config: Dict


class CurriedFactory:
    def __init__(self, func, factory_name, meta, kwargs):
        self.kwargs = kwargs
        self.factory = func
        self.factory_name = factory_name
        self.instantiated = None
        self.meta = meta

    def instantiate(
        obj: Any,
        nlp: "edsnlp.Pipeline",
        path: Sequence[str] = (),
    ):
        """
        To ensure compatibility with spaCy's API, we need to support
        passing in the nlp object and name to factories. Since they can be
        nested, we need to add them to every factory in the config.
        """
        if isinstance(obj, CurriedFactory):
            if obj.instantiated is not None:
                return obj.instantiated

            name = ".".join(path)

            kwargs = {
                key: CurriedFactory.instantiate(value, nlp, (*path, key))
                for key, value in obj.kwargs.items()
            }
            obj.instantiated = obj.factory(
                **{
                    "nlp": nlp,
                    "name": name,
                    **kwargs,
                }
            )
            Config._store_resolved(
                obj.instantiated,
                Config(
                    {
                        "@factory": obj.factory_name,
                        **kwargs,
                    }
                ),
            )
            PIPE_META[obj.instantiated] = obj.meta
            return obj.instantiated
        elif isinstance(obj, dict):
            return {
                key: CurriedFactory.instantiate(value, nlp, (*path, key))
                for key, value in obj.items()
            }
        elif isinstance(obj, tuple):
            return tuple(
                CurriedFactory.instantiate(value, nlp, (*path, str(i)))
                for i, value in enumerate(obj)
            )
        elif isinstance(obj, list):
            return [
                CurriedFactory.instantiate(value, nlp, (*path, str(i)))
                for i, value in enumerate(obj)
            ]
        else:
            return obj


class FactoryRegistry(Registry):
    """
    A registry that validates the input arguments of the registered functions.
    """

    def get(self, name: str) -> Any:
        """Get the registered function for a given name.

        name (str): The name.
        RETURNS (Any): The registered function.
        """
        namespace = list(self.namespace) + [name]
        if catalogue.check_exists(*namespace):
            return catalogue._get(namespace)

        spacy_namespace = ["spacy", "internal_factories", name]
        if catalogue.check_exists(*spacy_namespace):
            func = catalogue._get(spacy_namespace)
            meta = Language.get_factory_meta(name)

            def curried(**kwargs):
                return CurriedFactory(
                    func,
                    factory_name=name,
                    meta=meta,
                    kwargs=Config(meta.default_config).merge(kwargs),
                )

            return curried

        found_entry_point = False
        if self.entry_points:
            self.get_entry_point(name)

            if catalogue.check_exists(*namespace):
                found_entry_point = True

        if found_entry_point:
            return catalogue._get(namespace)

        available = self.get_available()
        current_namespace = " -> ".join(self.namespace)
        available_str = ", ".join(available) or "none"
        raise catalogue.RegistryError(
            f"Can't find '{name}' in registry {current_namespace}. "
            f"Available names: {available_str}"
        )

    def register(
        self,
        name: str,
        *,
        func: Optional[catalogue.InFunc] = None,
        default_config: Dict[str, Any] = FrozenDict(),
        assigns: Iterable[str] = FrozenList(),
        requires: Iterable[str] = FrozenList(),
        retokenizes: bool = False,
        default_score_weights: Dict[str, Optional[float]] = FrozenDict(),
    ) -> Callable[[catalogue.InFunc], catalogue.InFunc]:
        """
        This is a convenience wrapper around `confit.Registry.register`, that
        curries the function to be registered, allowing to instantiate the class
        later once `nlp` and `name` are known.

        Parameters
        ----------
        name: str
        func: Optional[catalogue.InFunc]
        default_config: Dict[str, Any]
        assigns: Iterable[str]
        requires: Iterable[str]
        retokenizes: bool
        default_score_weights: Dict[str, Optional[float]]

        Returns
        -------
        Callable[[catalogue.InFunc], catalogue.InFunc]
        """
        registerer = catalogue.Registry.register(self, name)

        save_params = {"@factory": name}

        def curry_and_register(fn: catalogue.InFunc) -> catalogue.InFunc:

            if len(accepted_arguments(fn, ["nlp", "name"])) < 2:
                raise ValueError(
                    "Factory functions must accept nlp and name as arguments."
                )
            # Officially register the factory, so we can later call
            # registry.resolve and refer to it in the config as
            # @factories = "spacy.Language.xyz". We use the class name here so
            # different classes can have different factories.

            validated_fn = validate_arguments(
                fn,
                config={"arbitrary_types_allowed": True},
                save_params=save_params,
                skip_save_params=["nlp", "name"],
            )

            # get default arguments from the function signature
            # using inspect
            inspect_args = inspect.signature(fn).parameters
            updated_default_config = Config(default_config).merge(
                {
                    key: value.default
                    for key, value in inspect_args.items()
                    if value.default is not inspect.Parameter.empty
                }
            )
            updated_default_config.pop("nlp", None)
            updated_default_config.pop("name", None)
            # merge with the default config
            meta = FactoryMeta(
                assigns=validate_attrs(assigns),
                requires=validate_attrs(requires),
                retokenizes=retokenizes,
                default_config=updated_default_config,
            )

            @wraps(
                fn,
                assigned=(
                    "__module__",
                    "__name__",
                    "__qualname__",
                    "__doc__",
                    "__annotations__",
                    "__signature__",
                ),
            )
            def configure(**kwargs):
                return CurriedFactory(
                    func=validated_fn,
                    factory_name=name,
                    meta=meta,
                    kwargs=updated_default_config.merge(kwargs),
                )

            registerer(configure)

            # Also register the function with spaCy to maintain compatibility
            spacy.Language.factory(
                name=name,
                default_config=updated_default_config,
                assigns=assigns,
                requires=requires,
                default_score_weights=default_score_weights,
                retokenizes=retokenizes,
                func=fn,
            )

            return validated_fn

        if func is not None:
            return curry_and_register(func)
        else:
            return curry_and_register


class registry:
    factory = factories = FactoryRegistry(("spacy", "factories"), entry_points=True)
    misc = Registry(("spacy", "misc"), entry_points=True)
    languages = Registry(("spacy", "languages"), entry_points=True)
    tokenizers = Registry(("spacy", "tokenizers"), entry_points=True)


set_default_registry(registry)
