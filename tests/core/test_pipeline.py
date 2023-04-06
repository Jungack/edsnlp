from io import BytesIO

import pytest
import torch

import edsnlp
from confit import Config
from edsnlp import Pipeline


@pytest.fixture
def hybrid_nlp():
    nlp = edsnlp.blank("eds")
    nlp.add_pipe("eds.normalizer")
    nlp.add_pipe("eds.sentences")
    nlp.add_pipe(
        "eds.transformer",
        config={
            "model": "prajjwal1/bert-tiny",
            "stride": 64,
            "window": 128,
        },
    )
    nlp.add_pipe(
        "eds.ner",
        config={
            "embedding": nlp.get_pipe("eds.transformer"),
        },
    )
    return nlp


def test_rule_based_pipeline():
    nlp = edsnlp.blank("eds")
    nlp.add_pipe("eds.normalizer")
    nlp.add_pipe("eds.covid")

    assert nlp.pipe_names == ["eds.normalizer", "eds.covid"]
    assert nlp.get_pipe("eds.normalizer") == nlp.pipeline[0][1]
    assert nlp.has_pipe("eds.covid")

    with pytest.raises(ValueError) as exc_info:
        nlp.get_pipe("unknown")

    assert str(exc_info.value) == "Pipe 'unknown' not found in pipeline."

    doc = nlp.make_doc("Mon patient a le covid")

    new_doc = nlp(doc)

    assert len(doc.ents) == 1
    assert new_doc is doc

    assert nlp.get_pipe_meta("eds.covid").assigns == ["doc.ents", "doc.spans"]


def test_config():
    config = Config.from_str(
        """
        [nlp]
        lang = "eds"
        pipeline = ["eds.normalizer", "eds.covid"]

        [components]

        [components."eds.normalizer"]
        @factory = "eds.normalizer"

        [components."eds.covid"]
        @factory = "eds.covid"
        """
    )

    with pytest.raises(ValueError) as exc_info:
        nlp = edsnlp.blank("fr", config=config)

    assert (
        str(exc_info.value) == "The language specified in the config "
        "does not match the lang argument."
    )

    nlp = edsnlp.blank("eds", config=config)
    assert nlp.pipe_names == ["eds.normalizer", "eds.covid"]


def test_config_error():
    config = Config.from_str(
        """
        [nlp]
        lang = "eds"
        pipeline = ["eds.normalizer"]

        [components]

        [components."eds.normalizer"]
        factory = "eds.normalizer"
        """
    )

    with pytest.raises(ValueError) as exc_info:
        edsnlp.blank("eds", config=config)

    assert str(exc_info.value) == (
        "Component 'eds.normalizer' is not instantiable. Please make sure "
        "that you didn't forget to add a '@factory' key to the component "
        "config."
    )


def test_config_serialization():
    nlp = edsnlp.blank("eds")
    nlp.add_pipe("eds.normalizer")
    nlp.add_pipe("eds.covid")

    str_config = nlp.config.to_str()
    assert (
        str_config
        == """
[nlp]
lang = "eds"
pipeline = ["eds.normalizer", "eds.covid"]
disabled = []

[nlp.tokenizer]
@tokenizers = "eds.tokenizer"

[components]

[components.'eds.normalizer']
@factory = "eds.normalizer"

[components.'eds.covid']
@factory = "eds.covid"

""".lstrip()
    )

    nlp2 = Pipeline.from_config(Config.from_str(str_config))
    assert nlp2.pipe_names == ["eds.normalizer", "eds.covid"]


def test_disk_serialization(tmp_path, hybrid_nlp):
    nlp = hybrid_nlp

    ner = nlp.get_pipe("eds.ner")
    ner.update_labels(["LOC", "PER"])
    nlp.to_disk(tmp_path / "model")

    assert (tmp_path / "model" / "config.cfg").exists()
    assert (
        tmp_path / "model" / "tensors" / "eds.ner+eds.transformer.safetensors"
    ).exists()

    print((tmp_path / "model" / "config.cfg").read_text())

    nlp = edsnlp.blank(
        "eds", config=Config.from_disk(tmp_path / "model" / "config.cfg")
    )
    nlp.from_disk(tmp_path / "model")
    assert nlp.get_pipe("eds.ner").labels == ["LOC", "PER"]


def test_torch_save(hybrid_nlp):
    hybrid_nlp.get_pipe("eds.ner").update_labels(["LOC", "PER"])
    buffer = BytesIO()
    torch.save(hybrid_nlp, buffer)
    buffer.seek(0)
    nlp = torch.load(buffer)
    assert nlp.get_pipe("eds.ner").labels == ["LOC", "PER"]
    assert len(list(nlp("Une phrase. Deux phrases.").sents)) == 2
