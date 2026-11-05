"""Small compatibility helpers for loading the VLMs used in this project.

minicons currently loads every non-causal VLM through
``AutoModelForVision2Seq``.  Newer Hugging Face model families (InternVL,
Gemma 3, ...) are registered only under ``AutoModelForImageTextToText``
instead, so detect that from the real Auto mapping (rather than hardcoding
per-family name checks) and instantiate those models explicitly, then wrap
the model/processor objects with VLMScorer.
"""

import torch
from minicons import scorer
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor
from transformers.models.auto.modeling_auto import MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES


def _needs_image_text_to_text(model_name, cache_dir):
    config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
    model_type = getattr(config, "model_type", None)
    return model_type not in MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES


def load_vlm(model_name, device="cuda", torch_dtype=torch.bfloat16, cache_dir=None):
    """Return a minicons VLMScorer for any HF-format VLM."""
    kwargs = {"torch_dtype": torch_dtype}
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir

    if not _needs_image_text_to_text(model_name, cache_dir):
        return scorer.VLMScorer(model_name, device=device, **kwargs)

    processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForImageTextToText.from_pretrained(model_name, **kwargs)
    # VLMScorer's object-model path assumes ``tokenizer.vocab_size`` exists,
    # which is true for the underlying tokenizer but not ProcessorMixin.  Let
    # it initialize its vocabulary bookkeeping from the tokenizer, then put
    # the multimodal processor back for image+text encoding.
    lm = scorer.VLMScorer(model, device=device, tokenizer=processor.tokenizer)
    lm.tokenizer = processor
    return lm


def get_tied_token_embeddings(model):
    """Get and validate the tied input/output token embedding modules."""
    emb = model.get_input_embeddings()
    lm_head = model.get_output_embeddings()
    if emb is None or lm_head is None:
        raise RuntimeError(
            f"{type(model).__name__} does not expose both input and output embeddings"
        )
    if emb.weight.data_ptr() != lm_head.weight.data_ptr():
        raise RuntimeError(
            "Input embeddings and LM head are not tied; this intervention "
            "requires one shared parameter matrix."
        )
    return emb, lm_head


def get_input_output_embeddings(model):
    """Like get_tied_token_embeddings, but for models where the two may be
    separate parameter tensors (e.g. InternVL/Qwen3 backbones with
    tie_word_embeddings=False). Returns (emb, lm_head, is_tied) without
    asserting tying, so callers can branch on is_tied."""
    emb = model.get_input_embeddings()
    lm_head = model.get_output_embeddings()
    if emb is None or lm_head is None:
        raise RuntimeError(
            f"{type(model).__name__} does not expose both input and output embeddings"
        )
    is_tied = emb.weight.data_ptr() == lm_head.weight.data_ptr()
    return emb, lm_head, is_tied
