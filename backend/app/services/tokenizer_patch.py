"""Monkey-patch prepare_for_model removed in transformers 5.x for FlagEmbedding."""


def _prepare_for_model(self, ids_1, ids_2=None, **kwargs):
    text = self.decode(ids_1, skip_special_tokens=True)
    text_pair = self.decode(ids_2, skip_special_tokens=True) if ids_2 is not None else None

    tokenizer_kwargs = {}
    for key in ("truncation", "max_length", "padding"):
        if key in kwargs:
            tokenizer_kwargs[key] = kwargs[key]

    return self(
        text,
        text_pair=text_pair,
        add_special_tokens=True,
        return_tensors=None,
        **tokenizer_kwargs,
    )


def apply():
    from transformers import PreTrainedTokenizerBase

    if hasattr(PreTrainedTokenizerBase, "prepare_for_model"):
        return  # already exists, no patch needed

    PreTrainedTokenizerBase.prepare_for_model = _prepare_for_model
