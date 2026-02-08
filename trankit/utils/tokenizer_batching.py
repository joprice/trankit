def batched_tokenize_pseudo_tokens(tokenizer, pseudo_tokens):
    """Batch-tokenize pseudo-tokens, returning piece strings per group.

    Returns a list of lists: groups[i] contains the wordpiece strings
    for pseudo_tokens[i], with standalone '▁' tokens filtered out.
    """
    if not pseudo_tokens:
        return []
    if not tokenizer.is_fast:
        raise TypeError(
            f"batched_tokenize_pseudo_tokens requires a fast tokenizer "
            f"(got {type(tokenizer).__name__})"
        )
    enc = tokenizer(pseudo_tokens, is_split_into_words=True, add_special_tokens=False)
    wids = enc.word_ids()
    if wids is None:
        raise RuntimeError(
            "word_ids() returned None — is_split_into_words may not be supported"
        )
    ids = enc.input_ids
    all_tokens = tokenizer.convert_ids_to_tokens(ids)
    n = len(pseudo_tokens)
    groups = [[] for _ in range(n)]
    for pos, wid in enumerate(wids):
        if wid is not None and all_tokens[pos] != '▁':
            groups[wid].append(all_tokens[pos])
    return groups
