"""
Stage-level batched inference for trankit.

Instead of processing each document through the full pipeline serially
(tokenize→posdep→lemmatize→ner per doc), this module merges sentences
from multiple documents and processes them through each stage together,
giving the GPU larger batches and reducing adapter-switch / CPU-prep overhead.

Usage:
    from trankit import Pipeline
    from trankit.batch_pipeline import batch_process

    p = Pipeline("english", gpu=True, embedding="xlm-roberta-large")
    results = batch_process(p, ["Doc one text...", "Doc two text...", ...])

All documents in a single batch_process call must be the same language
(the pipeline's current active language).
"""

import os
import torch
from copy import deepcopy
from torch.utils.data import DataLoader

from .iterators.tagger_iterators import TaggerDatasetLive
from .iterators.ner_iterators import NERDatasetLive
from .iterators import batch_to_device
from .utils.conll import ID, TEXT, SENTENCES, TOKENS, NER, LANG, UPOS, XPOS, FEATS, HEAD, DEPREL
from .utils.base_utils import get_output_doc
from .utils.chuliu_edmonds import chuliu_edmonds_one_root
from .utils.tbinfo import tbname2training_id, tbname2tagbatchsize, langwithner


_BATCH_TOKENIZE = os.environ.get('TRANKIT_BATCH_TOKENIZE', '1') == '1'


def batch_process(pipeline, docs, skip_dict_seq2seq=None, batch_tokenize=None):
    """Process multiple documents through the full pipeline with stage-level batching.

    Args:
        pipeline: An initialized trankit.Pipeline instance.
        docs: List of document strings (all same language as pipeline.active_lang).
        skip_dict_seq2seq: Optional override for lemmatizer dict-skip optimization.
        batch_tokenize: If True, merge tokenizer GPU passes across docs.
            If False, tokenize each doc independently. Defaults to
            TRANKIT_BATCH_TOKENIZE env var (1=on, 0=off), which defaults on.

    Returns:
        List of result dicts, one per input document, in the same format as
        pipeline(text) — each with 'text', 'sentences', and 'lang' keys.
    """
    if not docs:
        return []

    config = pipeline._config
    active_lang = config.active_lang

    # Guard: all docs processed under the same language / adapter set
    assert active_lang is not None, "Pipeline has no active language set."

    use_batch_tok = batch_tokenize if batch_tokenize is not None else _BATCH_TOKENIZE

    # ── Stage 1: Tokenize ──
    all_tokenized = []  # flat list of sentence dicts across all docs
    doc_sent_counts = []  # number of sentences per doc, for demux

    if use_batch_tok:
        all_doc_sents = pipeline._tokenize_docs(docs)
        for sents in all_doc_sents:
            doc_sent_counts.append(len(sents))
            all_tokenized.extend(sents)
    else:
        for doc_text in docs:
            sents = pipeline._tokenize_doc(in_doc=doc_text)
            doc_sent_counts.append(len(sents))
            all_tokenized.extend(sents)

    if not all_tokenized:
        return [{TEXT: doc_text, SENTENCES: [], LANG: active_lang} for doc_text in docs]

    # ── Stage 2: POS tagging + dependency parsing (merged batch) ──────────
    tagger_test_set = TaggerDatasetLive(
        tokenized_doc=all_tokenized,
        wordpiece_splitter=config.wordpiece_splitter,
        config=config,
    )
    tagger_test_set.numberize()

    pipeline._load_adapter_weights(model_name='tagger')

    eval_batch_size = tbname2tagbatchsize.get(config.treebank_name, pipeline._tagbatchsize)
    if config.embedding_name == 'xlm-roberta-large':
        eval_batch_size = int(eval_batch_size / 3)

    itos = config.itos[active_lang]

    with torch.inference_mode(), pipeline._autocast():
        for batch in DataLoader(tagger_test_set,
                                batch_size=eval_batch_size,
                                shuffle=False, collate_fn=tagger_test_set.collate_fn,
                                pin_memory=pipeline._pin_memory):
            batch = batch_to_device(batch, config.device, non_blocking=pipeline._non_blocking)
            batch_size = len(batch.word_num)

            word_reprs, cls_reprs = pipeline._embedding_layers.get_tagger_inputs(batch)
            predictions = pipeline._tagger[active_lang].predict(batch, word_reprs, cls_reprs)

            tag_stacked = torch.stack([predictions[0], predictions[1], predictions[2]]).detach().cpu().tolist()
            predicted_upos, predicted_xpos, predicted_feats = tag_stacked

            predicted_dep = predictions[3]
            dep_unlabeled = predicted_dep[0].cpu().numpy()
            dep_labeled = predicted_dep[1].cpu().numpy()
            sentlens = [l + 1 for l in batch.word_num]
            head_seqs = [chuliu_edmonds_one_root(adj[:l, :l])[1:] for adj, l in
                         zip(dep_unlabeled, sentlens)]
            deprel_seqs = [
                [itos[DEPREL][dep_labeled[i][j + 1][h]] for j, h in
                 enumerate(hs)] for i, hs in enumerate(head_seqs)]

            pred_tokens = [[[head_seqs[i][j], deprel_seqs[i][j]] for j in range(sentlens[i] - 1)] for i in
                           range(batch_size)]

            for bid in range(batch_size):
                sentid = batch.sent_index[bid]
                for i in range(batch.word_num[bid]):
                    wordid = batch.word_ids[bid][i]
                    tagger_test_set.conllu_doc[sentid][wordid][UPOS] = itos[UPOS][predicted_upos[bid][i]]
                    tagger_test_set.conllu_doc[sentid][wordid][XPOS] = itos[XPOS][predicted_xpos[bid][i]]
                    tagger_test_set.conllu_doc[sentid][wordid][FEATS] = itos[FEATS][predicted_feats[bid][i]]
                    tagger_test_set.conllu_doc[sentid][wordid][HEAD] = int(pred_tokens[bid][i][0])
                    tagger_test_set.conllu_doc[sentid][wordid][DEPREL] = pred_tokens[bid][i][1]

            del predictions, sentlens, head_seqs, deprel_seqs, pred_tokens

    tagged_doc = get_output_doc(all_tokenized, tagger_test_set.conllu_doc)

    # ── Stage 3: Lemmatization (merged batch) ────────────────────────────
    out = pipeline._lemmatize_doc(tagged_doc, skip_dict_seq2seq=skip_dict_seq2seq)

    # ── Stage 4: NER (merged batch, reusing tagger dataset) ──────────────
    if active_lang in langwithner:
        has_mwt = tbname2training_id[config.treebank_name] % 2 == 1
        if has_mwt:
            # MWT changes word structure; fall back to standard NER path
            out = pipeline._ner_doc(out)
        else:
            # Fast path: reuse tagger-prepared data
            ner_test_set = NERDatasetLive.from_tagger_data(config, tagger_test_set)

            pipeline._load_adapter_weights(model_name='ner')

            with torch.inference_mode(), pipeline._autocast():
                for batch in DataLoader(ner_test_set,
                                        batch_size=eval_batch_size,
                                        shuffle=False, collate_fn=ner_test_set.collate_fn,
                                        pin_memory=pipeline._pin_memory):
                    batch = batch_to_device(batch, config.device, non_blocking=pipeline._non_blocking)
                    word_reprs, cls_reprs = pipeline._embedding_layers.get_tagger_inputs(batch)
                    pred_entity_labels = pipeline._ner_model[active_lang].predict(batch, word_reprs)

                    batch_size = len(batch.word_num)
                    for bid in range(batch_size):
                        sentid = batch.sent_index[bid]
                        for i in range(batch.word_num[bid]):
                            wordid = batch.word_ids[bid][i]
                            out[sentid][TOKENS][wordid][NER] = pred_entity_labels[bid][i]

                    del pred_entity_labels

    # ── Demux: slice merged results back to per-document ─────────────────
    results = []
    offset = 0
    for doc_text, n_sents in zip(docs, doc_sent_counts):
        doc_sents = out[offset:offset + n_sents]
        # Restore per-document sentence IDs (1-based)
        for i, sent in enumerate(doc_sents):
            sent[ID] = i + 1
        results.append({TEXT: doc_text, SENTENCES: doc_sents, LANG: active_lang})
        offset += n_sents

    return results
