from .config import Config as MasterConfig
from .models.base_models import Multilingual_Embedding
from .models.classifiers import TokenizerClassifier, PosDepClassifier, NERClassifier
from .models.mwt_model import MWTWrapper
from .models.lemma_model import LemmaWrapper
from .iterators.tokenizer_iterators import TokenizeDatasetLive
from .iterators.tagger_iterators import TaggerDatasetLive
from .iterators.ner_iterators import NERDatasetLive
from .iterators import batch_to_device
from .utils.tokenizer_utils import *
from collections import defaultdict
from .utils.conll import *
from .utils.tbinfo import tbname2training_id, lang2treebank
from .utils.chuliu_edmonds import *
from adapters.loading import AdapterLoader
from adapters import AdapterConfig, Stack
from adapters.composition import parse_composition
from contextlib import nullcontext
from datetime import datetime
import langid
import re
import hashlib

import os

_BYPASS_ADAPTER_RESET = os.environ.get('TRANKIT_BYPASS_ADAPTER_RESET', '1') == '1'

from transformers import XLMRobertaTokenizerFast

TRANKIT_QUIET = os.environ.get("TRANKIT_QUIET", "").lower() in ("1", "true", "yes")

_ADAPTER_NAME_RE = re.compile(r'[^A-Za-z0-9_]+')


def _adapter_slot_name(task, lang):
    """Build a safe, collision-free adapter slot name.

    Sanitizes lang to [A-Za-z0-9_] and appends a short hash of the
    original lang to avoid collisions (e.g. 'french-partut' and
    'french_partut' would otherwise both sanitize to the same string).

    Example: 'tokenizer_french_partut_a1b2'
    """
    sanitized = _ADAPTER_NAME_RE.sub('_', lang)
    short_hash = hashlib.md5(lang.encode("utf-8")).hexdigest()[:4]
    return f"{task}_{sanitized}_{short_hash}"


def _active_adapter_names(xlmr):
    """Return set of active adapter name strings, defensively handling
    composition objects and None."""
    active = xlmr.active_adapters
    if active is None:
        return set()
    if isinstance(active, str):
        return {active}
    if hasattr(active, 'flatten'):
        try:
            return set(str(n) for n in active.flatten())
        except Exception:
            return set()
    if isinstance(active, (list, tuple)):
        return set(str(n) for n in active)
    return set()


def is_string(input):
    if isinstance(input, str) and len(input.strip()) > 0:
        return True
    return False


def is_list_strings(input):
    if isinstance(input, list) and len(input) > 0:
        for element in input:
            if not (isinstance(element, str) and not element.isspace()):
                return False
        return True
    return False


def is_list_list_strings(input):
    if isinstance(input, list) and len(input) > 0 and isinstance(input[0], list) and len(input[0]) > 0:
        for element in input[0]:
            if not (isinstance(element, str) and not element.isspace()):
                return False
        return True
    return False


class Pipeline:
    def __init__(self, lang, cache_dir=None, gpu=True, embedding='xlm-roberta-base',
                 cpu_lemma=None, fp16=None, cache_adapters=False):
        super(Pipeline, self).__init__()
        # auto detection of lang
        if lang == 'auto':
            lang = list(code2lang.values())[0]
            self.auto_mode = True
        else:
            self.auto_mode = False

        # set the embedding type
        assert embedding in supported_embeddings, f'{embedding} has not been supported.\nSupported embeddings: {supported_embeddings}'

        self.master_config = MasterConfig()
        self.master_config.embedding_name = embedding

        self._cache_dir = cache_dir
        self._gpu = gpu
        self._use_gpu = gpu
        self._ud_eval = False
        self._setup_config(lang)
        self._config.training = False
        # CPU lemma decode avoids per-step GPU→CPU sync; auto-enable on MPS
        if cpu_lemma is None:
            self._cpu_lemma = (self._config.device.type == 'mps')
        else:
            self._cpu_lemma = cpu_lemma
        # FP16 autocast: off by default (weights already fp16 via .half() on CUDA)
        if fp16 is None:
            self._fp16 = False
        else:
            self._fp16 = fp16
        device_type = self._config.device.type
        if self._fp16 and device_type in ('cuda', 'cpu'):
            self._autocast = lambda: torch.autocast(device_type, dtype=torch.float16)
        else:
            self._autocast = nullcontext

        self.added_langs = [lang]
        assert lang in lang2treebank, f'{lang} has not been supported. Currently supported languages: {list(lang2treebank.keys())}'

        # download saved model for initial language
        download(
            cache_dir=self._config._cache_dir,
            language=lang,
            saved_model_version=saved_model_version,  # manually set this to avoid duplicated storage
            embedding_name=self.master_config.embedding_name
        )

        # load ALL vocabs
        self._load_vocabs()

        # shared multilingual embeddings
        print('Loading pretrained XLM-Roberta, this may take a while...')
        self._embedding_layers = Multilingual_Embedding(self._config)
        self._embedding_layers.to(self._config.device)
        if self._use_half:
            self._embedding_layers.half()
        self._embedding_layers.eval()
        # for loading & auto-converting adapter weights
        self._adapter_loader = AdapterLoader(self._embedding_layers.xlmr, "text_task")

        self._cache_adapters = cache_adapters
        if cache_adapters:
            self._resident_adapters = set()
            self._active_slot = None  # tracks current set_active_adapters slot
            self._adapter_config = AdapterConfig.load(
                "pfeiffer",
                reduction_factor=6 if self._config.embedding_name == 'xlm-roberta-base' else 4
            )

        # tokenizers
        self._tokenizer = {}
        self._tokenizer[lang] = TokenizerClassifier(self._config, treebank_name=lang2treebank[lang])
        self._tokenizer[lang].to(self._config.device)
        if self._use_half:
            self._tokenizer[lang].half()
        self._tokenizer[lang].eval()

        # taggers
        self._tagger = {}
        self._tagger[lang] = PosDepClassifier(self._config, treebank_name=lang2treebank[lang])
        self._tagger[lang].to(self._config.device)
        if self._use_half:
            self._tagger[lang].half()
        self._tagger[lang].eval()

        # mwt and lemma:
        self._mwt_model = {}
        treebank_name = lang2treebank[lang]
        if tbname2training_id[treebank_name] % 2 == 1:
            self._mwt_model[lang] = MWTWrapper(self._config, treebank_name=treebank_name, use_gpu=self._use_gpu)

        self._lemma_model = {}
        treebank_name = lang2treebank[lang]
        self._lemma_model[lang] = LemmaWrapper(self._config, treebank_name=treebank_name, use_gpu=self._use_gpu, cpu_lemma=self._cpu_lemma)

        # ner if available
        self._ner_model = {}
        if lang in langwithner:
            self._ner_model[lang] = NERClassifier(self._config, lang)
            self._ner_model[lang].to(self._config.device)
            if self._use_half:
                self._ner_model[lang].half()
            self._ner_model[lang].eval()

        # load and hold the pretrained weights
        self._embedding_weights = self._embedding_layers.state_dict()

        if self.auto_mode:
            for l in code2lang.values():
                if l not in self.added_langs:
                    self.add(l)
            # constrain the language set for auto mode
            langid.set_languages([lang2code[l] for l in self.added_langs])
            self.code2lang = code2lang
            if not TRANKIT_QUIET:
                print('=' * 50)
                print(f'Trankit is in auto mode!\nAvailable languages: {self.added_langs}')
                print('=' * 50)
        else:
            self.set_active(lang)

    def _setup_config(self, lang):

        # decide whether to run on GPU or CPU
        if self._gpu and torch.cuda.is_available():
            self._use_gpu = True
            self._use_half = True
            self._pin_memory = True
            self._non_blocking = True
            self.master_config.device = torch.device('cuda')
            self._tokbatchsize = 6
            self._tagbatchsize = 24
        elif self._gpu and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self._use_gpu = True
            self._use_half = False
            self._pin_memory = False
            self._non_blocking = False
            self.master_config.device = torch.device('mps')
            self._tokbatchsize = 6
            self._tagbatchsize = 24
        else:
            self._use_gpu = False
            self._use_half = False
            self._pin_memory = False
            self._non_blocking = False
            self.master_config.device = torch.device('cpu')
            self._tokbatchsize = 2
            self._tagbatchsize = 12

        if self._cache_dir is None:
            self.master_config._cache_dir = 'cache/trankit'
        else:
            self.master_config._cache_dir = self._cache_dir

        if not os.path.exists(self.master_config._cache_dir):
            os.makedirs(self.master_config._cache_dir, exist_ok=True)

        tokenizer_cache_dir = os.environ.get("TRANKIT_TOKENIZER_CACHE_DIR") or os.environ.get("TRANSFORMERS_CACHE")
        if tokenizer_cache_dir:
            cache_dir = tokenizer_cache_dir
        else:
            cache_dir = os.path.join(self.master_config._cache_dir, self.master_config.embedding_name)

        disable_hf_transfer = os.environ.get("TRANKIT_DISABLE_HF_TRANSFER_TOKENIZER", "").lower() in ("1", "true", "yes")
        if disable_hf_transfer:
            previous_hf_transfer = os.environ.get("HF_HUB_ENABLE_HF_TRANSFER")
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

        try:
            self.master_config.wordpiece_splitter = XLMRobertaTokenizerFast.from_pretrained(
                self.master_config.embedding_name,
                cache_dir=cache_dir,
            )
        finally:
            if disable_hf_transfer:
                if previous_hf_transfer is None:
                    os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
                else:
                    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = previous_hf_transfer
        self._config = self.master_config
        # Track which language's adapter is loaded for each adapter type
        # This allows caching adapters across inferences for the same language
        if not getattr(self, '_cache_adapters', False):
            self._config.active_adapters = {'tokenizer': None, 'tagger': None, 'ner': None}
        self._config.max_input_length = tbname2max_input_length.get(lang2treebank[lang],
                                                                    400)  # this is for tokenizer only

    def set_auto(self, state):
        assert type(state) == bool
        if state is True:
            print(f'Turning on auto mode for {self.added_langs} ...')
            self.auto_mode = True

            cls_codes = []
            self.code2lang = {}
            for l in self.added_langs:
                if l in extra_lang2code:
                    cls_codes.append(extra_lang2code[l])
                    self.code2lang[extra_lang2code[l]] = l

            langid.set_languages(cls_codes)
            if not TRANKIT_QUIET:
                print('=' * 50)
                print('Trankit is in auto mode!')
                print('=' * 50)
        else:
            self.auto_mode = False
            lang = self.added_langs[0]
            old_lang = getattr(self._config, 'active_lang', None)
            self._config.active_lang = lang
            self.active_lang = lang
            # Only reset adapters if language actually changed
            if old_lang != lang and not self._cache_adapters:
                self._config.active_adapters = {'tokenizer': None, 'tagger': None, 'ner': None}
            self._config.treebank_name = lang2treebank[lang]
            self._config.max_input_length = tbname2max_input_length.get(lang2treebank[lang],
                                                                        400)  # this is for tokenizer only
            if not TRANKIT_QUIET:
                print('=' * 50)
                print('Trankit is in normal mode!')
                print('=' * 50)
                print(f'Active language: {self._config.active_lang}')
                print(f'Available languages: {self.added_langs}')
                print('=' * 50)

    def set_active(self, lang):
        assert not self.auto_mode, 'Cannot set a particular language as active in auto mode.\nPlease consider using Trankit in the normal mode to use this function.'
        assert is_string(lang) and lang in self.added_langs, f'Specified language must be added before being activated.\nCurrent added languages: {self.added_langs}'

        old_lang = getattr(self._config, 'active_lang', None)
        self._config.active_lang = lang
        self.active_lang = lang
        # Only reset adapters if language actually changed
        if old_lang != lang and not self._cache_adapters:
            self._config.active_adapters = {'tokenizer': None, 'tagger': None, 'ner': None}
        self._config.treebank_name = lang2treebank[lang]
        self._config.max_input_length = tbname2max_input_length.get(lang2treebank[lang],
                                                                    400)  # this is for tokenizer only
        if not TRANKIT_QUIET:
            print('=' * 50)
            print(f'Active language: {self._config.active_lang}')
            print('=' * 50)

    def add(self, lang):
        assert is_string(lang) and lang in supported_langs, f'Specified language must be one of the supported languages: {supported_langs}'

        # download saved models
        download(
            cache_dir=self._config._cache_dir,
            language=lang,
            saved_model_version=saved_model_version,  # manually set this to avoid duplicated storage
            embedding_name=self.master_config.embedding_name
        )
        # update vocabs
        treebank_name = lang2treebank[lang]
        with open(os.path.join(self._config._cache_dir, self.master_config.embedding_name,
                               f'{treebank2lang[treebank_name]}/{treebank2lang[treebank_name]}.vocabs.json')) as f:
            vocabs = json.load(f)
            self._config.vocabs[treebank_name] = vocabs
        if lang in langwithner:
            with open(os.path.join(self._config._cache_dir, self.master_config.embedding_name,
                                   f'{lang}/{lang}.ner-vocab.json')) as f:
                self._config.ner_vocabs[lang] = json.load(f)

        self._config.itos[lang][UPOS] = {v: k for k, v in vocabs[UPOS].items()}
        self._config.itos[lang][XPOS] = {v: k for k, v in vocabs[XPOS].items()}
        self._config.itos[lang][FEATS] = {v: k for k, v in vocabs[FEATS].items()}
        self._config.itos[lang][DEPREL] = {v: k for k, v in vocabs[DEPREL].items()}

        # add tokenizer
        self._tokenizer[lang] = TokenizerClassifier(self._config, treebank_name=lang2treebank[lang])
        self._tokenizer[lang].to(self._config.device)
        if self._use_half:
            self._tokenizer[lang].half()
        self._tokenizer[lang].eval()

        # add tagger
        self._tagger[lang] = PosDepClassifier(self._config, treebank_name=lang2treebank[lang])
        self._tagger[lang].to(self._config.device)
        if self._use_half:
            self._tagger[lang].half()
        self._tagger[lang].eval()

        # mwt if available
        treebank_name = lang2treebank[lang]
        if tbname2training_id[treebank_name] % 2 == 1:
            self._mwt_model[lang] = MWTWrapper(self._config, treebank_name=treebank_name, use_gpu=self._use_gpu)

        # lemma
        self._lemma_model[lang] = LemmaWrapper(self._config, treebank_name=treebank_name, use_gpu=self._use_gpu, cpu_lemma=self._cpu_lemma)

        # ner if available
        if lang in langwithner:
            self._ner_model[lang] = NERClassifier(self._config, lang)
            self._ner_model[lang].to(self._config.device)
            if self._use_half:
                self._ner_model[lang].half()
            self._ner_model[lang].eval()

        self.added_langs.append(lang)

    def _load_vocabs(self):
        self._config.vocabs = {}
        self._config.ner_vocabs = {}
        self._config.itos = defaultdict(dict)
        for lang in self.added_langs:
            treebank_name = lang2treebank[lang]
            with open(os.path.join(self._config._cache_dir, self.master_config.embedding_name,
                                   f'{lang}/{lang}.vocabs.json')) as f:
                vocabs = json.load(f)
                self._config.vocabs[treebank_name] = vocabs
            self._config.itos[lang][UPOS] = {v: k for k, v in vocabs[UPOS].items()}
            self._config.itos[lang][XPOS] = {v: k for k, v in vocabs[XPOS].items()}
            self._config.itos[lang][FEATS] = {v: k for k, v in vocabs[FEATS].items()}
            self._config.itos[lang][DEPREL] = {v: k for k, v in vocabs[DEPREL].items()}
            # ner vocabs
            if lang in langwithner:
                with open(os.path.join(self._config._cache_dir, self.master_config.embedding_name,
                                       f'{lang}/{lang}.ner-vocab.json')) as f:
                    self._config.ner_vocabs[lang] = json.load(f)

    def _load_adapter_weights(self, model_name):
        assert model_name in ['tokenizer', 'tagger', 'ner']
        current_lang = self._config.active_lang

        if self._cache_adapters:
            # Per-language adapter slots: zero-cost warm switch
            slot_name = _adapter_slot_name(model_name, current_lang)

            if slot_name not in self._resident_adapters:
                # Guard: check model state as source of truth
                if slot_name in self._embedding_layers.xlmr.adapters_config.adapters:
                    self._resident_adapters.add(slot_name)
                else:
                    self._embedding_layers.xlmr.add_adapter(slot_name, config=self._adapter_config)

                    if model_name == 'tokenizer':
                        pretrained_weights = self._tokenizer[current_lang].pretrained_tokenizer_weights
                    elif model_name == 'tagger':
                        pretrained_weights = self._tagger[current_lang].pretrained_tagger_weights
                    else:
                        pretrained_weights = self._ner_model[current_lang].pretrained_ner_weights

                    self._adapter_loader.load_from_state_dict(
                        pretrained_weights, model_name, load_as=slot_name, start_prefix="xlmr."
                    )
                    # Move new adapter params to model device (and half if needed).
                    # Already-on-device params are no-ops for .to()/.half().
                    self._embedding_layers.xlmr.to(self._config.device)
                    if self._use_half:
                        self._embedding_layers.xlmr.half()
                    self._resident_adapters.add(slot_name)

            # Warm path: skip if already active.
            if self._active_slot != slot_name:
                if _BYPASS_ADAPTER_RESET:
                    # Set adapter config directly, skipping reset_adapter
                    # (which iterates all modules for LoRA resets we don't need).
                    xlmr = self._embedding_layers.xlmr
                    xlmr.adapters_config.active_setup = parse_composition(
                        Stack(slot_name), model_type=xlmr.config.model_type
                    )
                    xlmr.adapters_config.skip_layers = None
                else:
                    self._embedding_layers.xlmr.set_active_adapters(Stack(slot_name))
                self._active_slot = slot_name
        else:
            # Original behavior: copy weights into fixed slot on language change
            cached_lang = self._config.active_adapters.get(model_name)

            if cached_lang != current_lang:
                if model_name == 'tokenizer':
                    pretrained_weights = self._tokenizer[current_lang].pretrained_tokenizer_weights
                elif model_name == 'tagger':
                    pretrained_weights = self._tagger[current_lang].pretrained_tagger_weights
                else:
                    pretrained_weights = self._ner_model[current_lang].pretrained_ner_weights

                self._adapter_loader.load_from_state_dict(
                    pretrained_weights, model_name, start_prefix="xlmr."
                )
                self._config.active_adapters[model_name] = current_lang

            self._embedding_layers.xlmr.set_active_adapters(Stack(model_name))

    def evict_language_adapters(self, lang):
        """Remove per-language adapter slots from XLM-R, freeing device memory.

        Call this when evicting a language from an LRU cache.
        No-op when cache_adapters=False or no adapters for this language are loaded.
        """
        if not self._cache_adapters:
            return
        xlmr = self._embedding_layers.xlmr
        for task in ('tokenizer', 'tagger', 'ner'):
            slot_name = _adapter_slot_name(task, lang)
            if slot_name in self._resident_adapters:
                if slot_name in _active_adapter_names(xlmr):
                    xlmr.set_active_adapters(Stack('tokenizer'))
                    self._active_slot = None  # invalidate: evicted adapter may have been active
                xlmr.delete_adapter(slot_name)
                self._resident_adapters.discard(slot_name)

    def _detect_lang_and_switch(self, text):
        detected_code = langid.classify(text)[0]
        assert detected_code in self.code2lang, f'Detected code "{detected_code}" must be in {self.code2lang.keys()}'

        lang = self.code2lang[detected_code]

        assert is_string(lang) and lang in self.added_langs, f'Specified language must be added before being activated.\nCurrent added languages: {self.added_langs}'

        old_lang = getattr(self._config, 'active_lang', None)
        self._config.active_lang = lang
        self.active_lang = lang
        # Only reset adapters if language actually changed
        if old_lang != lang and not self._cache_adapters:
            self._config.active_adapters = {'tokenizer': None, 'tagger': None, 'ner': None}
        self._config.treebank_name = lang2treebank[lang]
        self._config.max_input_length = tbname2max_input_length.get(lang2treebank[lang],
                                                                    400)  # this is for tokenizer only

    def ssplit(self, in_doc):  # assuming input is a document
        assert is_string(in_doc), 'Input must be a non-empty string.'
        # switch to detected lang if auto mode is on
        if self.auto_mode:
            self._detect_lang_and_switch(text=in_doc)

        eval_batch_size = tbname2tokbatchsize.get(lang2treebank[self.active_lang], self._tokbatchsize)
        # load input text
        config = self._config
        test_set = TokenizeDatasetLive(config, in_doc, max_input_length=tbname2max_input_length.get(
            lang2treebank[self.active_lang], 400))
        test_set.numberize(config.wordpiece_splitter)

        # load weights of tokenizer into the combined model
        self._load_adapter_weights(model_name='tokenizer')

        # make predictions
        wordpiece_pred_labels, wordpiece_ends, paragraph_indexes = [], [], []
        for batch in DataLoader(test_set, batch_size=eval_batch_size,
                                shuffle=False, collate_fn=test_set.collate_fn,
                                pin_memory=self._pin_memory):
            batch = batch_to_device(batch, self._config.device, non_blocking=self._non_blocking)
            wordpiece_reprs = self._embedding_layers.get_tokenizer_inputs(batch)
            predictions = self._tokenizer[self._config.active_lang].predict(batch, wordpiece_reprs)
            wp_pred_labels, wp_ends, para_ids = predictions[0], predictions[1], predictions[2]
            wp_pred_labels = wp_pred_labels.detach().cpu().tolist()

            wordpiece_pred_labels.extend(
                wp_labels[:len(wp_end_positions)] for wp_labels, wp_end_positions in zip(wp_pred_labels, wp_ends)
            )

            wordpiece_ends.extend(wp_ends)
            paragraph_indexes.extend(para_ids)
    
        # mapping
        para_id_to_wp_pred_labels = defaultdict(list)

        for wp_pred_ls, wp_es, p_index in zip(wordpiece_pred_labels, wordpiece_ends,
                                              paragraph_indexes):
            para_id_to_wp_pred_labels[p_index].extend(zip(wp_pred_ls, wp_es))

        # get predictions
        corpus_text = in_doc

        paragraphs = [s for pt in NEWLINE_WHITESPACE_RE.split(corpus_text) if (s := pt.rstrip())]
        all_wp_preds = []
        all_para_texts = []
        all_para_starts = []
        ##############
        global_offset = 0
        for para_index, para_text in enumerate(paragraphs):
            start_char_idx = in_doc.index(para_text, global_offset)
            global_offset = start_char_idx + len(para_text)
            all_para_starts.append(start_char_idx)

            para_wp_preds = [0 for _ in para_text]
            for wp_l, end_position in para_id_to_wp_pred_labels[para_index]:
                para_wp_preds[end_position] = wp_l

            all_wp_preds.append(para_wp_preds)
            all_para_texts.append(para_text)

        ###########################
        sentences = []
        for j in range(len(paragraphs)):
            para_text = all_para_texts[j]
            wp_pred = all_wp_preds[j]
            para_start = all_para_starts[j]

            current_tok = ''
            current_sent = []
            local_position = 0
            for t, wp_p in zip(para_text, wp_pred):
                local_position += 1
                current_tok += t
                if wp_p >= 1:
                    tok = normalize_token(test_set.treebank_name, current_tok, ud_eval=self._ud_eval)
                    assert '\t' not in tok, tok
                    if len(tok) <= 0:
                        current_tok = ''
                        continue
                    additional_info = {DSPAN: (para_start + local_position - len(tok),
                                               para_start + local_position)}
                    current_sent.append((tok, wp_p, additional_info))
                    current_tok = ''
                    if (wp_p == 2 or wp_p == 4):
                        sent_span = (current_sent[0][2][DSPAN][0], current_sent[-1][2][DSPAN][1])
                        sentences.append(
                            {ID: len(sentences) + 1, TEXT: in_doc[sent_span[0]: sent_span[1]],
                             DSPAN: (sent_span[0], sent_span[1])})
                        current_sent = []

            if len(current_tok):
                tok = normalize_token(test_set.treebank_name, current_tok, ud_eval=self._ud_eval)
                assert '\t' not in tok, tok
                if len(tok) > 0:
                    additional_info = {DSPAN: (para_start + local_position - len(tok),
                                               para_start + local_position)}
                    current_sent.append((tok, 2, additional_info))

            if len(current_sent):
                sent_span = (current_sent[0][2][DSPAN][0], current_sent[-1][2][DSPAN][1])
                sentences.append(
                    {ID: len(sentences) + 1, TEXT: in_doc[sent_span[0]: sent_span[1]],
                     DSPAN: (sent_span[0], sent_span[1])})

        return {TEXT: in_doc, SENTENCES: sentences, LANG: self.active_lang}

    def tokenize(self, input, is_sent=False):
        assert is_string(input), 'Input must be a non-empty string.'
        # switch to detected lang if auto mode is on
        if self.auto_mode:
            self._detect_lang_and_switch(text=input)

        if isinstance(input, str) and input.isspace():
            return []
        ori_text = input
        if is_sent:
            return {TEXT: ori_text, TOKENS: self._tokenize_sent(in_sent=input), LANG: self.active_lang}
        else:
            return {TEXT: ori_text, SENTENCES: self._tokenize_doc(in_doc=input), LANG: self.active_lang}

    def _tokenize_sent(self, in_sent):  # assuming input is a sentence
        eval_batch_size = tbname2tokbatchsize.get(lang2treebank[self.active_lang], self._tokbatchsize)
        if self._config.embedding_name == 'xlm-roberta-large':
            eval_batch_size = int(eval_batch_size / 2)

        # load input text
        config = self._config
        test_set = TokenizeDatasetLive(config, in_sent, max_input_length=tbname2max_input_length.get(
            lang2treebank[self.active_lang], 400))
        test_set.numberize(config.wordpiece_splitter)

        # load weights of tokenizer into the combined model
        self._load_adapter_weights(model_name='tokenizer')

        # make predictions
        wordpiece_pred_labels, wordpiece_ends, paragraph_indexes = [], [], []
        with self._autocast():
            for batch in DataLoader(test_set, batch_size=eval_batch_size,
                                    shuffle=False, collate_fn=test_set.collate_fn,
                                    pin_memory=self._pin_memory):
                batch = batch_to_device(batch, self._config.device, non_blocking=self._non_blocking)
                wordpiece_reprs = self._embedding_layers.get_tokenizer_inputs(batch)
                predictions = self._tokenizer[self._config.active_lang].predict(batch, wordpiece_reprs)
                wp_pred_labels, wp_ends, para_ids = predictions[0], predictions[1], predictions[2]
                wp_pred_labels = wp_pred_labels.detach().cpu().tolist()

                wordpiece_pred_labels.extend(
                    wp_labels[:len(wp_end_positions)] for wp_labels, wp_end_positions in zip(wp_pred_labels, wp_ends)
                )

                wordpiece_ends.extend(wp_ends)
                paragraph_indexes.extend(para_ids)
        # mapping
        para_id_to_wp_pred_labels = defaultdict(list)

        for wp_pred_ls, wp_es, p_index in zip(wordpiece_pred_labels, wordpiece_ends,
                                              paragraph_indexes):
            para_id_to_wp_pred_labels[p_index].extend(zip(wp_pred_ls, wp_es))

        # get predictions
        corpus_text = in_sent

        paragraphs = [s for pt in NEWLINE_WHITESPACE_RE.split(corpus_text) if (s := pt.rstrip())]
        all_wp_preds = []
        all_para_texts = []
        all_para_starts = []
        ##############
        global_offset = 0
        for para_index, para_text in enumerate(paragraphs):
            start_char_idx = in_sent.index(para_text, global_offset)
            global_offset = start_char_idx + len(para_text)
            all_para_starts.append(start_char_idx)

            para_wp_preds = [0 for _ in para_text]
            for wp_l, end_position in para_id_to_wp_pred_labels[para_index]:
                para_wp_preds[end_position] = wp_l

            all_wp_preds.append(para_wp_preds)
            all_para_texts.append(para_text)

        ###########################
        tokens = []
        for j in range(len(paragraphs)):
            para_text = all_para_texts[j]
            wp_pred = all_wp_preds[j]
            para_start = all_para_starts[j]

            current_tok = ''
            current_sent = []
            local_position = 0
            for t, wp_p in zip(para_text, wp_pred):
                local_position += 1
                current_tok += t
                if wp_p >= 1:
                    tok = normalize_token(test_set.treebank_name, current_tok, ud_eval=self._ud_eval)
                    assert '\t' not in tok, tok
                    if len(tok) <= 0:
                        current_tok = ''
                        continue
                    additional_info = {'current_len': len(tokens),
                                       SSPAN: (para_start + local_position - len(tok),
                                               para_start + local_position)}
                    current_sent.append((tok, wp_p, additional_info))
                    current_tok = ''
                    if (wp_p == 2 or wp_p == 4):
                        tokens.extend(get_output_sentence(current_sent))
                        current_sent = []

            if len(current_tok):
                tok = normalize_token(test_set.treebank_name, current_tok, ud_eval=self._ud_eval)
                assert '\t' not in tok, tok
                if len(tok) > 0:
                    additional_info = {'current_len': len(tokens),
                                       SSPAN: (para_start + local_position - len(tok),
                                               para_start + local_position)}
                    current_sent.append((tok, 2, additional_info))

            if len(current_sent):
                tokens.extend(get_output_sentence(current_sent))


        # multi-word expansion if required
        if tbname2training_id[self._config.treebank_name] % 2 == 1:
            tokens = self._mwt_expand([{TOKENS: tokens}])[0][TOKENS]


        return tokens

    def _tokenize_doc(self, in_doc):  # assuming input is a document
        eval_batch_size = tbname2tokbatchsize.get(lang2treebank[self.active_lang], self._tokbatchsize)
        if self._config.embedding_name == 'xlm-roberta-large':
            eval_batch_size = int(eval_batch_size / 2)
        # load input text
        config = self._config
        test_set = TokenizeDatasetLive(config, in_doc, max_input_length=tbname2max_input_length.get(
            lang2treebank[self.active_lang], 400))
        test_set.numberize(config.wordpiece_splitter)

        # load weights of tokenizer into the combined model
        self._load_adapter_weights(model_name='tokenizer')

        # make predictions
        wordpiece_pred_labels, wordpiece_ends, paragraph_indexes = [], [], []
        with self._autocast():
            for batch in DataLoader(test_set, batch_size=eval_batch_size,
                                    shuffle=False, collate_fn=test_set.collate_fn,
                                    pin_memory=self._pin_memory):
                batch = batch_to_device(batch, self._config.device, non_blocking=self._non_blocking)
                wordpiece_reprs = self._embedding_layers.get_tokenizer_inputs(batch)
                predictions = self._tokenizer[self._config.active_lang].predict(batch, wordpiece_reprs)
                wp_pred_labels, wp_ends, para_ids = predictions[0], predictions[1], predictions[2]
                wp_pred_labels = wp_pred_labels.detach().cpu().tolist()

                wordpiece_pred_labels.extend(
                    wp_labels[:len(wp_end_positions)] for wp_labels, wp_end_positions in zip(wp_pred_labels, wp_ends)
                )

                wordpiece_ends.extend(wp_ends)
                paragraph_indexes.extend(para_ids)
        # mapping
        para_id_to_wp_pred_labels = defaultdict(list)

        for wp_pred_ls, wp_es, p_index in zip(wordpiece_pred_labels, wordpiece_ends,
                                              paragraph_indexes):
            para_id_to_wp_pred_labels[p_index].extend(zip(wp_pred_ls, wp_es))

        # get predictions
        corpus_text = in_doc

        paragraphs = [s for pt in NEWLINE_WHITESPACE_RE.split(corpus_text) if (s := pt.rstrip())]
        all_wp_preds = []
        all_para_texts = []
        all_para_starts = []
        ##############
        global_offset = 0
        for para_index, para_text in enumerate(paragraphs):
            start_char_idx = in_doc.index(para_text, global_offset)
            global_offset = start_char_idx + len(para_text)
            all_para_starts.append(start_char_idx)

            para_wp_preds = [0 for _ in para_text]
            for wp_l, end_position in para_id_to_wp_pred_labels[para_index]:
                para_wp_preds[end_position] = wp_l

            all_wp_preds.append(para_wp_preds)
            all_para_texts.append(para_text)
        ###########################
        doc = []
        for j in range(len(paragraphs)):
            para_text = all_para_texts[j]
            wp_pred = all_wp_preds[j]
            para_start = all_para_starts[j]

            current_tok = ''
            current_sent = []
            local_position = 0
            for t, wp_p in zip(para_text, wp_pred):
                local_position += 1
                current_tok += t
                if wp_p >= 1:
                    tok = normalize_token(test_set.treebank_name, current_tok, ud_eval=self._ud_eval)
                    assert '\t' not in tok, tok
                    if len(tok) <= 0:
                        current_tok = ''
                        continue
                    additional_info = {DSPAN: (para_start + local_position - len(tok),
                                               para_start + local_position)}
                    current_sent.append((tok, wp_p, additional_info))
                    current_tok = ''
                    if (wp_p == 2 or wp_p == 4):
                        processed_sent = get_output_sentence(current_sent)
                        doc.append({
                            ID: len(doc) + 1,
                            TEXT: in_doc[processed_sent[0][DSPAN][0]: processed_sent[-1][DSPAN][
                                1]],
                            TOKENS: processed_sent,
                            DSPAN: (processed_sent[0][DSPAN][0], processed_sent[-1][DSPAN][1])
                        })
                        current_sent = []

            if len(current_tok):
                tok = normalize_token(test_set.treebank_name, current_tok, ud_eval=self._ud_eval)
                assert '\t' not in tok, tok
                if len(tok) > 0:
                    additional_info = {DSPAN: (para_start + local_position - len(tok),
                                               para_start + local_position)}
                    current_sent.append((tok, 2, additional_info))

            if len(current_sent):
                processed_sent = get_output_sentence(current_sent)
                doc.append({
                    ID: len(doc) + 1,
                    TEXT: in_doc[
                          processed_sent[0][DSPAN][0]: processed_sent[-1][DSPAN][1]],
                    TOKENS: processed_sent,
                    DSPAN: (processed_sent[0][DSPAN][0], processed_sent[-1][DSPAN][1])
                })

        # multi-word expansion if required
        if tbname2training_id[self._config.treebank_name] % 2 == 1:
            doc = self._mwt_expand(doc)

        return doc

    def posdep(self, input, is_sent=False):
        if is_sent:
            assert is_string(input) or is_list_strings(
                input), 'Input must be one of the following:\n(i) A non-empty string.\n(ii) A list of non-empty strings.'

            if is_list_strings(input):
                # switch to detected lang if auto mode is on
                if self.auto_mode:
                    self._detect_lang_and_switch(text=' '.join(input))

                input = [{ID: k + 1, TEXT: w} for k, w in enumerate(input)]
                return {TOKENS: self._posdep_sent(in_sent=input), LANG: self.active_lang}
            else:
                # switch to detected lang if auto mode is on
                if self.auto_mode:
                    self._detect_lang_and_switch(text=input)
                ori_text = input
                return {TEXT: ori_text, TOKENS: self._posdep_sent(in_sent=input), LANG: self.active_lang}

        else:
            assert is_string(input) or is_list_list_strings(
                input), 'Input must be one of the following:\n(i) A non-empty string.\n(ii) A list of lists of non-empty strings.'

            if is_list_list_strings(input):
                # switch to detected lang if auto mode is on
                if self.auto_mode:
                    self._detect_lang_and_switch(text='\n'.join([' '.join(sent) for sent in input]))

                input = [{ID: sid + 1, TOKENS: [{ID: tid + 1, TEXT: w} for tid, w in enumerate(sent)]} for sid, sent in
                         enumerate(input)]
                return {SENTENCES: self._posdep_doc(in_doc=input), LANG: self.active_lang}
            else:
                # switch to detected lang if auto mode is on
                if self.auto_mode:
                    self._detect_lang_and_switch(text=input)

                ori_text = input
                return {TEXT: ori_text, SENTENCES: self._posdep_doc(in_doc=input), LANG: self.active_lang}

    def _posdep_sent(self, in_sent):  # assuming input is a sentence
        if isinstance(in_sent, str):  # input sentence is an untokenized string in this case
            in_sent = self._tokenize_sent(in_sent)
        posdep_sent = [{ID: 1, TOKENS: in_sent}]
        # load outputs of tokenizer
        config = self._config
        test_set = TaggerDatasetLive(
            tokenized_doc=posdep_sent,
            wordpiece_splitter=config.wordpiece_splitter,
            config=config
        )
        test_set.numberize()

        # load weights of tagger into the combined model
        self._load_adapter_weights(model_name='tagger')

        # make predictions
        eval_batch_size = tbname2tagbatchsize.get(self._config.treebank_name, self._tagbatchsize)
        if self._config.embedding_name == 'xlm-roberta-large':
            eval_batch_size = int(eval_batch_size / 3)

        itos = self._config.itos[self._config.active_lang]
        with self._autocast():
            for batch in DataLoader(test_set,
                                    batch_size=eval_batch_size,
                                    shuffle=False, collate_fn=test_set.collate_fn,
                                    pin_memory=self._pin_memory):
                batch = batch_to_device(batch, self._config.device, non_blocking=self._non_blocking)
                batch_size = len(batch.word_num)

                word_reprs, cls_reprs = self._embedding_layers.get_tagger_inputs(batch)
                predictions = self._tagger[self._config.active_lang].predict(batch, word_reprs, cls_reprs)
                # stack upos/xpos/feats on GPU, one .cpu() transfer, unpack on CPU
                tag_stacked = torch.stack([predictions[0], predictions[1], predictions[2]]).detach().cpu().tolist()
                predicted_upos, predicted_xpos, predicted_feats = tag_stacked

                # head, deprel — different dtypes, transfer back-to-back
                predicted_dep = predictions[3]
                dep_unlabeled = predicted_dep[0].cpu().numpy()
                dep_labeled = predicted_dep[1].cpu().numpy()
                sentlens = [l + 1 for l in batch.word_num]
                head_seqs = [chuliu_edmonds_one_root(adj[:l, :l])[1:] for adj, l in
                             zip(dep_unlabeled, sentlens)]  # remove attachment for the root
                deprel_seqs = [
                    [itos[DEPREL][dep_labeled[i][j + 1][h]] for j, h in
                     enumerate(hs)] for
                    i, hs
                    in
                    enumerate(head_seqs)]

                pred_tokens = [[[head_seqs[i][j], deprel_seqs[i][j]] for j in range(sentlens[i] - 1)] for i in
                               range(batch_size)]

                for bid in range(batch_size):
                    sentid = batch.sent_index[bid]
                    for i in range(batch.word_num[bid]):
                        wordid = batch.word_ids[bid][i]

                        # upos
                        test_set.conllu_doc[sentid][wordid][UPOS] = itos[UPOS][predicted_upos[bid][i]]
                        # xpos
                        test_set.conllu_doc[sentid][wordid][XPOS] = itos[XPOS][predicted_xpos[bid][i]]
                        # feats
                        test_set.conllu_doc[sentid][wordid][FEATS] = itos[FEATS][predicted_feats[bid][i]]
                        # head
                        test_set.conllu_doc[sentid][wordid][HEAD] = int(pred_tokens[bid][i][0])
                        # deprel
                        test_set.conllu_doc[sentid][wordid][DEPREL] = pred_tokens[bid][i][1]

        tagged_doc = get_output_doc(posdep_sent, test_set.conllu_doc)

        return tagged_doc[0][TOKENS]

    def _posdep_doc(self, in_doc):  # assuming input is a document
        if isinstance(in_doc, str):  # in_doc is an untokenized string in this case
            in_doc = self._tokenize_doc(in_doc)
        # load outputs of tokenizer
        config = self._config
        test_set = TaggerDatasetLive(
            tokenized_doc=in_doc,
            wordpiece_splitter=config.wordpiece_splitter,
            config=config
        )
        test_set.numberize()

        # load weights of tagger into the combined model
        self._load_adapter_weights(model_name='tagger')

        # make predictions
        eval_batch_size = tbname2tagbatchsize.get(self._config.treebank_name, self._tagbatchsize)
        if self._config.embedding_name == 'xlm-roberta-large':
            eval_batch_size = int(eval_batch_size / 3)

        itos = self._config.itos[self._config.active_lang]
        with self._autocast():
            for batch in DataLoader(test_set,
                                    batch_size=eval_batch_size,
                                    shuffle=False, collate_fn=test_set.collate_fn,
                                    pin_memory=self._pin_memory):
                batch = batch_to_device(batch, self._config.device, non_blocking=self._non_blocking)
                batch_size = len(batch.word_num)

                word_reprs, cls_reprs = self._embedding_layers.get_tagger_inputs(batch)
                predictions = self._tagger[self._config.active_lang].predict(batch, word_reprs, cls_reprs)
                # stack upos/xpos/feats on GPU, one .cpu() transfer, unpack on CPU
                tag_stacked = torch.stack([predictions[0], predictions[1], predictions[2]]).detach().cpu().tolist()
                predicted_upos, predicted_xpos, predicted_feats = tag_stacked

                # head, deprel — different dtypes, transfer back-to-back
                predicted_dep = predictions[3]
                dep_unlabeled = predicted_dep[0].cpu().numpy()
                dep_labeled = predicted_dep[1].cpu().numpy()
                sentlens = [l + 1 for l in batch.word_num]
                head_seqs = [chuliu_edmonds_one_root(adj[:l, :l])[1:] for adj, l in
                             zip(dep_unlabeled, sentlens)]  # remove attachment for the root
                deprel_seqs = [
                    [itos[DEPREL][dep_labeled[i][j + 1][h]] for j, h in
                     enumerate(hs)] for
                    i, hs
                    in
                    enumerate(head_seqs)]

                pred_tokens = [[[head_seqs[i][j], deprel_seqs[i][j]] for j in range(sentlens[i] - 1)] for i in
                               range(batch_size)]

                for bid in range(batch_size):
                    sentid = batch.sent_index[bid]
                    for i in range(batch.word_num[bid]):
                        wordid = batch.word_ids[bid][i]

                        # upos
                        test_set.conllu_doc[sentid][wordid][UPOS] = itos[UPOS][predicted_upos[bid][i]]
                        # xpos
                        test_set.conllu_doc[sentid][wordid][XPOS] = itos[XPOS][predicted_xpos[bid][i]]
                        # feats
                        test_set.conllu_doc[sentid][wordid][FEATS] = itos[FEATS][predicted_feats[bid][i]]
                        # head
                        test_set.conllu_doc[sentid][wordid][HEAD] = int(pred_tokens[bid][i][0])
                        # deprel
                        test_set.conllu_doc[sentid][wordid][DEPREL] = pred_tokens[bid][i][1]

        tagged_doc = get_output_doc(in_doc, test_set.conllu_doc)

        return tagged_doc

    def lemmatize(self, input, is_sent=False):
        if is_sent:
            assert is_string(input) or is_list_strings(
                input), 'Input must be one of the following:\n(i) A non-empty string.\n(ii) A list of non-empty strings.'

            if is_list_strings(input):
                # switch to detected lang if auto mode is on
                if self.auto_mode:
                    self._detect_lang_and_switch(text=' '.join(input))

                input = [{ID: k + 1, TEXT: w} for k, w in enumerate(input)]
                return {TOKENS: self._lemmatize_sent(in_sent=input, obmit_tag=True), LANG: self.active_lang}
            else:
                # switch to detected lang if auto mode is on
                if self.auto_mode:
                    self._detect_lang_and_switch(text=input)

                ori_text = input
                return {TEXT: ori_text, TOKENS: self._lemmatize_sent(in_sent=input, obmit_tag=True), LANG: self.active_lang}

        else:
            assert is_string(input) or is_list_list_strings(
                input), 'Input must be one of the following:\n(i) A non-empty string.\n(ii) A list of lists of non-empty strings.'

            if is_list_list_strings(input):
                # switch to detected lang if auto mode is on
                if self.auto_mode:
                    self._detect_lang_and_switch(text='\n'.join([' '.join(sent) for sent in input]))

                input = [{ID: sid + 1, TOKENS: [{ID: tid + 1, TEXT: w} for tid, w in enumerate(sent)]} for sid, sent in
                         enumerate(input)]
                return {SENTENCES: self._lemmatize_doc(in_doc=input, obmit_tag=True), LANG: self.active_lang}
            else:
                # switch to detected lang if auto mode is on
                if self.auto_mode:
                    self._detect_lang_and_switch(text=input)

                ori_text = input
                return {TEXT: ori_text, SENTENCES: self._lemmatize_doc(in_doc=input, obmit_tag=True), LANG: self.active_lang}

    def _lemmatize_sent(self, in_sent, obmit_tag=False, skip_dict_seq2seq=None):
        if isinstance(in_sent, str):
            in_sent = self._tokenize_sent(in_sent)
            in_sent = self._posdep_sent(in_sent)

        lemmatized_sent = \
            self._lemma_model[self._config.active_lang].predict([{ID: 1, TOKENS: in_sent}], obmit_tag, skip_dict_seq2seq=skip_dict_seq2seq)[0][
                TOKENS]


        return lemmatized_sent

    def _lemmatize_doc(self, in_doc, obmit_tag=False, skip_dict_seq2seq=None):  # assuming input is a document
        if isinstance(in_doc, str):  # in_doc is a raw string in this case
            in_doc = self._tokenize_doc(in_doc)
            in_doc = self._posdep_doc(in_doc)

        lemmatized_doc = self._lemma_model[self._config.active_lang].predict(in_doc, obmit_tag, skip_dict_seq2seq=skip_dict_seq2seq)


        return lemmatized_doc

    def _mwt_expand(self, tokenized_doc):
        expanded_doc = self._mwt_model[self._config.active_lang].predict(tokenized_doc)

        return expanded_doc

    def ner(self, input, is_sent=False):
        if is_sent:
            assert is_string(input) or is_list_strings(
                input), 'Input must be one of the following:\n(i) A non-empty string.\n(ii) A list of non-empty strings.'

            if is_list_strings(input):
                # switch to detected lang if auto mode is on
                if self.auto_mode:
                    self._detect_lang_and_switch(text=' '.join(input))

                assert self.active_lang in langwithner, 'NER module is not available for "{}"'.format(self.active_lang)

                input = [{ID: k + 1, TEXT: w} for k, w in enumerate(input)]
                return {TOKENS: self._ner_sent(in_sent=input), LANG: self.active_lang}
            else:
                # switch to detected lang if auto mode is on
                if self.auto_mode:
                    self._detect_lang_and_switch(text=input)

                assert self.active_lang in langwithner, 'NER module is not available for "{}"'.format(self.active_lang)

                ori_text = input
                return {TEXT: ori_text, TOKENS: self._ner_sent(in_sent=input), LANG: self.active_lang}

        else:
            assert is_string(input) or is_list_list_strings(
                input), 'Input must be one of the following:\n(i) A non-empty string.\n(ii) A list of lists of non-empty strings.'

            if is_list_list_strings(input):
                # switch to detected lang if auto mode is on
                if self.auto_mode:
                    self._detect_lang_and_switch(text='\n'.join([' '.join(sent) for sent in input]))

                assert self.active_lang in langwithner, 'NER module is not available for "{}"'.format(self.active_lang)

                input = [{ID: sid + 1, TOKENS: [{ID: tid + 1, TEXT: w} for tid, w in enumerate(sent)]} for sid, sent in
                         enumerate(input)]
                return {SENTENCES: self._ner_doc(in_doc=input), LANG: self.active_lang}
            else:
                # switch to detected lang if auto mode is on
                if self.auto_mode:
                    self._detect_lang_and_switch(text=input)

                assert self.active_lang in langwithner, 'NER module is not available for "{}"'.format(self.active_lang)

                ori_text = input
                return {TEXT: ori_text, SENTENCES: self._ner_doc(in_doc=input), LANG: self.active_lang}

    def _ner_sent(self, in_sent):  # assuming input is a document
        if isinstance(in_sent, str):
            in_sent = self._tokenize_sent(in_sent)

        dner_doc = [{ID: 1, TOKENS: in_sent}]
        sentences = [[t[TEXT] for t in sentence[TOKENS]] for sentence in dner_doc]
        test_set = NERDatasetLive(
            config=self._config,
            tokenized_sentences=sentences
        )
        test_set.numberize()
        # load ner adapter weights
        self._load_adapter_weights(model_name='ner')
        eval_batch_size = tbname2tagbatchsize.get(self._config.treebank_name, self._tagbatchsize)
        if self._config.embedding_name == 'xlm-roberta-large':
            eval_batch_size = int(eval_batch_size / 3)

        with self._autocast():
            for batch in DataLoader(test_set,
                                    batch_size=eval_batch_size,
                                    shuffle=False, collate_fn=test_set.collate_fn,
                                    pin_memory=self._pin_memory):
                batch = batch_to_device(batch, self._config.device, non_blocking=self._non_blocking)
                word_reprs, cls_reprs = self._embedding_layers.get_tagger_inputs(batch)
                pred_entity_labels = self._ner_model[self._config.active_lang].predict(batch, word_reprs)

                batch_size = len(batch.word_num)

                for bid in range(batch_size):
                    sentid = batch.sent_index[bid]
                    for i in range(batch.word_num[bid]):
                        wordid = batch.word_ids[bid][i]

                        # NER tag
                        dner_doc[sentid][TOKENS][wordid][NER] = pred_entity_labels[bid][i]

        return dner_doc[0][TOKENS]

    def _ner_doc(self, in_doc):  # assuming input is a document
        if isinstance(in_doc, str):
            in_doc = self._tokenize_doc(in_doc)
        dner_doc = in_doc
        sentences = [[t[TEXT] for t in sentence[TOKENS]] for sentence in dner_doc]
        test_set = NERDatasetLive(
            config=self._config,
            tokenized_sentences=sentences
        )
        test_set.numberize()
        # load ner adapter weights
        self._load_adapter_weights(model_name='ner')
        eval_batch_size = tbname2tagbatchsize.get(self._config.treebank_name, self._tagbatchsize)
        if self._config.embedding_name == 'xlm-roberta-large':
            eval_batch_size = int(eval_batch_size / 3)

        with self._autocast():
            for batch in DataLoader(test_set,
                                    batch_size=eval_batch_size,
                                    shuffle=False, collate_fn=test_set.collate_fn,
                                    pin_memory=self._pin_memory):
                batch = batch_to_device(batch, self._config.device, non_blocking=self._non_blocking)
                word_reprs, cls_reprs = self._embedding_layers.get_tagger_inputs(batch)
                pred_entity_labels = self._ner_model[self._config.active_lang].predict(batch, word_reprs)

                batch_size = len(batch.word_num)

                for bid in range(batch_size):
                    sentid = batch.sent_index[bid]
                    for i in range(batch.word_num[bid]):
                        wordid = batch.word_ids[bid][i]

                        # NER tag
                        dner_doc[sentid][TOKENS][wordid][NER] = pred_entity_labels[bid][i]

        return dner_doc

    def batch(self, docs, skip_dict_seq2seq=None):
        """Process multiple documents with stage-level batching for higher throughput.

        All documents must be in the pipeline's current active language.
        Returns a list of result dicts, same format as calling pipeline(text).
        """
        from .batch_pipeline import batch_process
        return batch_process(self, docs, skip_dict_seq2seq=skip_dict_seq2seq)

    def __call__(self, input, is_sent=False, skip_dict_seq2seq=None):
        if is_sent:
            assert is_string(input) or is_list_strings(
                input), 'Input must be one of the following:\n(i) A non-empty string.\n(ii) A list of non-empty strings.'

            if is_list_strings(input):
                # switch to detected lang if auto mode is on
                if self.auto_mode:
                    self._detect_lang_and_switch(text=' '.join(input))

                tokenized_sent = [{ID: k + 1, TEXT: w} for k, w in enumerate(input)]
                tagged_sent = self._posdep_sent(tokenized_sent)
                out = self._lemmatize_sent(tagged_sent, skip_dict_seq2seq=skip_dict_seq2seq)
                if self._config.active_lang in langwithner:  # ner if possible
                    out = self._ner_sent(out)
                final = {TOKENS: out, LANG: self.active_lang}
            else:
                # switch to detected lang if auto mode is on
                if self.auto_mode:
                    self._detect_lang_and_switch(text=input)

                ori_text = input
                # Inline tagger flow so we can reuse the dataset for NER
                tokenized_sent = self._tokenize_sent(input)
                posdep_sent = [{ID: 1, TOKENS: tokenized_sent}]

                config = self._config
                tagger_test_set = TaggerDatasetLive(
                    tokenized_doc=posdep_sent,
                    wordpiece_splitter=config.wordpiece_splitter,
                    config=config
                )
                tagger_test_set.numberize()

                self._load_adapter_weights(model_name='tagger')

                eval_batch_size = tbname2tagbatchsize.get(self._config.treebank_name, self._tagbatchsize)
                if self._config.embedding_name == 'xlm-roberta-large':
                    eval_batch_size = int(eval_batch_size / 3)

                itos = self._config.itos[self._config.active_lang]
                with self._autocast():
                    for batch in DataLoader(tagger_test_set,
                                            batch_size=eval_batch_size,
                                            shuffle=False, collate_fn=tagger_test_set.collate_fn,
                                            pin_memory=self._pin_memory):
                        batch = batch_to_device(batch, self._config.device, non_blocking=self._non_blocking)
                        batch_size = len(batch.word_num)

                        word_reprs, cls_reprs = self._embedding_layers.get_tagger_inputs(batch)
                        predictions = self._tagger[self._config.active_lang].predict(batch, word_reprs, cls_reprs)
                        # stack upos/xpos/feats on GPU, one .cpu() transfer, unpack on CPU
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

                tagged_doc = get_output_doc(posdep_sent, tagger_test_set.conllu_doc)
                tagged_sent = tagged_doc[0][TOKENS]
                out = self._lemmatize_sent(tagged_sent, skip_dict_seq2seq=skip_dict_seq2seq)

                if self._config.active_lang in langwithner:
                    has_mwt = tbname2training_id[self._config.treebank_name] % 2 == 1
                    if has_mwt:
                        # MWT changes word structure; fall back to standard NER
                        out = self._ner_sent(out)
                    else:
                        # Reuse tagger dataset for NER instead of re-tokenizing
                        dner_doc = [{ID: 1, TOKENS: out}]
                        ner_test_set = NERDatasetLive.from_tagger_data(self._config, tagger_test_set)

                        self._load_adapter_weights(model_name='ner')

                        with self._autocast():
                            for batch in DataLoader(ner_test_set,
                                                    batch_size=eval_batch_size,
                                                    shuffle=False, collate_fn=ner_test_set.collate_fn,
                                                    pin_memory=self._pin_memory):
                                batch = batch_to_device(batch, self._config.device, non_blocking=self._non_blocking)
                                word_reprs, cls_reprs = self._embedding_layers.get_tagger_inputs(batch)
                                pred_entity_labels = self._ner_model[self._config.active_lang].predict(batch, word_reprs)

                                batch_size = len(batch.word_num)
                                for bid in range(batch_size):
                                    sentid = batch.sent_index[bid]
                                    for i in range(batch.word_num[bid]):
                                        wordid = batch.word_ids[bid][i]
                                        dner_doc[sentid][TOKENS][wordid][NER] = pred_entity_labels[bid][i]

                        out = dner_doc[0][TOKENS]

                final = {TEXT: ori_text, TOKENS: out, LANG: self.active_lang}
        else:
            assert is_string(input) or is_list_list_strings(
                input), 'Input must be one of the following:\n(i) A non-empty string.\n(ii) A list of lists of non-empty strings.'

            if is_list_list_strings(input):
                # switch to detected lang if auto mode is on
                if self.auto_mode:
                    self._detect_lang_and_switch(text='\n'.join([' '.join(sent) for sent in input]))

                input = [{ID: sid + 1, TOKENS: [{ID: tid + 1, TEXT: w} for tid, w in enumerate(sent)]} for sid, sent in
                         enumerate(input)]
                tagged_doc = self._posdep_doc(input)
                out = self._lemmatize_doc(tagged_doc, skip_dict_seq2seq=skip_dict_seq2seq)
                if self._config.active_lang in langwithner:  # ner if possible
                    out = self._ner_doc(out)
                final = {SENTENCES: out, LANG: self.active_lang}
            else:
                # switch to detected lang if auto mode is on
                if self.auto_mode:
                    self._detect_lang_and_switch(text=input)

                ori_text = input
                # Inline tagger flow so we can reuse the dataset for NER
                in_doc = self._tokenize_doc(in_doc=input)
                config = self._config
                tagger_test_set = TaggerDatasetLive(
                    tokenized_doc=in_doc,
                    wordpiece_splitter=config.wordpiece_splitter,
                    config=config
                )
                tagger_test_set.numberize()

                self._load_adapter_weights(model_name='tagger')

                eval_batch_size = tbname2tagbatchsize.get(self._config.treebank_name, self._tagbatchsize)
                if self._config.embedding_name == 'xlm-roberta-large':
                    eval_batch_size = int(eval_batch_size / 3)

                itos = self._config.itos[self._config.active_lang]
                with self._autocast():
                    for batch in DataLoader(tagger_test_set,
                                            batch_size=eval_batch_size,
                                            shuffle=False, collate_fn=tagger_test_set.collate_fn,
                                            pin_memory=self._pin_memory):
                        batch = batch_to_device(batch, self._config.device, non_blocking=self._non_blocking)
                        batch_size = len(batch.word_num)

                        word_reprs, cls_reprs = self._embedding_layers.get_tagger_inputs(batch)
                        predictions = self._tagger[self._config.active_lang].predict(batch, word_reprs, cls_reprs)
                        # stack upos/xpos/feats on GPU, one .cpu() transfer, unpack on CPU
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

                tagged_doc = get_output_doc(in_doc, tagger_test_set.conllu_doc)
                out = self._lemmatize_doc(tagged_doc, skip_dict_seq2seq=skip_dict_seq2seq)

                if self._config.active_lang in langwithner:
                    has_mwt = tbname2training_id[self._config.treebank_name] % 2 == 1
                    if has_mwt:
                        # MWT changes word structure; fall back to standard NER
                        out = self._ner_doc(out)
                    else:
                        # Reuse tagger dataset for NER instead of re-tokenizing
                        dner_doc = out
                        ner_test_set = NERDatasetLive.from_tagger_data(self._config, tagger_test_set)

                        self._load_adapter_weights(model_name='ner')

                        with self._autocast():
                            for batch in DataLoader(ner_test_set,
                                                    batch_size=eval_batch_size,
                                                    shuffle=False, collate_fn=ner_test_set.collate_fn,
                                                    pin_memory=self._pin_memory):
                                batch = batch_to_device(batch, self._config.device, non_blocking=self._non_blocking)
                                word_reprs, cls_reprs = self._embedding_layers.get_tagger_inputs(batch)
                                pred_entity_labels = self._ner_model[self._config.active_lang].predict(batch, word_reprs)

                                batch_size = len(batch.word_num)
                                for bid in range(batch_size):
                                    sentid = batch.sent_index[bid]
                                    for i in range(batch.word_num[bid]):
                                        wordid = batch.word_ids[bid][i]
                                        dner_doc[sentid][TOKENS][wordid][NER] = pred_entity_labels[bid][i]

                        out = dner_doc

                final = {TEXT: ori_text, SENTENCES: out, LANG: self.active_lang}
        return final

    def _conllu_predict(self, text_fpath):
        print('Running the pipeline on device={}'.format(self._config.device))
        with open(text_fpath) as f:
            raw_text = f.read()
        print('Beginning tokenization')
        tokenized_doc = self._tokenize_doc(raw_text)
        print('Beginning pos tagging and dependency parsing')
        tagged_doc = self._posdep_doc(tokenized_doc)
        print('Beginning lemmatization')
        lemmatized_doc = self._lemmatize_doc(tagged_doc)
        conllu_doc = []
        for sentence in lemmatized_doc:
            conllu_sentence = []
            for token in sentence[TOKENS]:
                if type(token[ID]) == int or len(token[ID]) == 1:
                    conllu_sentence.append(token)
                else:
                    conllu_sentence.append(token)
                    for word in token[EXPANDED]:
                        conllu_sentence.append(word)
            conllu_doc.append(conllu_sentence)

        pred_lemma_fpath = text_fpath + '.pred'
        CoNLL.dict2conll(conllu_doc, pred_lemma_fpath)
        return pred_lemma_fpath
