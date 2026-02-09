import os

from .base_utils import *
from .tokenizer_batching import batched_tokenize_pseudo_tokens

NEWLINE_WHITESPACE_RE = re.compile(r'\n\s*\n')

_fastpath_fallback_count = 0
NUMERIC_RE = re.compile(r'^([\d]+[,\.]*)+$')
WHITESPACE_RE = re.compile(r'\s')
PARAGRAPH_BREAK = re.compile(r'\n\s*\n')

PUNCTUATION = re.compile(
    r'''["’'\(\)\[\]\{\}<>:\,‒–—―…!\.«»\-‐\?‘’“”;/⁄␠·&@\*\\•\^¤¢\$€£¥₩₪†‡°¡¿¬\#№%‰‱¶′§~¨_\|¦⁂☞∴‽※"]''')


def normalize_input(input):
    tmp = input.lstrip()
    lstrip_offset = len(input) - len(input.lstrip())
    return tmp, lstrip_offset


def get_start_char_idx(substring, text):
    start_char_idx = text.index(substring)
    text = text[start_char_idx + len(substring):]
    return text, start_char_idx


def split_to_substrings(sent_text):
    tokens_by_space = sent_text.split()
    substrings = []
    for token in tokens_by_space:
        if len(PUNCTUATION.findall(token)) > 0:
            tmp = ''
            for char in token:
                if PUNCTUATION.match(char):
                    if tmp != '':
                        substrings.append(tmp)
                        tmp = ''
                    substrings.append(char)
                else:
                    tmp += char
            if tmp != '':
                substrings.append(tmp)
        else:
            substrings.append(token)

    return substrings


def get_startchar(word, text):
    start_char_idx = 0
    for k in range(len(text)):
        if len(text[k].strip()) > 0:
            start_char_idx = k
            break
    text = text[start_char_idx + len(word):]
    return text, start_char_idx


def get_character_locations(string_units, text):
    tmp_text = text
    offset = 0
    end_positions = []
    for str_unit in string_units:
        tmp_text, start_position = get_startchar(str_unit, tmp_text)
        start_position += offset
        end_position = start_position + len(str_unit) - 1
        end_positions.append(end_position)
        offset = start_position + len(str_unit)
    return end_positions


def get_mapping_wp_character_to_or_character(wordpiece_splitter, wp_single_string, or_single_string):
    if not or_single_string:
        return {}
    if not wordpiece_splitter.is_fast:
        raise TypeError(
            f"get_mapping_wp_character_to_or_character requires a fast tokenizer "
            f"(got {type(wordpiece_splitter).__name__})"
        )
    chars = list(or_single_string)
    enc = wordpiece_splitter(chars, is_split_into_words=True, add_special_tokens=False, verbose=False)
    wids = enc.word_ids()
    if wids is None:
        raise RuntimeError(
            "word_ids() returned None — is_split_into_words may not be supported"
        )
    all_tokens = wordpiece_splitter.convert_ids_to_tokens(enc.input_ids)

    # Build per-character converted strings (same logic as old code)
    char_converted = [''] * len(chars)
    for pos, wid in enumerate(wids):
        if wid is not None and all_tokens[pos] != '▁':
            tok = all_tokens[pos]
            if tok.startswith('▁'):
                tok = tok[1:]
            char_converted[wid] += tok

    # Build the mapping
    wp_char_to_or_char = {}
    wp_offset = 0
    for char_id, converted in enumerate(char_converted):
        for _ in converted:
            wp_char_to_or_char[wp_offset] = char_id
            wp_offset += 1
    return wp_char_to_or_char


def _compute_wordpiece_mapping_fast(pseudo_tokens, group_pieces, sent_labels, character_locations,
                                    sent_position_in_paragraph):
    """Compute wordpiece labels and ends directly from pseudo-token structure.

    Returns (flat_wordpiece_labels, flat_wordpiece_ends) on success, or None on any
    invariant failure (index OOB, length mismatch), signaling caller to fall back.
    """
    try:
        flat_wordpiece_labels = []
        flat_wordpiece_ends = []
        # Track position in single_original_string (sent_text with whitespace stripped)
        orig_offset = 0
        for pt_idx, (pseudo_token, pieces) in enumerate(zip(pseudo_tokens, group_pieces)):
            stripped_pt = pseudo_token.strip()
            if not pieces:
                orig_offset += len(stripped_pt)
                continue
            # Each piece covers a substring of stripped_pt
            piece_offset = 0
            for piece in pieces:
                if piece.startswith('▁'):
                    str_form = piece[1:]
                else:
                    str_form = piece
                end_char_in_piece = piece_offset + len(str_form) - 1
                end_char_in_orig = orig_offset + end_char_in_piece
                location_in_sentence = character_locations[end_char_in_orig]
                wp_label = int(sent_labels[location_in_sentence])
                wp_end = sent_position_in_paragraph + location_in_sentence
                flat_wordpiece_labels.append(wp_label)
                flat_wordpiece_ends.append(wp_end)
                piece_offset += len(str_form)
            orig_offset += len(stripped_pt)
        return flat_wordpiece_labels, flat_wordpiece_ends
    except (IndexError, KeyError):
        return None


def wordpiece_tokenize_from_raw_text(wordpiece_splitter, sent_text, sent_labels, sent_position_in_paragraph,
                                     treebank_name, fast_path=False):
    if 'Chinese' in treebank_name or 'Japanese' in treebank_name:
        pseudo_tokens = [c for c in sent_text]  # characters as pseudo tokens
    else:
        if treebank_name == 'UD_Urdu-UDTB':
            sent_text = sent_text.replace('۔', '.')
        elif treebank_name == 'UD_Uyghur-UDT':
            sent_text = sent_text.replace('-', '،')
        pseudo_tokens = split_to_substrings(sent_text)
    end_pids = set()
    group_pieces = batched_tokenize_pseudo_tokens(wordpiece_splitter, pseudo_tokens)
    flat_wordpieces = []
    for group in group_pieces:
        if len(group) > 0:
            for p in group:
                pid = len(flat_wordpieces)
                flat_wordpieces.append((p, pid))
            end_pids.add(len(flat_wordpieces) - 1)

    # Map each non-whitespace character position in sent_text to its index.
    # Equivalent to get_character_locations([c for c in stripped], sent_text)
    # when all string_units are single characters, but O(n) single-pass.
    character_locations = [k for k, c in enumerate(sent_text) if c.strip()]

    fast_result = None
    if fast_path and os.environ.get('TRANKIT_TOKENIZER_FASTPATH') != '0':
        fast_result = _compute_wordpiece_mapping_fast(
            pseudo_tokens, group_pieces, sent_labels, character_locations, sent_position_in_paragraph
        )

    debug_parity = os.environ.get('TRANKIT_TOKENIZER_FASTPATH_DEBUG') == '1'

    if fast_result is not None and not debug_parity:
        flat_wordpiece_labels, flat_wordpiece_ends = fast_result
    else:
        # Slow path: character-level mapping
        single_original_string = ''.join([c.strip() for c in sent_text])
        single_wordpiece_string = ''.join(
            [p if not p.startswith('▁') else p.lstrip('▁') for p, pid in flat_wordpieces])
        wp_character_2_or_character = get_mapping_wp_character_to_or_character(
            wordpiece_splitter, single_wordpiece_string, single_original_string)

        flat_wordpiece_labels = []
        flat_wordpiece_ends = []
        offset = 0
        for wordpiece, _ in flat_wordpieces:
            if wordpiece.startswith('▁'):
                str_form = wordpiece[1:]
            else:
                str_form = wordpiece
            end_char = offset + len(str_form) - 1
            ori_char = wp_character_2_or_character[end_char]
            location_in_sentence = character_locations[ori_char]
            wp_label = int(sent_labels[location_in_sentence])
            wp_end = sent_position_in_paragraph + location_in_sentence
            flat_wordpiece_labels.append(wp_label)
            flat_wordpiece_ends.append(wp_end)
            offset = end_char + 1

        if debug_parity and fast_result is not None:
            fast_labels, fast_ends = fast_result
            assert flat_wordpiece_labels == fast_labels, (
                f"Fast path label mismatch:\n"
                f"  slow={flat_wordpiece_labels}\n  fast={fast_labels}\n"
                f"  sent_text={sent_text!r}"
            )
            assert flat_wordpiece_ends == fast_ends, (
                f"Fast path ends mismatch:\n"
                f"  slow={flat_wordpiece_ends}\n  fast={fast_ends}\n"
                f"  sent_text={sent_text!r}"
            )

        if fast_path and fast_result is None:
            global _fastpath_fallback_count
            _fastpath_fallback_count += 1

    return flat_wordpieces, flat_wordpiece_labels, flat_wordpiece_ends, end_pids


def split_to_sentences(paragraph_text, charlabels):
    sent_text = ''
    sent_labels = ''

    sentences = []
    start = 0

    for k in range(len(charlabels)):
        sent_text += paragraph_text[k]
        sent_labels += charlabels[k]

        if charlabels[k] == '2' or charlabels[k] == '4':
            end = k  # (start, end) local position in REFURBISHED paragraph (REFURBISHED means the \newline characters are removed from a paragraph text
            sentences.append((sent_text, sent_labels, start, end))
            start = end + 1
            sent_text = ''
            sent_labels = ''

    if len(sentences) > 0:  # case: train data
        # a paragraph not always ends with a 2 or 4 label
        if not (len(sent_text) == 0 and len(sent_labels) == 0):
            sentences.append(
                (sent_text, sent_labels, start, len(paragraph_text) - 1))
    else:
        sentences = [(paragraph_text, charlabels, 0, len(paragraph_text) - 1)]
    return sentences


def split_to_subsequences(wordpieces, wordpiece_labels, wordpiece_ends, end_piece_ids,
                          max_input_length):
    subsequences = []
    subseq = [[], [], []]

    for wp_wpid, wl, we in zip(wordpieces, wordpiece_labels, wordpiece_ends):
        wp, wpid = wp_wpid
        subseq[0].append((wp, wpid))
        subseq[1].append(wl)
        subseq[2].append(we)
        if wpid in end_piece_ids and len(subseq[0]) >= max_input_length - 10:
            subsequences.append((subseq[0], subseq[1], subseq[2], end_piece_ids))

            subseq = [[], [], []]

    if len(subseq[0]) > 0:
        subsequences.append((subseq[0], subseq[1], subseq[2], end_piece_ids))
    return subsequences


def charlevel_format_to_wordpiece_format(wordpiece_splitter, max_input_length, plaintext, treebank_name,
                                         char_labels_output_fpath=None, fast_path=False):
    if char_labels_output_fpath is not None:
        with open(char_labels_output_fpath) as f:
            corpus_labels = ''.join(f.readlines()).rstrip()
    else:
        corpus_labels = '\n\n'.join('0' * len(pt.rstrip()) for pt in NEWLINE_WHITESPACE_RE.split(plaintext))

    data = [
        {'text': s, 'charlabels': pc}
        for pt, pc in zip(NEWLINE_WHITESPACE_RE.split(plaintext), NEWLINE_WHITESPACE_RE.split(corpus_labels))
        if (s := pt.rstrip())
    ]

    wordpiece_examples = []
    kept_tokens = 0
    total_tokens = 0
    for paragraph_index, paragraph in enumerate(data):
        paragraph_text = paragraph['text']
        paragraph_labels = paragraph['charlabels']
        # split to sentences
        sentences = split_to_sentences(paragraph_text, paragraph_labels)
        tmp_examples = []
        for sent in sentences:
            sent_text, sent_labels, sent_start, sent_end = sent
            wordpieces, wordpiece_labels, wordpiece_ends, end_piece_ids = wordpiece_tokenize_from_raw_text(
                wordpiece_splitter, sent_text,
                sent_labels, sent_start,
                treebank_name, fast_path=fast_path)
            kept_tokens += sum(1 for x in wordpiece_labels if x != 0)
            total_tokens += sum(1 for x in sent_labels if x != '0')
            if len(wordpieces) <= max_input_length - 2:  # minus 2: reserved for <s> and </s>
                tmp_examples.append((wordpieces, wordpiece_labels, wordpiece_ends, end_piece_ids))
            else:
                subsequences = split_to_subsequences(wordpieces, wordpiece_labels, wordpiece_ends,
                                                     end_piece_ids,
                                                     max_input_length)
                for subseq in subsequences:
                    tmp_examples.append(subseq)
        # merge consecutive sentences/subsequences
        new_example = [[], [], []]
        for example in tmp_examples:
            if len(new_example[0]) + len(example[0]) > max_input_length - 2:
                num_extra_wordpieces = min(max_input_length - 2 - len(new_example[0]), len(example[0]))
                end_piece_ids = example[-1]
                takeout_position = 0
                for tmp_id in range(num_extra_wordpieces):
                    wp, wpid = example[0][tmp_id]
                    if wpid in end_piece_ids:
                        takeout_position = tmp_id + 1
                num_extra_wordpieces = takeout_position
                new_example[0] += example[0][: num_extra_wordpieces]
                new_example[1] += example[1][: num_extra_wordpieces]
                new_example[2] += example[2][: num_extra_wordpieces]
                wordpiece_examples.append(
                    ([wp for wp, wpid in new_example[0]], new_example[1], new_example[2],
                     paragraph_index))
                # start new example
                new_example = [[], [], []]

            new_example[0] += example[0]
            new_example[1] += example[1]
            new_example[2] += example[2]
        if len(new_example[0]) > 0:
            wordpiece_examples.append(
                ([wp for wp, wpid in new_example[0]], new_example[1], new_example[2], paragraph_index))

    final_examples = []
    for wp_example in wordpiece_examples:
        wordpieces, wordpiece_labels, wordpiece_ends, paragraph_index = wp_example
        final_examples.append({
            'wordpieces': wordpieces,
            'wordpiece_labels': wordpiece_labels,
            'wordpiece_ends': wordpiece_ends,
            'paragraph_index': paragraph_index
        })

    if fast_path and os.environ.get('TRANKIT_TOKENIZER_FASTPATH_STATS') == '1':
        print(f'[tokenizer fast path] fallback count: {_fastpath_fallback_count}')

    return final_examples


def conllu_to_charlevel_format(plaintext_file, conllu_file, char_labels_output_fpath):
    '''
    Adapted from https://github.com/stanfordnlp/stanza/blob/master/stanza/utils/prepare_tokenizer_data.py
    Date: 2021/01/11
    '''
    with open(plaintext_file, 'r') as f:
        corpus_text = ''.join(f.readlines())

    ensure_dir(os.path.abspath(os.path.join(char_labels_output_fpath, '..')))
    output = open(char_labels_output_fpath, 'w')

    index = 0  

    def is_para_break(index, text):
        if text[index] == '\n':
            para_break = PARAGRAPH_BREAK.match(text[index:])
            if para_break:
                break_len = len(para_break.group(0))
                return True, break_len
        return False, 0

    def find_next_word(index, text, word, output):
        idx = 0
        word_sofar = ''
        yeah = False
        while index < len(text) and idx < len(word):
            para_break, break_len = is_para_break(index, text)
            if para_break:
                if len(word_sofar) > 0:
                    word_sofar = ''

                output.write('\n\n')
                index += break_len - 1
            elif re.match(r'^\s$', text[index]) and not re.match(r'^\s$', word[idx]):
                word_sofar += text[index]
            else:
                word_sofar += text[index]
                idx += 1
            index += 1
        return index, word_sofar

    with open(conllu_file, 'r') as f:
        buf = ''
        mwtbegin = 0
        mwtend = -1
        last_comments = ""
        for line in f:
            line = line.strip()
            if len(line):
                if line[0] == "#":
                    if len(last_comments) == 0:
                        last_comments = line
                    continue

                line = line.split('\t')
                if '.' in line[0]:
                    continue

                word = line[1]
                if '-' in line[0]:
                    mwtbegin, mwtend = [int(x) for x in line[0].split('-')]
                elif mwtbegin <= int(line[0]) < mwtend:
                    continue
                elif int(line[0]) == mwtend:
                    mwtbegin = 0
                    mwtend = -1
                    continue

                if len(buf):
                    output.write(buf)
                index, word_found = find_next_word(index, corpus_text, word, output)
                buf = '0' * (len(word_found) - 1) + ('1' if '-' not in line[0] else '3')
            else:
                if len(buf):
                    assert int(buf[-1]) >= 1
                    output.write(buf[:-1] + '{}'.format(int(buf[-1]) + 1))
                    buf = ''
                last_comments = ''
    output.close()
