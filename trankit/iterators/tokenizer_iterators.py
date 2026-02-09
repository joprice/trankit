from . import *

# for sents
instance_fields = [
    'paragraph_index',
    'wordpieces', 'wordpiece_labels', 'wordpiece_ends',
    'piece_idxs', 'attention_masks', 'token_type_idxs',
    'wordpiece_num'
]

batch_fields = [
    'paragraph_index',
    'wordpieces', 'wordpiece_labels', 'wordpiece_ends',
    'piece_idxs', 'attention_masks', 'token_type_idxs',
    'wordpiece_num'
]

Instance = namedtuple('Instance', field_names=instance_fields)

Batch = namedtuple('Batch', field_names=batch_fields)


class TokenizeDatasetLive(Dataset):
    def __init__(self, config, raw_text, max_input_length=512):
        self.config = config
        self.max_input_length = max_input_length
        self.treebank_name = config.treebank_name
        self.raw_text = raw_text

        self.data = []
        self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def load_data(self):
        self.data = charlevel_format_to_wordpiece_format(
            wordpiece_splitter=self.config.wordpiece_splitter,
            max_input_length=self.max_input_length,
            plaintext=self.raw_text,
            treebank_name=self.config.treebank_name,
            fast_path=True
        )

    def numberize(self, wordpiece_splitter):  # wordpiece tokenizer
        data = []
        for inst in self.data:
            wordpieces = inst['wordpieces']
            wordpiece_labels = inst['wordpiece_labels']
            wordpiece_ends = inst['wordpiece_ends']
            paragraph_index = inst['paragraph_index']
            # Pad word pieces with special tokens
            piece_idxs = encode_pieces(wordpiece_splitter, wordpieces, self.max_input_length)
            attn_masks = [1] * len(piece_idxs)

            # token type idxs
            token_type_idxs = [-100 if piece_id >= len(wordpieces) else wordpiece_labels[piece_id] for piece_id in
                               range(len(piece_idxs) - 2)]

            instance = Instance(
                paragraph_index=paragraph_index,
                wordpieces=wordpieces,
                wordpiece_labels=wordpiece_labels,
                wordpiece_ends=wordpiece_ends,
                piece_idxs=piece_idxs,
                attention_masks=attn_masks,
                token_type_idxs=token_type_idxs,
                wordpiece_num=len(wordpieces)
            )
            data.append(instance)
        self.data = data

    def collate_fn(self, batch):
        batch_paragraph_index = []
        batch_wordpieces = []
        batch_wordpiece_labels = []
        batch_wordpiece_ends = []
        batch_wordpiece_num = []

        for inst in batch:
            batch_paragraph_index.append(inst.paragraph_index)
            batch_wordpieces.append(inst.wordpieces)
            batch_wordpiece_labels.append(inst.wordpiece_labels)
            batch_wordpiece_ends.append(inst.wordpiece_ends)
            batch_wordpiece_num.append(inst.wordpiece_num)

        bs = len(batch)
        raw_max_wp = max(len(inst.piece_idxs) for inst in batch)
        if raw_max_wp < 2:
            raise RuntimeError(
                f"Tokenizer batch has max piece_idxs length {raw_max_wp}, expected >= 2 (BOS+EOS)"
            )
        max_wp = pad_to_bucket(raw_max_wp)

        piece_idxs = np.zeros((bs, max_wp), dtype=np.int64)
        attn_masks = np.zeros((bs, max_wp), dtype=np.int64)
        token_type_idxs = np.full((bs, max_wp - 2), -100, dtype=np.int64)

        for i, inst in enumerate(batch):
            n_pi = len(inst.piece_idxs)
            piece_idxs[i, :n_pi] = inst.piece_idxs
            attn_masks[i, :n_pi] = inst.attention_masks
            n_tt = len(inst.token_type_idxs)
            token_type_idxs[i, :n_tt] = inst.token_type_idxs

        return Batch(
            paragraph_index=batch_paragraph_index,
            wordpieces=batch_wordpieces,
            wordpiece_labels=batch_wordpiece_labels,
            wordpiece_ends=batch_wordpiece_ends,
            piece_idxs=torch.from_numpy(piece_idxs),
            attention_masks=torch.from_numpy(attn_masks),
            token_type_idxs=torch.from_numpy(token_type_idxs),
            wordpiece_num=torch.tensor(batch_wordpiece_num, dtype=torch.long),
        )


class TokenizeDataset(Dataset):
    def __init__(self, config, txt_fpath, conllu_fpath, evaluate=False):
        self.config = config
        self.evaluate = evaluate

        self.plaintext_file = txt_fpath
        self.conllu_file = conllu_fpath

        self.treebank_name = config.treebank_name
        self.char_labels_output_fpath = os.path.join(self.config._save_dir, os.path.basename(txt_fpath) + '.character')

        self.data = []
        self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def load_data(self):
        if not self.evaluate:
            conllu_to_charlevel_format(
                plaintext_file=self.plaintext_file,
                conllu_file=self.conllu_file,
                char_labels_output_fpath=self.char_labels_output_fpath
            )
            with open(self.plaintext_file, 'r') as f:
                plaintext = ''.join(f.readlines())
            self.data = charlevel_format_to_wordpiece_format(
                wordpiece_splitter=self.config.wordpiece_splitter,
                max_input_length=self.config.max_input_length,
                plaintext=plaintext,
                treebank_name=self.treebank_name,
                char_labels_output_fpath=self.char_labels_output_fpath
            )
        else:
            with open(self.plaintext_file, 'r') as f:
                plaintext = ''.join(f.readlines())
            self.data = charlevel_format_to_wordpiece_format(
                wordpiece_splitter=self.config.wordpiece_splitter,
                max_input_length=self.config.max_input_length,
                plaintext=plaintext,
                treebank_name=self.treebank_name,
                char_labels_output_fpath=None
            )

        print('Loaded {} examples from: \n(i) {}\n(ii) {}'.format(len(self), self.plaintext_file, self.conllu_file))
        print('-' * 50)

    def numberize(self):  # tokenizer: wordpiece tokenizer
        data = []
        for inst in self.data:
            wordpieces = inst['wordpieces']
            wordpiece_labels = inst['wordpiece_labels']
            wordpiece_ends = inst['wordpiece_ends']
            paragraph_index = inst['paragraph_index']
            # Pad word pieces with special tokens
            piece_idxs = encode_pieces(self.config.wordpiece_splitter, wordpieces, self.config.max_input_length)
            assert len(piece_idxs) <= self.config.max_input_length

            pad_num = self.config.max_input_length - len(piece_idxs)
            attn_masks = [1] * len(piece_idxs) + [0] * pad_num
            piece_idxs = piece_idxs + [0] * pad_num

            # token type idxs
            token_type_idxs = [-100 if piece_id >= len(wordpieces) else wordpiece_labels[piece_id] for piece_id in
                               range(len(piece_idxs) - 2)]

            instance = Instance(
                paragraph_index=paragraph_index,
                wordpieces=wordpieces,
                wordpiece_labels=wordpiece_labels,
                wordpiece_ends=wordpiece_ends,
                piece_idxs=piece_idxs,
                attention_masks=attn_masks,
                token_type_idxs=token_type_idxs,
                wordpiece_num=len(wordpieces)
            )
            data.append(instance)
        self.data = data

    def collate_fn(self, batch):
        batch_paragraph_index = []
        batch_wordpieces = []
        batch_wordpiece_labels = []
        batch_wordpiece_ends = []

        batch_piece_idxs = []
        batch_attention_masks = []
        batch_token_type_idxs = []
        batch_wordpiece_num = []

        for inst in batch:
            batch_paragraph_index.append(inst.paragraph_index)
            batch_wordpieces.append(inst.wordpieces)
            batch_wordpiece_labels.append(inst.wordpiece_labels)
            batch_wordpiece_ends.append(inst.wordpiece_ends)

            batch_piece_idxs.append(inst.piece_idxs)
            batch_attention_masks.append(inst.attention_masks)

            batch_token_type_idxs.append(inst.token_type_idxs)

            batch_wordpiece_num.append(inst.wordpiece_num)

        batch_piece_idxs = torch.tensor(batch_piece_idxs, dtype=torch.long, device=self.config.device)
        batch_attention_masks = torch.tensor(batch_attention_masks, dtype=torch.long, device=self.config.device)
        batch_token_type_idxs = torch.tensor(batch_token_type_idxs, dtype=torch.long, device=self.config.device)
        batch_wordpiece_num = torch.tensor(batch_wordpiece_num, dtype=torch.long, device=self.config.device)

        return Batch(
            paragraph_index=batch_paragraph_index,
            wordpieces=batch_wordpieces,
            wordpiece_labels=batch_wordpiece_labels,
            wordpiece_ends=batch_wordpiece_ends,
            piece_idxs=batch_piece_idxs,
            attention_masks=batch_attention_masks,
            token_type_idxs=batch_token_type_idxs,
            wordpiece_num=batch_wordpiece_num
        )
