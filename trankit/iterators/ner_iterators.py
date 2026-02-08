from . import *


instance_fields = [
    'sent_index', 'word_ids',
    'words', 'word_num',
    'piece_idxs', 'attention_masks', 'word_lens',
    'entity_label_idxs'
]

batch_fields = [
    'sent_index', 'word_ids',
    'words', 'word_num', 'word_mask',
    'piece_idxs', 'attention_masks', 'word_lens',
    'entity_label_idxs'
]

Instance = namedtuple('Instance', field_names=instance_fields)

Batch = namedtuple('Batch', field_names=batch_fields)

train_instance_fields = [
    'words', 'word_num',
    'piece_idxs', 'attention_masks', 'word_lens',
    'entity_label_idxs'
]

train_batch_fields = [
    'words', 'word_num', 'word_mask',
    'piece_idxs', 'attention_masks', 'word_lens',
    'entity_label_idxs'
]

Train_Instance = namedtuple('Train_Instance', field_names=train_instance_fields)

Train_Batch = namedtuple('Train_Batch', field_names=train_batch_fields)


class NERDatasetLive(Dataset):
    def __init__(self, config, tokenized_sentences):
        self.config = config
        self.wordpiece_splitter = config.wordpiece_splitter
        self.max_input_length = 512
        # load data
        self.data = [{'sent_index': sid, 'words': sentence, 'word_ids': list(range(len(sentence)))} for sid, sentence in
                     enumerate(tokenized_sentences)]

        # split long sentences into 512-length chunks
        new_data = []
        for inst in self.data:
            words = inst['words']
            pieces = [[p for p in self.wordpiece_splitter.tokenize(w) if p != '▁'] for w in words]
            for ps in pieces:
                if len(ps) == 0:
                    ps += ['-']
            flat_pieces = [p for ps in pieces for p in ps]
            if len(flat_pieces) > self.max_input_length - 2:
                sub_insts = []
                cur_inst = deepcopy(inst)
                for key in ['words', 'word_ids', 'flat_pieces', 'pieces']:
                    cur_inst[key] = []

                for i in range(len(inst['words'])):
                    for key in ['words', 'word_ids']:
                        cur_inst[key].append(inst[key][i])
                    cur_inst['pieces'].append(pieces[i])
                    cur_inst['flat_pieces'].extend(pieces[i])
                    if len(cur_inst['flat_pieces']) >= self.max_input_length - 10:
                        sub_insts.append(cur_inst)

                        cur_inst = deepcopy(inst)
                        for key in ['words', 'word_ids', 'flat_pieces', 'pieces']:
                            cur_inst[key] = []

                if len(cur_inst['flat_pieces']) > 0:
                    sub_insts.append(cur_inst)

                # all sub instances share the same sent_index,
                # 'word_ids' is used for later filling predictions into the right place
                new_data.extend(sub_insts)
            else:
                inst['pieces'] = pieces
                new_data.append(inst)
        self.data = new_data

        # load vocab
        self.vocabs = self.config.ner_vocabs[self.config.active_lang]

    @classmethod
    def from_tagger_data(cls, config, tagger_dataset):
        """Build NER dataset directly from a numberized TaggerDatasetLive,
        reusing its piece_idxs/attention_masks/word_lens to avoid redundant
        wordpiece tokenization.

        Only valid when MWT expansion is not used (tagger and NER process
        the same words). The tagger uses 1-based word_ids while NER uses
        0-based, so we convert here."""
        obj = cls.__new__(cls)
        obj.config = config
        obj.wordpiece_splitter = config.wordpiece_splitter
        obj.max_input_length = 512
        obj.vocabs = config.ner_vocabs[config.active_lang]
        # Convert tagger instances to NER instances, remapping word_ids
        # from 1-based (tagger) to 0-based (NER)
        obj.data = [
            Instance(
                sent_index=inst.sent_index,
                word_ids=[wid - 1 for wid in inst.word_ids],
                words=inst.words,
                word_num=inst.word_num,
                piece_idxs=inst.piece_idxs,
                attention_masks=inst.attention_masks,
                word_lens=inst.word_lens,
                entity_label_idxs=[0] * inst.word_num,
            )
            for inst in tagger_dataset.data
        ]
        return obj

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def numberize(self):
        data = []
        for inst in self.data:
            pieces = inst['pieces']
            word_lens = [len(x) for x in pieces]
            flat_pieces = [p for ps in pieces for p in ps]
            # Pad word pieces with special tokens
            piece_idxs = self.wordpiece_splitter.encode(
                flat_pieces,
                add_special_tokens=True,
                max_length=self.max_input_length,
                truncation=True
            )

            attn_masks = [1] * len(piece_idxs)
            piece_idxs = piece_idxs

            instance = Instance(
                sent_index=inst['sent_index'],
                word_ids=inst['word_ids'],
                words=inst['words'],
                word_num=len(inst['words']),
                piece_idxs=piece_idxs,
                attention_masks=attn_masks,
                word_lens=word_lens,
                entity_label_idxs=[0 for _ in inst['words']]
            )
            data.append(instance)
        self.data = data

    def collate_fn(self, batch):
        batch_sent_index = [inst.sent_index for inst in batch]
        batch_word_ids = [inst.word_ids for inst in batch]
        batch_words = [inst.words for inst in batch]
        batch_word_num = [inst.word_num for inst in batch]
        batch_word_lens = [inst.word_lens for inst in batch]

        bs = len(batch)
        max_wn = max(batch_word_num)
        max_wp = max(len(inst.piece_idxs) for inst in batch)

        piece_idxs = np.zeros((bs, max_wp), dtype=np.int64)
        attn_masks = np.zeros((bs, max_wp), dtype=np.float32)
        word_mask = np.zeros((bs, max_wn), dtype=np.int64)
        entity_label_idxs = np.zeros((bs, max_wn), dtype=np.int64)

        for i, inst in enumerate(batch):
            n_wp = len(inst.piece_idxs)
            n_w = inst.word_num

            piece_idxs[i, :n_wp] = inst.piece_idxs
            attn_masks[i, :n_wp] = inst.attention_masks
            word_mask[i, :n_w] = 1
            entity_label_idxs[i, :n_w] = inst.entity_label_idxs

        return Batch(
            sent_index=batch_sent_index,
            word_ids=batch_word_ids,
            words=batch_words,
            word_num=torch.tensor(batch_word_num, dtype=torch.long),
            word_mask=torch.from_numpy(word_mask).eq(0),
            piece_idxs=torch.from_numpy(piece_idxs),
            attention_masks=torch.from_numpy(attn_masks),
            word_lens=batch_word_lens,
            entity_label_idxs=torch.from_numpy(entity_label_idxs),
        )


class NERDataset(Dataset):
    def __init__(self, config, bio_fpath, evaluate=False):
        self.config = config
        self.evaluate = evaluate

        # load data
        self.config.vocab_fpath = os.path.join(self.config._save_dir, '{}.ner-vocab.json'.format(self.config.lang))
        self.data = get_examples_from_bio_fpath(self.config, bio_fpath, evaluate)

        with open(self.config.vocab_fpath) as f:
            self.vocabs = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def numberize(self):
        data = []
        skip = 0
        for inst in self.data:
            words = inst['words']
            pieces = [[p for p in self.config.wordpiece_splitter.tokenize(w) if p != '▁'] for w in words]
            for ps in pieces:
                if len(ps) == 0:
                    ps += ['-']
            word_lens = [len(x) for x in pieces]
            assert 0 not in word_lens
            flat_pieces = [p for ps in pieces for p in ps]
            assert len(flat_pieces) > 0

            if len(flat_pieces) > self.config.max_input_length - 2:
                skip += 1
                continue

            # Pad word pieces with special tokens
            piece_idxs = self.config.wordpiece_splitter.encode(
                flat_pieces,
                add_special_tokens=True,
                max_length=self.config.max_input_length,
                truncation=True
            )

            attn_masks = [1] * len(piece_idxs)
            piece_idxs = piece_idxs
            assert len(piece_idxs) > 0

            entity_label_idxs = [self.vocabs[label] for label in inst['entity-labels']]

            instance = Train_Instance(
                words=inst['words'],
                word_num=len(inst['words']),
                piece_idxs=piece_idxs,
                attention_masks=attn_masks,
                word_lens=word_lens,
                entity_label_idxs=entity_label_idxs
            )
            data.append(instance)
        print('Skipped {} over-length examples'.format(skip))
        print('Loaded {} examples'.format(len(data)))
        self.data = data

    def collate_fn(self, batch):
        batch_words = [inst.words for inst in batch]
        batch_word_num = [inst.word_num for inst in batch]

        batch_piece_idxs = []
        batch_attention_masks = []
        batch_word_lens = []
        batch_entity_label_idxs = []

        max_word_num = max(batch_word_num)
        max_wordpiece_num = max([len(inst.piece_idxs) for inst in batch])
        batch_word_mask = []

        for inst in batch:
            batch_piece_idxs.append(inst.piece_idxs + [0] * (max_wordpiece_num - len(inst.piece_idxs)))
            batch_attention_masks.append(inst.attention_masks + [0] * (max_wordpiece_num - len(inst.piece_idxs)))
            batch_word_lens.append(inst.word_lens)

            batch_entity_label_idxs.append(inst.entity_label_idxs +
                                           [0] * (max_word_num - inst.word_num))
            batch_word_mask.append([1] * inst.word_num + [0] * (max_word_num - inst.word_num))

        batch_piece_idxs = torch.tensor(batch_piece_idxs, dtype=torch.long, device=self.config.device)
        batch_attention_masks = torch.tensor(batch_attention_masks, dtype=torch.float, device=self.config.device)
        batch_entity_label_idxs = torch.tensor(batch_entity_label_idxs, dtype=torch.long, device=self.config.device)
        batch_word_num = torch.tensor(batch_word_num, dtype=torch.long, device=self.config.device)
        batch_word_mask = torch.tensor(batch_word_mask, dtype=torch.long, device=self.config.device).eq(0)

        return Train_Batch(
            words=batch_words,
            word_num=batch_word_num,
            word_mask=batch_word_mask,
            piece_idxs=batch_piece_idxs,
            attention_masks=batch_attention_masks,
            word_lens=batch_word_lens,
            entity_label_idxs=batch_entity_label_idxs
        )
