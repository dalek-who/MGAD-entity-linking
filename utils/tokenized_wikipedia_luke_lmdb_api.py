from typing import List, Dict, Tuple, NamedTuple, Union, Optional
import lmdb
from pathlib import Path
import numpy as np
import os


class TokenizedLukeLinkData(NamedTuple):
    word_ids: Union[np.ndarray, np.memmap]
    mention_start_end: List[Tuple[int, int]]
    gold_entity_pageid: Union[np.ndarray, np.memmap]


class TokenizedWikipediaLukeDataLmdbAPI(object):
    def __init__(self, lmdb_dir: Union[str, Path]):
        lmdb_dir = str(lmdb_dir.absolute()) if isinstance(lmdb_dir, Path) else lmdb_dir
        assert os.path.exists(lmdb_dir) and os.path.isdir(lmdb_dir)
        self._env = lmdb.open(lmdb_dir, readonly=True, lock=False, max_dbs=4)
        self._db_word_ids = self._env.open_db(b'__db_word_ids__')
        self._db_gold_entity_pageid = self._env.open_db(b'__gold_entity_pageid__')
        self._db_mention_start = self._env.open_db(b'__mention_start__')
        self._db_mention_end = self._env.open_db(b'__mention_end__')

    def _get_array_from_db(self, db, context_id: int) -> Optional[np.ndarray]:
        with self._env.begin(db=db) as txn:
            bytes_data = txn.get(str(context_id).encode(), None)
            array_data = np.frombuffer(bytes_data, dtype=np.int32) if bytes_data is not None else None
        return array_data

    def __getitem__(self, item: int) -> Optional[TokenizedLukeLinkData]:
        array_word_ids = self._get_array_from_db(db=self._db_word_ids, context_id=item)
        array_mention_start = self._get_array_from_db(db=self._db_mention_start, context_id=item)
        array_mention_end = self._get_array_from_db(db=self._db_mention_end, context_id=item)
        array_gold_entity_pageid = self._get_array_from_db(db=self._db_gold_entity_pageid, context_id=item)

        if array_word_ids is None:
            raise IndexError(item)
        else:
            mention_start_end = [(start, end) for start, end in zip(array_mention_start, array_mention_end)]
            data = TokenizedLukeLinkData(
                word_ids=array_word_ids,
                mention_start_end=mention_start_end,
                gold_entity_pageid=array_gold_entity_pageid,
            )
            return data


if __name__ == '__main__':
    from transformers import AutoTokenizer

    bert_model = "roberta-base"
    bert_dir = f"/home/wangyuanzheng/resources/pretrain_weight/{bert_model}"
    tokenizer = AutoTokenizer.from_pretrained(bert_dir)

    lmdb_dir = "/home/wangyuanzheng/projects/long_tail_link/data/full_wikipedia_link_luke_format/tokenized/roberta-base/lmdb"
    api = TokenizedWikipediaLukeDataLmdbAPI(lmdb_dir=lmdb_dir)
    data = api[0]
    pass
