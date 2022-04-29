from typing import List, Dict, Tuple, NamedTuple, Union, Optional
import lmdb
from pathlib import Path
import numpy as np
import os


class TokenizedEntity(NamedTuple):
    wikipedia_pageid: int
    title: Optional[np.ndarray]
    description: Optional[np.ndarray]


class TokenizedWikipediaLmdbAPI(object):
    def __init__(self, lmdb_dir: Union[str, Path]):
        lmdb_dir = str(lmdb_dir.absolute()) if isinstance(lmdb_dir, Path) else lmdb_dir
        assert os.path.exists(lmdb_dir) and os.path.isdir(lmdb_dir)
        self._env = lmdb.open(lmdb_dir, readonly=True, lock=False, max_dbs=2)
        self._db_title = self._env.open_db(b'__title__')
        self._db_description = self._env.open_db(b'__description__')

    def _get_array_from_db(self, db, pageid: int) -> Optional[np.ndarray]:
        with self._env.begin(db=db) as txn:
            bytes_data = txn.get(str(pageid).encode(), None)
            array_data = np.frombuffer(bytes_data, dtype=np.long) if bytes_data is not None else None
        return array_data

    def get_entity(self, pageid: int) -> Optional[TokenizedEntity]:
        array_title = self._get_array_from_db(db=self._db_title, pageid=pageid)
        array_description = self._get_array_from_db(db=self._db_description, pageid=pageid)
        if array_title is None and array_description is None:
            return None
        else:
            entity = TokenizedEntity(
                wikipedia_pageid=pageid,
                title=array_title,
                description=array_description
            )
            return entity

if __name__ == '__main__':
    from transformers import AutoTokenizer

    bert_model = "roberta-base"
    bert_dir = f"/home/wangyuanzheng/resources/pretrain_weight/{bert_model}"
    tokenizer = AutoTokenizer.from_pretrained(bert_dir)

    lmdb_dir = Path("/home/wangyuanzheng/projects/long_tail_link/data/wikipedia_tokenized") / bert_model
    api = TokenizedWikipediaLmdbAPI(lmdb_dir=lmdb_dir)
    entity = api.get_entity(pageid=31415)
    pass