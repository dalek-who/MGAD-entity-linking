from typing import List, Dict, Tuple, NamedTuple, Optional, Union
import sqlite3
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from tqdm import tqdm


class TokenizedWikipediaData(NamedTuple):
    word_ids: Union[np.ndarray, np.memmap]
    mention_start_end: List[Tuple[int, int]]
    gold_entity_pageid: Union[np.ndarray, np.memmap]


class TokenizedWikipediaDataAPI(object):
    def __init__(self, data_dir: Union[str, Path]):
        self.data_dir = data_dir if isinstance(data_dir, Path) else Path(data_dir)

        # memmap dtype and shape
        meta_data_path = self.data_dir / "meta.json"
        self.meta = json.load(meta_data_path.open())

        # memmap
        self.entity_num = self._read_memmap("entity_num")
        self.gold_entity_pageid = self._read_memmap("gold_entity_pageid")
        self.mention_start = self._read_memmap("mention_start")
        self.mention_end = self._read_memmap("mention_end")

        self.len_word_ids = self._read_memmap("len_word_ids")
        self.word_ids = self._read_memmap("word_ids")

        # entity & memmap mapping
        entity_context_db_path = self.data_dir / "entity_context_map.db"
        assert entity_context_db_path.exists() and entity_context_db_path.is_file()
        self.entity_context_db = sqlite3.connect(str(entity_context_db_path.absolute()))

    def _read_memmap(self, name: str) -> np.memmap:
        memmap = np.memmap(
            filename=self.data_dir / f"{name}.memmap",
            mode="r",
            dtype=self.meta[name]["dtype"],
            shape=tuple(self.meta[name]["shape"]),
        )
        return memmap

    def __getitem__(self, item) -> TokenizedWikipediaData:
        # entity
        entity_num = self.entity_num[item]
        mention_start = self.mention_start[item][:entity_num]
        mention_end = self.mention_end[item][:entity_num]
        mention_start_end = [(start, end) for start, end in zip(mention_start, mention_end)]
        gold_entity_pageid = self.gold_entity_pageid[item][:entity_num]

        # context
        len_word_ids = self.len_word_ids[item]
        word_ids = self.word_ids[item][:len_word_ids]

        data = TokenizedWikipediaData(
            word_ids=word_ids,
            mention_start_end=mention_start_end,
            gold_entity_pageid=gold_entity_pageid,
        )

        return data

    @property
    def context_num(self) -> int:
        return self.word_ids.shape[0]

    @property
    def mention_num(self) -> int:
        sql = """
                select
                    count(*)
                from
                    entity_context_map
        """
        df = pd.read_sql(sql=sql, con=self.entity_context_db)
        count = df.values[0,0]
        return count

    def entity_memmap_index(self, pageid: int) -> pd.DataFrame:
        sql = f"""
                select
                    wikipedia_pageid,
                    memmap_row,
                    inline_idx
                from
                    entity_context_map
                where 
                    wikipedia_pageid={pageid}
        """
        df = pd.read_sql(sql=sql, con=self.entity_context_db)
        return df



if __name__ == '__main__':
    data_dir = "/home/wangyuanzheng/projects/long_tail_link/data/full_wikipedia_link_luke_format/tokenized/roberta-base/memmap"
    data_api = TokenizedWikipediaDataAPI(data_dir=data_dir)
    data = data_api[5]
    df_memmap_index = data_api.entity_memmap_index(pageid=273285)  # 273285: USA
    print(data_api.mention_num)
    pass
