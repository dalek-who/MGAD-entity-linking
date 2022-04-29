from typing import List, Dict, Tuple, NamedTuple, Union, Optional
import json
import logging
from pathlib import Path
import numpy as np
from pandas import DataFrame
from typing import Tuple
import time
from sklearn.metrics import accuracy_score
import sqlite3
import os


def safely_connect_sqlite(db_path: Union[Path, str], check_same_thread: bool = True) -> sqlite3.Connection:
    assert os.path.exists(db_path)
    assert os.path.isfile(db_path)
    # 以只读模式打开
    db_con = sqlite3.connect(
        database=f"file:{db_path}?mode=ro",  # 以只读模式打开
        uri=True,
        check_same_thread=check_same_thread,
    )  # 如果不assert，即使数据库不存在也不会报错，而是创建新数据库
    return db_con


def current_time_string():
    """
    格式化的当前时间字符串
    :return:
    """
    return time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))


def is_jsonable(obj):
    """
    判断对象obj是否可json序列化
    :param obj:
    :return:
    """
    try:
        json.dumps(obj)
        return True
    except (TypeError, OverflowError):  # OverflowError是对象太大导致的无法序列化
        return False


def init_logger(log_file=None, log_file_level=logging.NOTSET, logger_name="experiment") -> logging.Logger:
    '''
    Example:
        init_logger(log_file)
        logger.info("abc'")
    '''

    # 创建一个logger
    # 在其他任何地方调用logging.getLogger("experiment")得到的都是同一个logger对象
    logger = logging.getLogger(logger_name)
    logger.propagate = False  # 防止屏幕上的logger被显示多次

    # 设置logger级别，高于给定级别的才会显示
    # 级别由低到高: notset < debug < info < warning < error < critical
    logger.setLevel(logging.INFO)  # 例：logger.debug(...)不显示, logger.info(...), logger.warning(...)会显示

    # 定义handler的输出格式
    # log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    log_format = logging.Formatter("%(message)s")

    logger.handlers = []  # 防止如果init_logger被调用调用多次时, 把同一个handler加入好几遍，会导致把一行信息输出多遍
    # 创建一个handler，用于输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    # 再创建一个handler，用于写入日志文件
    if isinstance(log_file, Path):
        log_file = str(log_file)
    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    return logger


def find_best_nil_threshold(
        array_true: np.ndarray, array_pred: np.ndarray, max_score_pred: np.ndarray, nil_token) \
        -> Tuple[float, np.ndarray]:
    """
    寻找最好的NIL分割阈值
    :param array_true: truth index
    :param array_pred: 预测的index
    :param max_score_pred: 得分最高实体的得分
    :return:
        best_threshold: 最佳nil分割阈值
        best_pred_with_threshold: 用最佳阈值分割后的最好预测结果
    """
    threshold = 0.  # 寻找的阈值
    increase = .01  # 每次迭代时的阈值增量
    best_threshold = 0.  # 最佳阈值
    best_metrics = 0.  # 最佳预测结果

    # 网格搜索最佳nil阈值。速度很快，500000条（train数据的两倍）也只需要一分钟
    assert type(nil_token) == type(array_pred[0]), (type(nil_token), type(array_pred[0]))
    while threshold <= 1.:
        index_pred_with_threshold = array_pred.copy()
        index_pred_with_threshold[max_score_pred < threshold] = nil_token  # 根据阈值修正预测结果
        metrics = accuracy_score(y_true=array_true, y_pred=index_pred_with_threshold)
        if metrics > best_metrics:
            best_metrics = metrics
            best_threshold = threshold
        threshold += increase

    best_pred_with_threshold = array_pred.copy()
    best_pred_with_threshold[max_score_pred < best_threshold] = nil_token  # 用最佳阈值分割后的最好预测结果
    return best_threshold, best_pred_with_threshold


def reset_nil_by_threshold(df_pred_result: DataFrame, nil_threshold: float, nil_index: int, nil_kb_id: str):
    """
    利用nil分割阈值将result中打分低于阈值的设置为NIL
    :param df_pred_result: 预测结果DataFrame
    :param nil_threshold: nil分割阈值
    :param nil_index: 给nil的candidate list index，默认-100
    :param nil_kb_id: nil的kb_id, 默认"NIL"
    :return: 将打分低于阈值的预测结果设置为nil后的DataFrame
    """
    # 用最佳阈值分割后的最好预测结果DataFrame
    df_pred_result_with_threshold: DataFrame = df_pred_result.copy()
    nil_row_index = df_pred_result_with_threshold["max_score_pred"] < nil_threshold
    df_pred_result_with_threshold.loc[nil_row_index, "entity_index_pred"] = nil_index
    df_pred_result_with_threshold.loc[nil_row_index, "kb_id_pred"] = nil_kb_id
    return df_pred_result_with_threshold


def md_table_confusion_matrix(confusion_matrix, categories_list) -> Tuple[str, str]:
    # 把混淆矩阵转换成Markdown table的字符串格式
    used_categories_list = categories_list
    df_confusion_matrix = DataFrame(confusion_matrix, index=used_categories_list, columns=used_categories_list)
    df_confusion_matrix["total"] = df_confusion_matrix.sum(axis="columns")
    md_table_cm = df_confusion_matrix.to_markdown()  # 未归一化的混淆矩阵表格

    normalize_confusion_matrix: np.ndarray = confusion_matrix / confusion_matrix.sum(1).reshape((-1, 1))  # 按行归一化的混淆矩阵
    normalize_confusion_matrix = normalize_confusion_matrix.round(3)  # 保留三位小数
    df_normalize_confusion_matrix = DataFrame(normalize_confusion_matrix, index=used_categories_list, columns=used_categories_list)
    df_normalize_confusion_matrix["total"] = df_normalize_confusion_matrix.sum(axis="columns")
    md_table_normalized_cm = df_normalize_confusion_matrix.to_markdown()  # 归一化的混淆矩阵表格

    return md_table_cm, md_table_normalized_cm
