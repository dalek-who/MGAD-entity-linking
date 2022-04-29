from pytrec_eval import RelevanceEvaluator
from collections import defaultdict
import numpy as np


def micro_average(result_by_query: dict, measure: str):
    """
    微平均：按mention取平均
    :param result_by_query: RelevanceEvaluator 的按query测评结果
        # key: query_id (形式："doc_id:index")  value: dict
                key: measure_name, value: measure_value（0或1）
    :param measure: measure名
    :return:
    """
    score_list = [r[measure] for r in result_by_query.values()]
    return np.mean(score_list)


def macro_average(result_by_query: dict, measure: str):
    """
    宏平均：先每篇文档内按mention取平均，再按文档取平均
    :param result_by_query: RelevanceEvaluator 的按query测评结果
        # key: query_id (形式："doc_id:index")  value: dict
                key: measure_name, value: measure_value（0或1）
    :param measure: measure名
    """
    score_list_each_doc = defaultdict(list)
    for query_id, result_dict in result_by_query.items():
        doc_name = query_id.split(":")[0]
        score_list_each_doc[doc_name].append(result_dict[measure])
    return np.mean([np.mean(score_list) for score_list in score_list_each_doc.values()])

def trec_count(result_by_query: dict):
    """
    统计trec结果中query、含query的文档数量
    :param result_by_query: RelevanceEvaluator 的按query测评结果
    :return:
    """
    query_num = len(result_by_query)
    doc_name_all = []
    for query_id, result_dict in result_by_query.items():
        doc_name = query_id.split(":")[0]
        doc_name_all.append(doc_name)
    doc_num = len(set(doc_name_all))
    return query_num, doc_num  # query数量，含query的文档数量

def my_trec_eval(gold: dict, run: dict, measure_list: list):
    """
    用gold和run计算每个measure的宏平均、微平均
    :param gold:
    :param run:
    :param measure_list: 指标名字的列表，需要被pytrec_eval支持
    :return:
    """
    trec_evaluator = RelevanceEvaluator(gold, measures=measure_list)
    result_by_query = trec_evaluator.evaluate(run)

    final_result = dict()
    for measure in measure_list:
        final_result[f"micro_{measure}"] = micro_average(result_by_query=result_by_query, measure=measure)
    for measure in measure_list:
        final_result[f"macro_{measure}".upper()] = macro_average(result_by_query=result_by_query, measure=measure)
    final_result["query_num"], final_result["doc_num"] = trec_count(result_by_query=result_by_query)
    return final_result
