from dataclasses import dataclass
from tqdm.auto import tqdm
import pandas as pd

# 参考
# https://www.ai-shift.co.jp/techblog/3803

# DOC_NUM = len(df_faq)

# https://qiita.com/ttyszk/items/01934dc42cbd4f6665d2
@dataclass
class EvaluationResults:
    df_result: pd.DataFrame
    mrr: float
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    recall_at_20: float
    recall_at_30: float

def mrr(rank_array):
    return (1 / rank_array).mean()

def recall_at_k(rank_array, k):
    return (rank_array <= k).mean()

def get_rank(query, search_func):
    results = search_func(query)
    chunk_id_list = []
    for doc in results:
        # aid = metadata[documents.index(doc.page_content)]["chunk_id"]
        chunk_id = doc.metadata['chunk_id']
        chunk_id_list.append(chunk_id)
    return chunk_id_list

def evaluate(query_list, search_func, DOC_NUM):
    result_list = []
    for query, chunk_id in tqdm(query_list):
        rank_result = get_rank(query, search_func=search_func)
        if chunk_id not in rank_result:
            rank = DOC_NUM + 1
        else:
            rank = rank_result.index(chunk_id) + 1
        result_list.append((query, chunk_id, rank, rank_result))

    # queryに対する検索結果
    df_result = pd.DataFrame(result_list, columns=["query","chunk_id","rank","rank_result"])
    
    eval_result = EvaluationResults(
        df_result, 
        mrr(df_result["rank"]), 
        recall_at_k(df_result["rank"], 1),
        recall_at_k(df_result["rank"], 3),
        recall_at_k(df_result["rank"], 5),
        recall_at_k(df_result["rank"], 10),
        recall_at_k(df_result["rank"], 20),
        recall_at_k(df_result["rank"], 30)
    )
    return eval_result


def make_result(result_dic):
    result_list=[]
    for model_name,df_result in result_dic.items():
        result_list.append(
            [
                model_name, df_result.mrr, df_result.recall_at_1, df_result.recall_at_3, df_result.recall_at_5,
                df_result.recall_at_10, df_result.recall_at_20, df_result.recall_at_30
            ],
        )
    result_df = pd.DataFrame(
        result_list,
        columns = ["model","mrr","recall_at_1","rank_result_3","rank_result_5","rank_result_10","rank_result_20","rank_result_30"]
    )#.sort_values("mrr", ascending=False)
    
    return result_df


def execute_evaluation(faiss_db,df,DOC_NUM):
    faiss_similarity_result = evaluate(
        query_list = list(zip(df["Query"].values, df["ID"].values)),
        search_func= lambda q: faiss_db.similarity_search(q, k=DOC_NUM),
        DOC_NUM=DOC_NUM
    )
    return faiss_similarity_result