import json
from transformers import AutoTokenizer
# import spacy
import re
from sql_table_column_parse import get_table_and_column
# import numpy as np
import joblib
import pandas as pd

model_path = ""
tokenizer = AutoTokenizer.from_pretrained(model_path)
def count_output_tokens(text):
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return len(token_ids)


def count_keywords(sql):
    sql_keywords = ['select', 'from', 'where', 'group', 'order', 'limit', 'intersect', 'union', \
                    'except', 'join', 'on', 'as', 'not', 'between', 'in', 'like', 'is', 'exists', 'max', 'min', \
                    'count', 'sum', 'avg', 'and', 'or', 'desc', 'asc', 'with']
    sql = sql.lower()
    tokens = re.findall(r"[a-z_]+", sql)

    cnt = 0
    for t in tokens:
        if t in sql_keywords:
            cnt += 1

    return cnt

def count_join(sql):
    sql = sql.lower()
    tokens = re.findall(r"[a-z_]+", sql)
    return sum(1 for t in tokens if t == "join")



def count_sql_conditions(sql):
    sql = sql.lower()
    tokens = re.findall(r"[a-z_]+", sql)

    condition_count = 0
    for t in tokens:
        if t in ("where", "and", "or"):
            condition_count += 1

    return condition_count

reg_low = joblib.load("")
reg_high = joblib.load("")


def predict_token_budget(db_id,question,correct_sqls,use_db):
    try:
        # Schema
        table_num = len(use_db["table_names_original"])
        column_num = len(use_db["column_names_original"]) - 1
        avg_column_num = column_num / table_num
        fk_num = len(use_db["foreign_keys"])
        table_interconnectedness = fk_num / table_num

        # NL Question
        question_token_len = count_output_tokens(question)

        # SQL
        sum_tokens = 0
        for sql in correct_sqls:
            sum_tokens = sum_tokens + count_output_tokens(sql)
        avg_sql_tokens = sum_tokens / len(correct_sqls)


        join_sum = 0
        for sql in correct_sqls:
            join_sum = join_sum + count_join(sql)
        avg_join_num = join_sum / len(correct_sqls)


        keyword_sum = 0
        for sql in correct_sqls:
            keyword_sum = keyword_sum + count_keywords(sql)
        avg_keyword_num = keyword_sum / len(correct_sqls)


        data = [use_db]
        use_table_nums = 0
        use_column_nums = 0

        use_table_ratio_sum = 0.0
        use_column_ratio_sum = 0.0
        for sql in correct_sqls:
            try:
                use_tables, use_columns = get_table_and_column(db=db_id, sql=sql, data=data)
                use_table_nums = use_table_nums + len(use_tables)
                use_column_nums = use_column_nums + len(use_columns)

                use_table_ratio_sum = use_table_ratio_sum + len(use_tables) / table_num
                use_column_ratio_sum = use_column_ratio_sum + len(use_columns) / column_num
            except:
                pass

        avg_use_tables = use_table_nums / len(correct_sqls)
        avg_use_columns = use_column_nums / len(correct_sqls)

        use_table_ratio = use_table_ratio_sum / len(correct_sqls)
        use_column_ratio = use_column_ratio_sum / len(correct_sqls)


        where_condition_sum = 0
        for sql in correct_sqls:
            where_condition_sum = where_condition_sum + count_sql_conditions(sql)
        avg_where_condition_num = where_condition_sum / len(correct_sqls)

        lower_features = [
            "feature_sql_avg_token_len",
            "feature_sql_avg_keyword_num",
            "feature_sql_avg_use_column_num",
            "feature_sql_avg_join_num",
            "feature_sql_avg_use_table_num",
            "feature_question_token_len",
            "feature_sql_avg_use_column_ratio",
            "feature_sql_avg_use_table_ratio"
        ]
        upper_features = [
            "feature_sql_avg_token_len",
            "feature_sql_avg_keyword_num",
            "feature_sql_avg_use_column_num",
            "feature_sql_avg_join_num",
            "feature_sql_avg_use_table_num",
            "feature_question_token_len",
            "feature_sql_avg_use_column_ratio",
            "feature_sql_avg_use_table_ratio"
        ]
        feature_vector_lower = pd.DataFrame([[
            avg_sql_tokens,
            avg_keyword_num,
            avg_use_columns,
            avg_join_num,
            avg_use_tables,
            question_token_len,
            use_column_ratio,
            use_table_ratio
        ]], columns=lower_features)

        lower_pred = reg_low.predict(feature_vector_lower)[0]

        feature_vector_upper = pd.DataFrame([[
            avg_sql_tokens,
            avg_keyword_num,
            avg_use_columns,
            avg_join_num,
            avg_use_tables,
            question_token_len,
            use_column_ratio,
            use_table_ratio
        ]], columns=lower_features)
        upper_pred = reg_high.predict(feature_vector_upper)[0]

        lower = max(1, int(round(lower_pred)))
        upper = max(lower, int(round(upper_pred)))


    except Exception as e:
        print(e)
        lower = 1
        upper = 1500
    return lower,upper




