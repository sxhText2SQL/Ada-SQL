import json
import re
import sqlite3
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
import time
import threading
import numpy as np


train_data_path = ""
train_data_tables_path = ""
with open(train_data_path,'r', encoding="utf-8") as f2:
    train_data_reward = json.load(f2)
with open(train_data_tables_path,'r', encoding="utf-8") as f4:
    bird_tables = json.load(f4)

train_data_reward_dict = {}
for example in train_data_reward:
    train_data_reward_dict[example["input_prompt"]] = json.loads(example["output_answer"])


def parse_response(response):
    pattern = r"```sql\s*(.*?)\s*```"
    sql_blocks = re.findall(pattern, response, re.DOTALL)
    if sql_blocks:
        last_sql = sql_blocks[-1].strip()
        return last_sql
    else:
        # print("No SQL blocks found.")
        return ""

def structural_reward(completion):
    if "<think>" in completion and "</think>" in completion and "<answer>" in completion and "</answer>" in completion:
        return 0.0
    else:
        return -0.2


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("operator timeout")


def evaluate_sql(db_path: str, sql: str, timeout: int = 30):
    result = {
        "executable": False,
        "result": None,
        "error": None,
        "execution_time": None
    }

    start_time = time.time()

    def _execute_sql():
        nonlocal result
        try:
            conn = sqlite3.connect(db_path, timeout=timeout)
            conn.execute("PRAGMA journal_mode=WAL;")

            cursor = conn.cursor()
            cursor.execute(sql)
            res = cursor.fetchall()
            conn.close()

            result["executable"] = True
            result["result"] = frozenset(res)

        except Exception as e:
            result["error"] = str(e)
            result["executable"] = False

    try:

        execution_thread = threading.Thread(target=_execute_sql)
        execution_thread.daemon = True

        execution_thread.start()
        execution_thread.join(timeout=timeout)


        if execution_thread.is_alive():

            result["error"] = f"SQL time out ({timeout}s)"
            result["executable"] = False

    except Exception as e:
        result["error"] = str(e)
        result["executable"] = False

    finally:
        result["execution_time"] = time.time() - start_time

    return result


model_path = ""
tokenizer = AutoTokenizer.from_pretrained(model_path)
def count_output_tokens(text):
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return len(token_ids)

def calculate_reward(prompt,completions,step,total_steps):
    progress = step / total_steps

    try:
        prompt_processed = prompt.split("<|im_start|>user\n")[1].split("<|im_end|>")[0]
        output_answer = train_data_reward_dict[prompt_processed]
        gold_SQL = output_answer["gold_sql"]
        db_id = output_answer["db_id"]
        benchmark = output_answer["benchmark"]
        question = output_answer["question"]

        question_infos = {
            "question_token_len" : output_answer["question_token_len"] ,
            "question_ner_num": output_answer["question_ner_num"] ,
            "question_number_num" : output_answer["question_number_num"] ,
            "question_dependency_tree_depth":output_answer["question_dependency_tree_depth"]
        }

        db_path = ""
        tables = bird_tables


        use_db = {}
        for db in tables:
            if db_id == db["db_id"]:
                use_db = db
                break


        result_dict = {}
        cnt = -1
        for completion in completions:
            cnt += 1
            result_dict[cnt] = {
                "reward": 0.0,
                "strcut_reward": 0.0,
                "syntax_reward": 0.0,
                "execution_result_reward": 0.0,
                "group_explore_reward": 0.0,
                "completion": completion,
                "reasoning_tokens": 0
            }

        passed_id = []
        # structural_reward
        for com_id in result_dict:
            completion = result_dict[com_id]["completion"]
            struct_reward = structural_reward(completion)
            result_dict[com_id]["strcut_reward"] = struct_reward
            if struct_reward == 0.0:
                passed_id.append(com_id)


        # syntax & semantic
        calculate_id = passed_id
        passed_id =  []
        correct_sqls = []

        no_reasoning_nums = 0
        no_reasoning_acc = 0
        reasoning_nums = 0
        reasoning_acc = 0
        for com_id in calculate_id:
            completion = result_dict[com_id]["completion"]
            rollout_text = completion.split('<think>')[1].split('</think>')[0].strip()
            reasoning_tokens = count_output_tokens(rollout_text)
            result_dict[com_id]["reasoning_tokens"] = reasoning_tokens
            if reasoning_tokens > 0:
                reasoning_nums += 1
            else:
                no_reasoning_nums += 1


        gold_sql_res_dict = evaluate_sql(db_path=db_path, sql=gold_SQL,timeout=30)
        gold_sql_exeable = gold_sql_res_dict["executable"]
        gold_sql_res = gold_sql_res_dict["result"]

        if gold_sql_exeable:
            for com_id in calculate_id:
                completion = result_dict[com_id]["completion"]
                rollout_text = completion.split('<answer>')[1].split('</answer>')[0]
                rollout_sql = parse_response(rollout_text)
                rollout_sql_result_dict = evaluate_sql(db_path=db_path,sql=rollout_sql,timeout=30)
                rollout_sql_exeable = rollout_sql_result_dict["executable"]
                rollout_sql_res = rollout_sql_result_dict["result"]

                if rollout_sql_exeable:
                    result_dict[com_id]["syntax_reward"] = 0.1
                    if rollout_sql_res == gold_sql_res:
                        passed_id.append(com_id)
                        result_dict[com_id]["execution_result_reward"] = 1.0
                        correct_sqls.append(rollout_sql)
                        reasoning_tokens = result_dict[com_id]["reasoning_tokens"]
                        if reasoning_tokens > 0:
                            reasoning_acc += 1
                        else:
                            no_reasoning_acc += 1
                    else:
                        result_dict[com_id]["execution_result_reward"] = 0.0
                else:
                    result_dict[com_id]["syntax_reward"] = -0.1
        else:
            print("gold SQL error")
            passed_id = calculate_id


        no_reasoning_ex = no_reasoning_acc / no_reasoning_nums if no_reasoning_nums > 0 else 0.0
        reasoning_ex = reasoning_acc / reasoning_nums if reasoning_nums > 0 else 0.0

        all_nums = no_reasoning_nums + reasoning_nums
        if all_nums > 0:
            no_reasoning_proportion = no_reasoning_nums / all_nums
            reasoning_proportion = reasoning_nums / all_nums
        else:
            no_reasoning_proportion = 0.0
            reasoning_proportion = 0.0

        # group_ex_reward
        calculate_id = passed_id
        passed_id = []
        for com_id in calculate_id:
            reasoning_tokens = result_dict[com_id]["reasoning_tokens"]
            if reasoning_tokens == 0:
                # NO Reasoning
                result_dict[com_id]["group_explore_reward"] = 1 - no_reasoning_proportion
            else:
                # Reasoning
                result_dict[com_id]["group_explore_reward"] = 1 - reasoning_proportion

        if progress < 0.1:
            explore_weight = 1 - progress
        else:
            explore_weight = 0.0

        reward_list = []
        for com_id in result_dict:
            struct_reward_f = result_dict[com_id]["strcut_reward"]
            syn_reward_f = result_dict[com_id]["syntax_reward"]
            ex_reward_f = result_dict[com_id]["execution_result_reward"]
            group_explore_f = result_dict[com_id]["group_explore_reward"]
            reward_f = 0

            if struct_reward_f == 0.0:
                if syn_reward_f == 0.1:
                    if ex_reward_f == 1.0:
                        reward_f = ex_reward_f + explore_weight * group_explore_f
                    else:
                        reward_f = 0.1
                else:
                    reward_f = -0.1
            else:
                reward_f = -0.2

            result_dict[com_id]["reward"] = float(reward_f)
            reward_list.append(result_dict[com_id]["reward"])

    except Exception as e:
        print(e)
        reward_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    return reward_list