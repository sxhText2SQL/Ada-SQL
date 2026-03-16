import json
import re
from vllm import LLM, SamplingParams
import torch
from transformers import AutoTokenizer
import time


def parse_response(response):
    pattern = r"```sql\s*(.*?)\s*```"
    sql_blocks = re.findall(pattern, response, re.DOTALL)
    if sql_blocks:
        last_sql = sql_blocks[-1].strip()
        return last_sql
    else:
        # print("No SQL blocks found.")
        return ""


model_path = ""
print("Model:",model_path)

N = 0
temperature = 0.0

output_path = f""


print("Loading model...")

if "Qwen2.5-" in model_path:
    stop_token_ids = [151645]  # 151645 is the token id of <|im_end|> (end of turn token in Qwen2.5)
elif "deepseek-coder-" in model_path:
    stop_token_ids = [32021]
elif "DeepSeek-Coder-V2" in model_path:
    stop_token_ids = [100001]
elif "OpenCoder-" in model_path:
    stop_token_ids = [96539]
elif "Meta-Llama-" in model_path:
    stop_token_ids = [128009, 128001]
elif "granite-" in model_path:
    stop_token_ids = [0]  # <|end_of_text|> is the end token of granite-3.1 and granite-code
elif "starcoder2-" in model_path:
    stop_token_ids = [0]  # <|end_of_text|> is the end token of starcoder2
elif "Codestral-" in model_path:
    stop_token_ids = [2]
elif "Mixtral-" in model_path:
    stop_token_ids = [2]
elif "OmniSQL-" in model_path:
    stop_token_ids = [151645]  # OmniSQL uses the same tokenizer as Qwen2.5
else:
    print("Use Qwen2.5's stop tokens by default.")
    stop_token_ids = [151645]


max_model_len = 8192
max_input_len = 6144
max_output_len = 2048

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
llm = LLM(
    model=model_path,
    dtype="float32",
    tensor_parallel_size=1,
    max_model_len=8192,
    trust_remote_code=True,
    gpu_memory_utilization=0.92,
    swap_space=42,
    enforce_eager=True,
    disable_custom_all_reduce=True
)

if "Qwen3-" in model_path:
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_output_len,
        n=N,
        top_p=0.8,
        top_k=20,
    )
else:
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_output_len,
        n=N,
        stop_token_ids=stop_token_ids,
    )


# make ddl
def make_ddl_prompt(db):
    ddl_prompt = ""
    foreign_keys = db["foreign_keys"]
    for table_info in db["schema_items"]:
        table_name = table_info["table_name"]
        column_names = table_info["column_names"]
        column_types = table_info["column_types"]
        column_comments = table_info["column_comments"]
        column_contents = table_info["column_contents"]

        pk_indicators = table_info["pk_indicators"]



        table_ddl = f'CREATE TABLE "{table_name}" ('
        column_nums = len(column_names)
        for i in range(column_nums):
            column_name = column_names[i]
            column_type = column_types[i]
            column_comment = column_comments[i]
            column_content = column_contents[i]
            if len(column_content) == 0:
                column_ddl = f'  "{column_name}" {column_type} /* {column_comment} */,'
            else:
                value_str = f'{column_content}'
                column_ddl = f'  "{column_name}" {column_type} /* {column_comment} */ -- example: {value_str},'
            table_ddl = table_ddl + "\n" + column_ddl
        if sum(pk_indicators) == 0:
            pass
        else:
            pk_cols_list = []
            for i in range(len(column_names)):
                if pk_indicators[i] == 1:
                    pk_cols_list.append(column_names[i])
            pk_cols_ddl = ""
            for i in range(len(pk_cols_list)):
                if i == len(pk_cols_list) - 1:
                    pk_cols_ddl = pk_cols_ddl + f'"{pk_cols_list[i]}"'
                else:
                    pk_cols_ddl = pk_cols_ddl + f'"{pk_cols_list[i]}", '

            pk_ddl = f'  PRIMARY KEY ({pk_cols_ddl})'
            table_ddl = table_ddl + "\n" +pk_ddl

        for foreign_key in foreign_keys:
            table1 = foreign_key[0]
            column1 = foreign_key[1]
            table2 = foreign_key[2]
            column2 = foreign_key[3]
            if table1 == table_name:
                fk_ddl = f'  CONSTRAINT fk_{table_name}_{column1} FOREIGN KEY ("{column1}") REFERENCES {table2} ("{column2}")'
                table_ddl = table_ddl + ", \n" + fk_ddl

        table_ddl = table_ddl + "\n" + ")"
        ddl_prompt = ddl_prompt + table_ddl + "\n"

    ddl_prompt = ddl_prompt[0:-1]
    return ddl_prompt


def make_input_prompt(ddl_prompt,question,knowledge):
    prompt = f'''You are a helpful SQL expert assistant. 
The assistant first thinks about how to write the SQL query by analyzing the question, database schema and external knowledge, then provides the final SQL query. 
The reasoning process and SQL query are enclosed within <think> </think> and <answer> </answer> tags respectively. The answer must contain the SQL query within ```sql...``` tags. 

Database Schema: {ddl_prompt} 

External Knowledge: {knowledge} 

Question: {question}
    '''
    return prompt


input_dataset = []
dev_path = ""
schema_path = ""

with open(dev_path, 'r', encoding="utf-8") as rf:
    dev_data = json.load(rf)
with open(schema_path, 'r', encoding="utf-8") as rf2:
    tables = json.load(rf2)

for example in dev_data:
    question_id = example["question_id"]
    db_id = example['db_id']
    question = example['question']
    external_knowledge = example['evidence']
    gold_sql = example['SQL']


    use_db = tables[db_id]
    ddl_prompt = make_ddl_prompt(db=use_db)

    input_prompt = make_input_prompt(ddl_prompt=ddl_prompt, question=question, knowledge=external_knowledge)
    input_dataset.append({"input_prompt": input_prompt})


if "Qwen3-" in model_path:
    chat_prompts = [tokenizer.apply_chat_template(
        [{"role": "user", "content": data["input_prompt"]}],
        add_generation_prompt=True, tokenize=False
    ) for data in input_dataset]
else:
    chat_prompts = [tokenizer.apply_chat_template(
        [{"role": "user", "content": data["input_prompt"]}],
        add_generation_prompt=True, tokenize=False
    ) for data in input_dataset]

print("Generating responses...")
start_time = time.time()
outputs = llm.generate(chat_prompts, sampling_params)
end_time = time.time()

results = []
total_input_tokens = 0
total_output_tokens = 0

for data, output in zip(input_dataset, outputs):
    responses = [o.text for o in output.outputs]
    sqls = []
    for response in responses:
        if "<answer>" in response:
            sqls.append(parse_response(response.split("<answer>")[1]))
        else:
            sqls.append(parse_response(response))

    data["responses"] = responses
    data["pred_sqls"] = sqls

    total_input_tokens += len(output.prompt_token_ids)
    total_output_tokens += sum(len(o.token_ids) for o in output.outputs)

    current_question_tokens_sum = 0
    response_token_list = []
    for o in output.outputs:
        current_question_tokens_sum += len(o.token_ids)
        token_dict = {
            "response": o.text,
            "sql": parse_response(o.text),
            "tokens": len(o.token_ids)
        }
        response_token_list.append(token_dict)

    current_question_tokens_avg = current_question_tokens_sum / N

    data["question_token_sum"] = current_question_tokens_sum
    data["question_token_avg"] = current_question_tokens_avg
    data["response_token"] = response_token_list

    results.append(data)

total_time = end_time - start_time
num_requests = len(input_dataset)
print(f"Total inference time: {total_time:.2f} seconds")
print(f"Number of requests: {num_requests}")
print(f"Average latency per question: {total_time / num_requests:.2f} seconds")
print(f"Total input tokens: {total_input_tokens}")
print(f"Total output tokens: {total_output_tokens}")
print(f"Output tokens per question: {total_output_tokens / num_requests:.2f}")

with open(output_path, "w", encoding="utf-8") as wf:
    json.dump(results, wf, indent=4, ensure_ascii=False)