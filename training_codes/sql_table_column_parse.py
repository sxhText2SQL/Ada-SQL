import sqlglot
from sqlglot import parse_one, exp
import re

def count_left_spaces(s):
    leading_spaces = len(s) - len(s.lstrip(' '))
    return leading_spaces


def get_subqueries(sql):
    tree = sqlglot.transpile(sql, write="sqlite", identify=True, pretty=True)[0]
    sql_list = tree.split('\n')
    query_list = []
    prefixes = ('UNION', 'EXCEPT', 'INTERSECT')
    for i in range(len(sql_list)):
        left_space = count_left_spaces(sql_list[i])
        if sql_list[i].lstrip().startswith("SELECT"):
            subquery_list = []
            for j in range(i, len(sql_list)):
                if count_left_spaces(sql_list[j]) >= count_left_spaces(sql_list[i]):
                    subquery_list.append(sql_list[j])
                else:
                    break
            query_list.append(merge(split_iue(subquery_list)))

    original_sql_list = []
    original_sql = ""
    for i in range(len(sql_list)):
        original_sql_list.append(sql_list[i])
    original_sql = merge(original_sql_list)
    if original_sql not in query_list:
        query_list.append(original_sql)

    return query_list

def merge(query_list):
    for i in range(len(query_list)):
        query_list[i] = query_list[i].lstrip()
    return ' '.join(query_list)


def split_iue(query_list):
    len_list = []
    prefixes = ('UNION', 'EXCEPT', 'INTERSECT')
    for i in range(len(query_list)):
        len_list.append(count_left_spaces(query_list[i]))
    for i in range(len(query_list)):
        if query_list[i].lstrip().startswith(prefixes) and count_left_spaces(query_list[i]) == min(len_list):
            list1 = query_list[0:i]
            return list1
    return query_list


def contains_multiple_selects_regex(input_string):
    matches = re.findall(r'\bSELECT\b', input_string.upper())
    return len(matches) >= 2

def build_forest(strings):
    strings = sorted(set(strings), key=len, reverse=True)
    forest = {}
    child_to_parent = {}

    for s in strings:
        found_parent = False
        for potential_parent in strings:
            if potential_parent != s and s in potential_parent:
                if not found_parent or len(potential_parent) < len(child_to_parent[s]):
                    if contains_multiple_selects_regex(potential_parent):
                        overflap_flag = 0
                        for s1 in strings:
                            if potential_parent != s1 and s1 != s and s1 in potential_parent:
                                if len(s1) > len(s) and potential_parent.find(s) == potential_parent.find(s1):
                                    overflap_flag = 1
                        if overflap_flag == 1:
                            pass
                        else:
                            child_to_parent[s] = potential_parent
                            found_parent = True
        if not found_parent:
            forest[s] = []
    for child, parent in child_to_parent.items():
        if parent in forest:
            forest[parent].append(child)
        else:
            forest[parent] = [child]
    return forest


def replace_common_substring(forest):
    replaced_forest = {}

    def dfs(node, parent=None):
        children = forest.get(node, [])
        replaced_children = []

        for child in children:
            replaced_child, replaced_node = get_replaced_strings(child, node if parent else None)
            replaced_children.append(replaced_child)
            dfs(child, replaced_node)

        replaced_forest[node if not parent else '[sub]' + node.replace(replaced_node, '')] = replaced_children

    roots = [node for node in forest if node not in sum([children for children in forest.values()], [])]
    for root in roots:
        dfs(root)

    return replaced_forest


def get_replaced_strings(child, parent):
    if not parent or len(child) >= len(parent):
        return child, ''

    max_common = ""
    for i in range(len(parent)):
        for j in range(i + 1, len(parent) + 1):
            if parent[i:j] in child and len(parent[i:j]) > len(max_common):
                max_common = parent[i:j]

    return '[sub]' + child.replace(max_common, ''), max_common

def extract_table_col(sql):
    table_list = []
    col_list = []
    col_list_final = []

    sql = sql.replace("`", "\"")
    sql = sql.replace("`", "\"")
    tree = sqlglot.transpile(sql, write="sqlite", identify=True, pretty=True)[0]
    tree_list = tree.split('\n')
    alias_dict = {}
    for table in parse_one(tree).find_all(exp.Table):
        table_list.append(table.name)
        alias_dict.update({table.alias.lower(): table.name})
        alias_dict.update({table.alias.upper(): table.name})

    table_list = list(set(table_list))
    for column in parse_one(tree).find_all(exp.Column):
        col_list.append(column.sql())
    col_list = list(set(col_list))
    for item in col_list:
        if '.' in item:
            item = item.replace('"', '')
            if item.split('.')[0].upper() in alias_dict.keys() or item.split('.')[0].lower() in alias_dict.keys():
                table_name = item.split('.')[0].replace(item.split('.')[0], alias_dict[item.split('.')[0].lower()])
                col_name_base = item.split('.')[1]
                col_name = table_name + "." + col_name_base
            else:
                col_name = item
            col_list_final.append(col_name)
        else:
            col_list_final.append(item)
    return table_list, col_list_final


def get_table_and_column(db, sql, data):
    query_list = []
    if "SELECT *" in sql:
        sql = sql.replace("SELECT *", "SELECT all_columns_star_tag")
    if "select *" in sql:
        sql = sql.replace("select *", "select all_columns_star_tag")

    if "COUNT(*)" in sql:
        sql = sql.replace("COUNT(*)", "COUNT(all_columns_star_tag)")
    if "count(*)" in sql:
        sql = sql.replace("count(*)", "count(all_columns_star_tag)")


    real_col_list = []
    real_table_list = []
    for item in data:
        if item["db_id"] == db:
            real_table_list = item["table_names_original"]
            for i in range(len(item["table_names_original"])):
                for j in range(len(item["column_names_original"])):
                    if item["column_names_original"][j][0] == i:
                        real_col_list.append(
                            item["table_names_original"][i] + '.' + item["column_names_original"][j][1])

    query_list.clear()
    sql = sql.replace("`", "\"")
    sql = sql.replace("`", "\"")
    try:
        tree = sqlglot.transpile(sql, write="sqlite", identify=True, pretty=True)[0]
    except:
        pass
    subqueries_ = get_subqueries(tree)
    subqueries = list(set(subqueries_))
    subqueries.sort(key=subqueries_.index)


    forest = build_forest(subqueries)
    replaced_forest = replace_common_substring(forest)
    replaced_list = []


    replaced_keys = replaced_forest.keys()
    replaced_keys = sorted(replaced_keys, key=len, reverse=True)
    for key, value in replaced_forest.items():
        if len(value) == 0:
            replaced_list.append(key)
        else:
            for item in value:
                key = key.replace(item, "SUB_QUERY")
            for sqlstr in replaced_forest.keys():
                if sqlstr in key:
                    key = key.replace(sqlstr, "SUB_QUERY")
            replaced_list.append(key)

    for replaced_sql in replaced_list:
        if replaced_sql.count("SELECT") > 1:
            pass

    tables_ = []
    tables = []
    cols_ = []
    cols = []
    for item in replaced_list:
        try:
            tables_sub, cols_ori = extract_table_col(item)
            tables_ = tables_ + tables_sub
            if len(tables_sub) == 1:
                for col in cols_ori:
                    if f"{tables_sub[0]}." in col:
                        cols_.append(col.replace('"', ''))
                    else:
                        cols_.append(tables_sub[0] + '.' + col.replace('"', ''))
            else:
                cols_ = cols_ + cols_ori
        except Exception as e:
            pass

    tables_ = list(set(tables_))
    cols_ = list(set(cols_))


    if len(tables_) == 0 or len(cols_) == 0:
        pass
    for item in tables_:
        for real_table in real_table_list:
            item_lower = item.lower()
            real_table_lower = real_table.lower()
            if item_lower == real_table_lower:
                tables.append(real_table)
                break
    for item in cols_:
        if "all_columns_star_tag" in item:
            item = item.replace("all_columns_star_tag", "*")
            item_table = item.split(".")[0]
            table_flag = 0
            for table in tables:
                item_table_lower = item_table.lower()
                table_lower = table.lower()
                if item_table_lower == table_lower:
                    table_flag = 1
                    break
            if table_flag == 1:
                cols.append(item)
                continue
        for real_col in real_col_list:
            item_lower = item.lower()
            real_col_lower = real_col.lower()
            if item_lower == real_col_lower:
                cols.append(real_col)
                break

    return tables, cols