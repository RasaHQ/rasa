# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This script is responsible for translating SQL to SemQL in a flexible and readable method.
"""
from typing import Dict, List, Tuple, Optional, Text
import json
from collections import deque
from copy import deepcopy
import logging
import re

from rasa.core.knowledge_base.converter.process_sql import parse_sql, tokenize
from rasa.core.knowledge_base.schema.database_schema import (
    DatabaseSchema,
    TableColumn,
    Table,
)
from rasa.core.knowledge_base.grammar.grammar import (
    GrammarRule,
    A,
    C,
    T,
    GrammarType,
    Join,
    Select,
    Filter,
    Order,
    Root,
    Grammar,
    GrammarRuleTreeNode,
    Statement,
)
from rasa.core.knowledge_base.grammar.graph import Graph


class SparcType:
    # https://github.com/taoyds/spider/blob/master/preprocess/parsed_sql_examples.sql#L30
    # 'not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists'
    FilterNot = 0
    FilterBetween = 1
    FilterEqual = 2
    FilterGreater = 3
    FilterLess = 4
    FilterGeq = 5
    FilterLeq = 6
    FilterNeq = 7
    FilterIn = 8
    FilterLike = 9
    FilterIs = 10
    FilterExist = 11

    # https://github.com/taoyds/spider/blob/master/preprocess/parsed_sql_examples.sql#L37
    # 'and', 'or'

    # https://github.com/taoyds/spider/blob/0b0c9cad97e4deeef1bc37c8435950f4bdefc141/preprocess/parsed_sql_examples.sql#L32
    # 'none', 'max', 'min', 'count', 'sum', 'avg'
    ANone = 0
    AMax = 1
    AMin = 2
    ACount = 3
    ASum = 4
    AAvg = 5

    # https://github.com/taoyds/spider/blob/master/preprocess/parsed_sql_examples.sql#L31
    # 'none', '-', '+', "*", '/'


class SQLConverter(object):
    """
    The class is designed to handle the process from structural query dict into intermediate action sequence.
    """

    def __init__(self, database_schema: DatabaseSchema):
        """
        :param db_context: data context for database
        """
        self.db_context = database_schema
        self.col_names = database_schema.id_to_columns
        self.table_names = database_schema.id_to_tables

    def convert_to_grammar_rules(self, query: Text) -> List[GrammarRule]:
        sql_clause = {}

        toks = tokenize(query)
        tables_with_aliases = {
            table.alias: table.name for table in self.db_context.tables
        }
        _, sql_clause = parse_sql(toks, 0, tables_with_aliases, self.db_context)

        return self._process_statement(sql_clause)

    def _process_statement(self, sql_clause: Dict) -> List[GrammarRule]:
        """
        Except the intersect/union/except, the remaining parsing method iss implemented here.
        :return:
        """
        if sql_clause["intersect"] is not None:
            inter_seq = [Statement(GrammarType.StateInter)]
            nest_sql_clause = sql_clause["intersect"]
            inter_seq.extend(self._process_root(sql_clause))
            inter_seq.extend(self._process_root(nest_sql_clause))
            return inter_seq

        if sql_clause["union"] is not None:
            inter_seq = [Statement(GrammarType.StateUnion)]
            nest_sql_clause = sql_clause["union"]
            inter_seq.extend(self._process_root(sql_clause))
            inter_seq.extend(self._process_root(nest_sql_clause))
            return inter_seq

        if sql_clause["except"] is not None:
            inter_seq = [Statement(GrammarType.StateExcept)]
            nest_sql_clause = sql_clause["except"]
            inter_seq.extend(self._process_root(sql_clause))
            inter_seq.extend(self._process_root(nest_sql_clause))
            return inter_seq

        # Statement None
        inter_seq = [Statement(GrammarType.StateNone)]
        inter_seq.extend(self._process_root(sql_clause))
        return inter_seq

    def _process_agg_col_table(self, sql_clause, agg_id, col_ind) -> [A, C, T]:
        """
        Combine the three operators into one function to process
        :param col_ind: col index in table. Equal to 0 means occurrence of `*`.
        :param agg_id: aggregation operation id
        :param sql_clause: sql clause for get all tables
        :return: the list of A, C and T.
        """

        def _process_tab() -> T:
            """
            Get table grammar according to col index. Note that the most difficult thing is that
            we should decide a specific table for `*` token.
            :return: selected table grammar
            """
            # col index not equal to 0 means specific column, return its table
            if col_ind != 0:
                table_name = self.col_names[col_ind].refer_table.name
                _table_grammar = T(table_name)
            # * case
            else:
                # Fetch table names, check data format
                from_clause = sql_clause["from"]
                assert "table_units" in from_clause and "conds" in from_clause
                assert isinstance(from_clause["table_units"], list)
                assert isinstance(from_clause["conds"], list)

                table_units = from_clause["table_units"]
                # only one table
                if len(table_units) == 1:
                    ret = table_units[0][1]
                    if type(ret) != int:
                        # use default setting
                        ret = 0
                    # get table name
                    table_name = self.table_names[ret].name
                    _table_grammar = T(table_name)
                # multiple tables
                else:
                    table_set = set()
                    for table_unit_tuple in table_units:
                        # table unit tuple[1] is the table id
                        if type(table_unit_tuple[1]) == int:
                            table_set.add(self.table_names[table_unit_tuple[1]].name)
                    # collect other tables
                    other_set = set()
                    select_clause = sql_clause["select"]
                    where_clause = sql_clause["where"]
                    group_clause = sql_clause["groupBy"]

                    # join table is also used
                    if "join" in sql_clause:
                        other_set.add(sql_clause["join"])

                    for sel_part in select_clause[1]:
                        sel_col_ind = sel_part[1][1][1]
                        if sel_col_ind != 0:
                            # find table according to col index
                            other_set.add(self.col_names[sel_col_ind].refer_table.name)
                    # number of where clause
                    where_num = len(where_clause)
                    if where_num >= 1:
                        where_col_ind = where_clause[0][2][1][1]
                        other_set.add(self.col_names[where_col_ind].refer_table.name)
                    # 3, 5
                    if where_num >= 3:
                        where_col_ind = where_clause[2][2][1][1]
                        other_set.add(self.col_names[where_col_ind].refer_table.name)
                    # 5
                    if where_num >= 5:
                        where_col_ind = where_clause[4][2][1][1]
                        other_set.add(self.col_names[where_col_ind].refer_table.name)

                    # get candidates
                    candi_set = table_set - other_set

                    if len(candi_set) == 1:
                        table_name = candi_set.pop()
                        _table_grammar = T(table_name)
                    elif len(candi_set) == 0 and len(group_clause) != 0:
                        group_col_ind = group_clause[0][1]
                        # get table name
                        table_name = self.col_names[group_col_ind].refer_table.name
                        _table_grammar = T(table_name)
                    # add the first of table unit
                    else:
                        tab_ind = table_units[0][1]
                        _table_grammar = T(self.table_names[tab_ind].name)

            return _table_grammar

        def _process_agg() -> A:
            """
            map sparc id into corresponding grammar for A
            :return: aggregation grammar
            """
            sparc_to_grammar = {
                SparcType.ANone: GrammarType.ANone,
                SparcType.AMax: GrammarType.AMax,
                SparcType.AMin: GrammarType.AMin,
                SparcType.ASum: GrammarType.ASum,
                SparcType.AAvg: GrammarType.AAvg,
                SparcType.ACount: GrammarType.ACount,
            }
            if agg_id in sparc_to_grammar:
                _agg_grammar = A(sparc_to_grammar[agg_id])
            else:
                raise ValueError(f"No support for the aggregate {agg_id}")
            return _agg_grammar

        def _process_col() -> C:
            sel_col_name = self.col_names[col_ind].name
            _col_grammar = C(sel_col_name)
            return _col_grammar

        agg_grammar = _process_agg()
        col_grammar = _process_col()
        table_grammar = _process_tab()
        return [agg_grammar, col_grammar, table_grammar]

    def _process_join(self, sql_clause) -> List[GrammarRule]:
        assert "join" in sql_clause
        assert isinstance(sql_clause["join"], str)

        join_tab_name = sql_clause["join"]
        from_conds = sql_clause["from"]["conds"]
        join_col_inds = []
        for condition in from_conds:
            """
            every from condition is composed by:
            [ #have not#, #index of WHERE_OPS#,
                [ #index of unit_op#, 
                    [ #index of AGG_OPS#, #index of column names#, #is DISTINCT#], 
                  None
                ],
                [ #index of AGG_OPS#, #index of column names#, #is DISTINCT# ],
                None
            ]         
            """
            if isinstance(condition, list):
                from_cond_col_inds = [condition[2][1][1], condition[3][1]]
                for col_ind in from_cond_col_inds:
                    if self.col_names[col_ind].refer_table.name == join_tab_name:
                        join_col_inds.append(col_ind)

        inter_seq: List[GrammarRule] = [Join(0)]

        # use Join A A
        if len(join_col_inds) >= 1:
            # two A
            join_col_inds = [join_col_inds[0]]
        else:
            print(f"Error in Join, there are {len(join_col_inds)} Join columns!")

        for col_ind in join_col_inds:
            inter_seq.extend(
                self._process_agg_col_table(
                    sql_clause=sql_clause, agg_id=SparcType.ANone, col_ind=col_ind
                )
            )
        return inter_seq

    def _process_select(self, sql_clause) -> List[GrammarRule]:
        """
        the select clause will be mapped into A, C and T.
        :return:
        """
        sql_select_clause = sql_clause["select"]
        # check the instance type
        # assert isinstance(sql_select_clause, list)
        # boolean / list of column items
        distinct, sel_items = sql_select_clause[0], sql_select_clause[1]
        # find index of @Select.grammar_dict and initialize intermediate select action sequence
        inter_seq: List[GrammarRule] = [Select(len(sel_items) - 1)]
        # traverse sel items, including aggregation and others
        for sel_item in sel_items:
            # aggregation grammar
            agg_id = sel_item[0]
            col_ind = sel_item[1][1][1]
            inter_seq.extend(
                self._process_agg_col_table(
                    sql_clause=sql_clause, agg_id=agg_id, col_ind=col_ind
                )
            )
        return inter_seq

    def _process_condition(self, sql_clause, cond: List) -> List[GrammarRule]:
        """
        Son function of filter, which aims to align @SparcType with @GrammarType.
        :return:
        """
        inter_seq: List[GrammarRule] = []
        # if the condition is a nested query
        is_nested_query = True if type(cond[3]) == dict else False
        # corresponding where operation index
        sparc_type = cond[1]
        # if there is `Not`, cond[0] becomes `True`
        if cond[0] is True:
            sparc_to_grammar = {
                # add not
                SparcType.FilterIn: GrammarType.FilterNotInNes,
                SparcType.FilterLike: GrammarType.FilterNotLike,
            }
            if sparc_type in sparc_to_grammar:
                filter_grammar = Filter(sparc_to_grammar[sparc_type])
            else:
                raise ValueError(f"No support for sparc type:{sparc_type}")
        else:
            if is_nested_query:
                sparc_to_direct_nested = {
                    SparcType.FilterBetween: GrammarType.FilterBetweenNes,
                    SparcType.FilterEqual: GrammarType.FilterEqualNes,
                    SparcType.FilterNeq: GrammarType.FilterNeqNes,
                    SparcType.FilterGreater: GrammarType.FilterGreaterNes,
                    SparcType.FilterLess: GrammarType.FilterLessNes,
                    SparcType.FilterLeq: GrammarType.FilterLeqNes,
                    SparcType.FilterGeq: GrammarType.FilterGeqNes,
                    # TODO: like and in does not care nested
                    SparcType.FilterLike: GrammarType.FilterLike,
                    SparcType.FilterIn: GrammarType.FilterInNes,
                }
                if sparc_type in sparc_to_direct_nested:
                    filter_grammar = Filter(sparc_to_direct_nested[sparc_type])
                else:
                    raise ValueError(
                        f"Grammar {sparc_type} does not support nested setting"
                    )
            else:
                sparc_to_grammar = {
                    SparcType.FilterBetween: GrammarType.FilterBetween,
                    SparcType.FilterEqual: GrammarType.FilterEqual,
                    SparcType.FilterNeq: GrammarType.FilterNeq,
                    SparcType.FilterGreater: GrammarType.FilterGreater,
                    SparcType.FilterLess: GrammarType.FilterLess,
                    SparcType.FilterLeq: GrammarType.FilterLeq,
                    SparcType.FilterGeq: GrammarType.FilterGeq,
                    SparcType.FilterLike: GrammarType.FilterLike,
                    SparcType.FilterIn: GrammarType.FilterInNes,
                }
                if sparc_type in sparc_to_grammar:
                    filter_grammar = Filter(sparc_to_grammar[sparc_type])
                else:
                    raise ValueError(
                        f"Grammar {sparc_type} does not have a corresponding Filter"
                    )

        inter_seq.append(filter_grammar)
        # A, C, T
        agg_id = cond[2][1][0]
        col_ind = cond[2][1][1]
        inter_seq.extend(
            self._process_agg_col_table(
                sql_clause=sql_clause, agg_id=agg_id, col_ind=col_ind
            )
        )
        # handle with nested query
        if is_nested_query:
            nested_sql_clause = cond[3]
            root_grammar = self._process_root(nested_sql_clause)
            inter_seq.extend(root_grammar)

        return inter_seq

    def _process_filter(self, sql_clause) -> List[GrammarRule]:
        """
        Process where and having clause, merge them into filter operations
        :return: filter action sequences
        """
        sql_where_clause = sql_clause["where"]
        sql_having_clause = sql_clause["having"]
        assert isinstance(sql_where_clause, list)
        assert isinstance(sql_having_clause, list)

        # pre-condition: the where or having has one non-zero
        assert len(sql_where_clause) != 0 or len(sql_having_clause) != 0

        inter_seq = []
        if len(sql_where_clause) != 0 and len(sql_having_clause) != 0:
            # TODO: why do not statistic them together
            filter_grammar = Filter(GrammarType.FilterAnd)
            inter_seq.append(filter_grammar)

        if len(sql_where_clause) != 0:
            # only ordinary number : where1 and where2 or where3 ...
            if len(sql_where_clause) == 1:
                cond_grammar = self._process_condition(
                    sql_clause=sql_clause, cond=sql_where_clause[0]
                )
                inter_seq.extend(cond_grammar)
            elif len(sql_where_clause) == 3:
                # check what is the operation
                if sql_where_clause[1] == "or":
                    filter_grammar = Filter(GrammarType.FilterOr)
                else:
                    filter_grammar = Filter(GrammarType.FilterAnd)
                inter_seq.append(filter_grammar)
                # TODO: parent feeding
                left_cond_grammar = self._process_condition(
                    sql_clause=sql_clause, cond=sql_where_clause[0]
                )
                right_cond_grammar = self._process_condition(
                    sql_clause=sql_clause, cond=sql_where_clause[2]
                )
                inter_seq.extend(left_cond_grammar)
                inter_seq.extend(right_cond_grammar)
            else:
                # enumerate all combinations
                op_to_grammar = {
                    "and": [Filter(GrammarType.FilterAnd)],
                    "or": [Filter(GrammarType.FilterOr)],
                }
                # get operation str, and convert them into grammar
                left_op = sql_where_clause[1]
                left_filter_grammar = op_to_grammar[left_op]

                right_op = sql_where_clause[3]
                right_filter_grammar = op_to_grammar[right_op]

                left_cond_grammar: List[GrammarRule] = self._process_condition(
                    sql_clause=sql_clause, cond=sql_where_clause[0]
                )
                middle_cond_grammar = self._process_condition(
                    sql_clause=sql_clause, cond=sql_where_clause[2]
                )
                right_cond_grammar = self._process_condition(
                    sql_clause=sql_clause, cond=sql_where_clause[4]
                )
                # the priority of `and` is higher than `or`, so we care for the order
                extend_list = [
                    left_cond_grammar,
                    middle_cond_grammar,
                    right_cond_grammar,
                ]
                combine_type = f"{left_op}@{right_op}"
                if (
                    combine_type == "and@and"
                    or combine_type == "or@or"
                    or combine_type == "and@or"
                ):
                    # 1. where1 and(l) where2 and(r) where3 -> and(r) and(l) where1 where2 where3
                    # 2. where1 and(l) where2 or(r) where3 -> or(r) and(l) where1 where2 where3
                    extend_list.insert(0, left_filter_grammar)
                    extend_list.insert(0, right_filter_grammar)
                elif combine_type == "or@and":
                    # where1 or(l) where2 and(r) where3 -> or(l) where1 and(r) where2 where3
                    extend_list.insert(1, right_filter_grammar)
                    extend_list.insert(0, left_filter_grammar)
                else:
                    raise ValueError(
                        f"We do not support Filter combine type:{combine_type}"
                    )

                for extend_grammar in extend_list:
                    inter_seq.extend(extend_grammar)
                    # TODO: now we do not consider where which has more clauses than 3

        # handle having clause
        if len(sql_having_clause) != 0:
            cond_grammar = self._process_condition(
                sql_clause=sql_clause, cond=sql_having_clause[0]
            )
            inter_seq.extend(cond_grammar)
        # no non-terminal
        return inter_seq

    def _process_order(self, sql_clause) -> List[GrammarRule]:
        """
        the orderby clause will be mapped into Order, A, C, T
        :return:
        """
        sql_order_clause = sql_clause["orderBy"]
        sql_limit_clause = sql_clause["limit"]

        # pre-condition: if has order by, it will be processed by this function
        assert len(sql_order_clause) != 0

        inter_seq = []
        if sql_limit_clause is not None:
            if sql_order_clause[0] == "asc":
                order_grammar = Order(GrammarType.OrderAscLim)
            else:
                order_grammar = Order(GrammarType.OrderDesLim)
        else:
            if sql_order_clause[0] == "asc":
                order_grammar = Order(GrammarType.OrderAsc)
            else:
                order_grammar = Order(GrammarType.OrderDes)

        # orderBy grammar
        inter_seq.append(order_grammar)

        # aggregate grammar
        agg_id = sql_order_clause[1][0][1][0]
        col_ind = sql_order_clause[1][0][1][1]
        inter_seq.extend(
            self._process_agg_col_table(
                sql_clause=sql_clause, agg_id=agg_id, col_ind=col_ind
            )
        )
        # no non-terminal
        return inter_seq

    def _process_root(self, sql_clause: Dict) -> List[GrammarRule]:
        """
        Process statement and return its corresponding transaction
        :return: grammar transaction clauses
        """

        def _process_step(step_state: str):
            """
            Process every step using the step state
            :param step_state: represent the top state which should be parsed
            :return: returned inner intermediate action sequence and the next state
            """
            call_back_mapping = {
                "Select": self._process_select,
                "Order": self._process_order,
                "Filter": self._process_filter,
                "Join": self._process_join,
            }
            return call_back_mapping[step_state](sql_clause)

        def _use_sep_join(_sql_clause: Dict) -> bool:
            """
            This function will additionally add some new fields `join` into _sql_clause
            :param _sql_clause: sql clause, which a dictionary containing all sql information
            :return: whether to use join
            """
            # determine whether to use Join clause.
            # the main process is to check whether there exists some tables
            # that never appearing in the select/where/group by/order by
            # but appearing in join and on.
            infer_tables = set()
            for condition in _sql_clause["from"]["conds"]:
                """
                every from condition is composed by:
                [ #have not#, #index of WHERE_OPS#,
                    [ #index of unit_op#, 
                        [ #index of AGG_OPS#, #index of column names#, #is DISTINCT#], 
                      None
                    ],
                    [ #index of AGG_OPS#, #index of column names#, #is DISTINCT# ],
                    None
                ]         
                """
                if isinstance(condition, list):
                    from_cond_col_inds = [condition[2][1][1]]
                    # only if there is a column
                    if isinstance(condition[3], list):
                        from_cond_col_inds.append(condition[3][1])
                    for col_ind in from_cond_col_inds:
                        infer_tables.add(self.col_names[col_ind].refer_table.name)

            join_tables = []
            for table_unit_tuple in _sql_clause["from"]["table_units"]:
                # table unit tuple[1] is the table id
                if (
                    type(table_unit_tuple[1]) == int
                    and self.table_names[table_unit_tuple[1]].name in infer_tables
                ):
                    join_tables.append(self.table_names[table_unit_tuple[1]].name)

            if len(join_tables) == 0 or len(join_tables) == 1:
                return False

            used_tables = set()

            select_clause = _sql_clause["select"]
            where_clause = _sql_clause["where"]
            # whether the table is used
            # group_clause = _sql_clause['groupBy']
            order_clause = _sql_clause["orderBy"]
            having_clause = _sql_clause["having"]

            # add table in select into (except *)
            for sel_part in select_clause[1]:
                sel_col_ind = sel_part[1][1][1]
                if sel_col_ind != 0:
                    # find table according to col index
                    used_tables.add(self.col_names[sel_col_ind].refer_table.name)
                elif len(select_clause[1]) == 1:
                    # if len(select_clause[1]) == 1, we could assume
                    # the first join_table should own the *
                    used_tables.add(join_tables[0])
                else:
                    # TODO: we cannot identify, we return False in default
                    return False

            # # add table in group by into (except *)
            # if len(group_clause):
            #     group_col_ind = group_clause[0][1]
            #     assert group_col_ind != 0
            #     # find table according to col index
            #     used_tables.add(self.col_names[group_col_ind].refer_table.name)

            if len(order_clause):
                order_col_ind = order_clause[1][0][1][1]
                if order_col_ind != 0:
                    used_tables.add(self.col_names[order_col_ind].refer_table.name)
                # else:
                #     # TODO: we cannot identify whether the * belongs to which table, so we keep the original
                #     return False

            # add table in where into
            where_num = len(where_clause)
            if where_num >= 1:
                where_col_ind = where_clause[0][2][1][1]
                used_tables.add(self.col_names[where_col_ind].refer_table.name)
            # 3, 5
            if where_num >= 3:
                where_col_ind = where_clause[2][2][1][1]
                used_tables.add(self.col_names[where_col_ind].refer_table.name)
            # 5
            if where_num >= 5:
                where_col_ind = where_clause[4][2][1][1]
                used_tables.add(self.col_names[where_col_ind].refer_table.name)

            # add having
            if len(having_clause) > 0:
                having_col_ind = having_clause[0][2][1][1]
                if having_col_ind != 0:
                    used_tables.add(self.col_names[having_col_ind].refer_table.name)

            if (
                len(join_tables) == 2
                and len(used_tables) == 1
                and len(set(join_tables) - used_tables)
            ):
                # we need identify the table while should appear in JOIN
                join_table_name = list(set(join_tables) - used_tables)[0]
                _sql_clause["join"] = join_table_name
                return True
            else:
                return False

        join_used = _use_sep_join(sql_clause)

        if sql_clause["orderBy"]:
            order_used = True
        else:
            order_used = False

        # check the where
        if sql_clause["where"] == [] and sql_clause["having"] == []:
            filter_used = False
        else:
            filter_used = True

        if filter_used and order_used:
            if join_used:
                inter_seq, next_states = (
                    [Root(GrammarType.RootJSFO)],
                    ["Join", "Select", "Filter", "Order"],
                )
            else:
                inter_seq, next_states = (
                    [Root(GrammarType.RootSFO)],
                    ["Select", "Filter", "Order"],
                )
        elif filter_used:
            if join_used:
                inter_seq, next_states = (
                    [Root(GrammarType.RootJSF)],
                    ["Join", "Select", "Filter"],
                )
            else:
                inter_seq, next_states = (
                    [Root(GrammarType.RootSF)],
                    ["Select", "Filter"],
                )
        elif order_used:
            if join_used:
                inter_seq, next_states = (
                    [Root(GrammarType.RootJSO)],
                    ["Join", "Select", "Order"],
                )
            else:
                inter_seq, next_states = [Root(GrammarType.RootSO)], ["Select", "Order"]
        else:
            if join_used:
                inter_seq, next_states = [Root(GrammarType.RootJS)], ["Join", "Select"]
            else:
                inter_seq, next_states = [Root(GrammarType.RootS)], ["Select"]

        while len(next_states) > 0:
            # pop from left to right, to keep the readable
            cur_state = next_states.pop(0)
            # parse it
            step_inter_seq = _process_step(cur_state)
            inter_seq.extend(step_inter_seq)
        return inter_seq


class ActionConverter(object):
    """
    This class is designed for post-processing on SemQL(also named action) sequence into SQL clause. Note that we DO NOT
    handle any logical problem in this class(e.g. the column should be in the corresponding SQL in `A -> None C T`. You
    should process it in another separate function such as `ConditionStatelet`.
    """

    def __init__(self, db_context: DatabaseSchema):
        """
        :param db_context: data context for database, mainly usage on its knowledge graph for building foreign-key table
        relation graph. Then it is used to inference the JOIN path.
        """
        self.db_context = db_context
        self.graph, self.foreign_pairs = self._build_graph()
        self.processor = {
            Select: self._process_select,
            Order: self._process_order,
            Filter: self._process_filter,
            Join: self._process_sep_join,
        }

    def translate_to_sql(self, action_seq: List[str]):
        """
        Given an action sequence, we should postprocessing it into
        :param action_seq: `List[str]` each item represents an action defined in context/grammar.py
        :return:
        """
        # convert action sequence into action
        action_seq = [
            GrammarRule.from_string(action_repr) for action_repr in action_seq
        ]
        # translation sequence into tree
        root_node = Grammar.build_syntax_tree(action_seq)

        # from root node, traverse the tree
        statement_clause = self._process_statement(root_node)
        # clean up spaces
        statement_clause = re.sub("\\s+", " ", statement_clause)
        return statement_clause

    def _process_join(
        self,
        component_mapping: Dict[str, List[GrammarRuleTreeNode]],
        repr: Dict[str, str],
        is_subquery: bool,
    ):
        """

        :param component_mapping: component mapping records `select`, `order`, `where`, `having`, `from`.
        :param is_subquery: whether is subqueyry, if true, show ON clause.
        :return: From clause result
        """

        def _process_on(_route):
            for i in range(1, len(_route)):
                for j in range(0, i):
                    tab_1_name, tab_2_name = (
                        _route[j].split(" ")[0],
                        _route[i].split(" ")[0],
                    )
                    candidate_routes = []
                    for (
                        key_tab_name,
                        key_col_name,
                        val_tab_name,
                        val_col_name,
                    ) in self.foreign_pairs:
                        if tab_1_name == key_tab_name and tab_2_name == val_tab_name:
                            # TODO: the order of ON matters?
                            candidate_routes.append(
                                (
                                    f" ON {key_tab_name}.{key_col_name} = {val_tab_name}.{val_col_name}",
                                    key_col_name,
                                    val_col_name,
                                )
                            )
                    # check the number of valid routes
                    if len(candidate_routes) == 1:
                        _route[i] += candidate_routes[0][0]
                    elif len(candidate_routes) > 1:
                        # course_id = pred_id, course_id = course_id (between two tables)
                        best_route = candidate_routes[0][0]
                        # there is a circle, we should select the val col and key col euqal one
                        for _route_repr, key_col_name, val_col_name in candidate_routes:
                            if key_col_name == val_col_name:
                                best_route = _route_repr
                                break
                        _route[i] += best_route
            return _route

        # for group by between two tables
        used_tab_names = [
            node.grammar_rule.rule_id for node in component_mapping["from"]
        ]
        join_tables = []

        if len(used_tab_names) != 2:
            # TODO: too complex to handle
            join_tables = used_tab_names
        elif len(used_tab_names) == 2:
            for tab_name in used_tab_names:
                # any table not appeared
                if tab_name not in self.graph.vertices:
                    join_tables = used_tab_names
                    break
            # break will break the else clause
            else:
                tab_start, tab_end = used_tab_names[0], used_tab_names[1]
                route = list(self.graph.dijkstra(tab_start, tab_end))
                if is_subquery:
                    route = _process_on(route)
                join_tables = route if len(route) != 0 else used_tab_names

        repr["from"] = "FROM " + " JOIN ".join(join_tables)

    def _process_group_by(
        self,
        component_mapping: Dict[str, List[GrammarRuleTreeNode]],
        repr: Dict[str, str],
    ):
        """
        Define rules to judge whether the SQL should contain GROUP BY
        :param component_mapping:
        :return:
        """
        having_nodes = component_mapping["having"]
        # first determine whether to group by
        if len(having_nodes) > 0:
            keep_group_by = True
        else:
            keep_group_by = False

        select_nodes = component_mapping["select"]
        order_nodes = component_mapping["order"]
        # if there are two or more columns in select and any one in [count,max,min,avg,sum], we need group
        if len(select_nodes) > 1 and any(
            [
                node
                for node in select_nodes
                if node.grammar_rule.rule_id != GrammarType.ANone
            ]
        ):
            keep_group_by = True
        # if there is any one should be count in order by, we need group
        # WARNING: from the design principle of IRNet, we should notice
        # that there is no rule that group by should be assigned into order.
        # Here is only a judgement to identify whether to use.
        elif any(
            [
                node
                for node in order_nodes
                if node.grammar_rule.rule_id != GrammarType.ANone
            ]
        ):
            keep_group_by = True

        # for group by between two tables
        used_tab_names = [
            node.grammar_rule.rule_id for node in component_mapping["from"]
        ]

        if not keep_group_by:
            return
        else:
            group_by_clause = None
            from_nodes = component_mapping["from"]

            if len(from_nodes) != 2:
                # TODO: if contains table > 2, we select a column which has no agg as group by
                #  the algorithm may be unstable, but we now use it.

                for node in select_nodes:
                    if node.grammar_rule.rule_id == GrammarType.ANone:
                        # TODO: no more mapping
                        agg_repr = self._process_agg(node, {"from": []})
                        group_by_clause = f"GROUP BY {agg_repr}"
                        break
                # if all have aggregation
                # TODO: where is ORDER BY ?
                if group_by_clause is None and len(having_nodes) > 0:
                    # without any aggregator
                    for agg_node in select_nodes:
                        col_name = agg_node.child[0].grammar_rule.rule_id
                        tab_name = agg_node.child[1].grammar_rule.rule_id
                        if col_name == "*":
                            continue
                        agg_repr = f"{tab_name}.{col_name}"
                        group_by_clause = f"GROUP BY {agg_repr}"
                        break
            # TODO: rule-based. When there are two tables, we should group by via foreign keys
            else:
                if len(select_nodes) == 1 and len(having_nodes) == 0:
                    # TODO: check the linking
                    pass

                # find foreign key
                for key_tab_name, key_col_name, val_tab_name, _ in self.foreign_pairs:
                    if (
                        key_tab_name in used_tab_names
                        and val_tab_name in used_tab_names
                    ):
                        agg_repr = f"{key_tab_name}.{key_col_name}"
                        assert key_col_name != "*"
                        group_by_clause = f"GROUP BY {agg_repr}"
                        break

                # if having, select the column in select as the group by one
                if group_by_clause is None:
                    for node in select_nodes:
                        if node.grammar_rule.rule_id == GrammarType.ANone:
                            agg_repr = self._process_agg(node, {"from": []})
                            if agg_repr == "*":
                                continue
                            group_by_clause = f"GROUP BY {agg_repr}"
                            break
                    if group_by_clause is None:
                        # remove having
                        if "having" in repr:
                            repr.pop("having")

            if group_by_clause is not None:
                # for separate group by and others
                repr["group"] = group_by_clause
            else:
                return

    def _process_statement(self, node: Optional[GrammarRuleTreeNode]) -> str:
        """
        Process statement node and return the SQL clause of statement
        :return: SQL clause equal to node
        """
        action = node.grammar_rule
        action_type = action.rule_id
        assert isinstance(action, Statement)
        if action_type == GrammarType.StateNone:
            assert len(node.child) == 1
            root_repr = self._process_root(node.child[0], False)
            return root_repr
        else:
            # two children
            assert len(node.child) == 2
            left_child = self._process_root(node.child[0], False)
            right_child = self._process_root(node.child[1], False)
            if action_type == GrammarType.StateInter:
                return f"{left_child} INTERSECT {right_child}"
            elif action_type == GrammarType.StateExcept:
                return f"{left_child} EXCEPT {right_child}"
            elif action_type == GrammarType.StateUnion:
                return f"{left_child} UNION {right_child}"
            else:
                raise ValueError(f"Not support for statement type:{action_type}")

    def _process_root(self, node: Optional[GrammarRuleTreeNode], is_subquery):
        """
        Process root node and return the root representation
        :param node:
        :return:
        """
        # traverse node child
        assert isinstance(node.grammar_rule, Root)
        component_mapping: Dict[str, List[GrammarRuleTreeNode]] = {
            "select": [],
            "where": [],
            "having": [],
            "order": [],
            "from": [],
        }

        repr_mapping: Dict[str, str] = {}

        for node_son in node.child:
            action_cls = node_son.grammar_rule.__class__
            # must in Select, Order or Filter
            assert action_cls in [Select, Order, Filter, Join]
            process_func = self.processor[action_cls]
            process_func(node_son, component_mapping, repr_mapping)
        # process group by
        # TODO: here we assume that group by could occur in sub-queries.
        self._process_group_by(component_mapping, repr_mapping)

        # TODO: if is subquery, we should explain ON clause explicitly
        self._process_join(component_mapping, repr_mapping, is_subquery)

        action_repr = ""

        # handle them in order
        for component_key in ["select", "from", "where", "group", "having", "order"]:
            if component_key in repr_mapping:
                action_repr += repr_mapping[component_key] + " "

        action_repr = action_repr.strip()
        return action_repr

    def _process_order(
        self,
        node: Optional[GrammarRuleTreeNode],
        component: Dict[str, List[GrammarRuleTreeNode]],
        repr: Dict[str, str],
    ):
        """
        Process order by clause
        """

        assert isinstance(node.grammar_rule, Order)
        assert len(node.child) == 1
        assert isinstance(node.child[0].grammar_rule, A)
        agg_repr = self._process_agg(node.child[0], component)
        basic_repr = f"ORDER BY {agg_repr} "
        action_type = node.grammar_rule.rule_id
        if action_type == GrammarType.OrderAsc:
            basic_repr += "ASC"
        elif action_type == GrammarType.OrderDes:
            basic_repr += "DESC"
        elif action_type == GrammarType.OrderAscLim:
            basic_repr += "ASC LIMIT 1"
        elif action_type == GrammarType.OrderDesLim:
            basic_repr += "DESC LIMIT 1"
        else:
            raise ValueError(f"Not support for order type:{action_type}")
        repr["order"] = basic_repr
        component["order"] = node.child

    def _process_sep_join(
        self,
        node: Optional[GrammarRuleTreeNode],
        component: Dict[str, List[GrammarRuleTreeNode]],
        repr: Dict[str, str],
    ):
        """
        process separate join path and return nothing
        :return: modifiy component and insert another join table into it
        """
        assert isinstance(node.grammar_rule, Join)
        for child in node.child:
            # append table name into join path
            self._process_agg(child, component)

    def _process_select(
        self,
        node: Optional[GrammarRuleTreeNode],
        component: Dict[str, List[GrammarRuleTreeNode]],
        repr: Dict[str, str],
    ):
        """
        process select clause and return the select clause
        :return: modifiy repr and make the key `select` as SELECT representation
        """
        assert isinstance(node.grammar_rule, Select)
        agg_reprs = []
        for child in node.child:
            agg_reprs.append(self._process_agg(child, component))
        action_repr = ",".join(agg_reprs)
        repr["select"] = f"SELECT {action_repr}"
        component["select"] = node.child

    def _process_filter(
        self,
        node: Optional[GrammarRuleTreeNode],
        component: Dict[str, List[GrammarRuleTreeNode]],
        repr: Dict[str, str],
    ):
        """
        Process filter, return where clause and having clause
        :param node: root node of Filter
        :return: modifies on repr to return the representation of different components
        """

        def _mark_node_type(cur_node: GrammarRuleTreeNode, keep_having: bool):
            """
            Split current node into two separate trees.
            :param keep_having: specify whether to keep having nodes
            """
            # deep copy a node
            for ind, child_node in enumerate(cur_node.child):
                action_type = child_node.grammar_rule.rule_id
                # if root node, do not mark
                if isinstance(child_node.grammar_rule, Root):
                    continue
                if isinstance(child_node.grammar_rule, Filter):
                    _mark_node_type(child_node, keep_having)
                    continue
                # if it is A node and mark where
                if action_type != GrammarType.ANone:
                    # action max/min... -> having, not keep having -> where
                    if not keep_having:
                        # assign having node as None
                        cur_node.child[ind] = None
                    else:
                        # add having node into current mapping
                        component["having"].append(child_node)
                elif action_type == GrammarType.ANone:
                    # action none -> where, keep having -> having
                    if keep_having:
                        cur_node.child[ind] = None
                    else:
                        component["where"].append(child_node)

        # get two separate root nodes
        where_root_node = deepcopy(node)
        _mark_node_type(where_root_node, False)

        having_root_node = deepcopy(node)
        _mark_node_type(having_root_node, True)

        def _recursive_repr(
            _inner_node: Optional[GrammarRuleTreeNode]
        ) -> Optional[str]:
            """
            Recursively represent the _inner_node
            :return: string or None(if all marked as None)
            """
            assert isinstance(_inner_node.grammar_rule, Filter)
            action_type = _inner_node.grammar_rule.rule_id

            if (
                action_type == GrammarType.FilterAnd
                or action_type == GrammarType.FilterOr
            ):
                # recursive find filter
                assert len(_inner_node.child) == 2

                left_repr = None
                right_repr = None

                if _inner_node.child[0] is not None:
                    # left repr and right repr
                    left_repr = _recursive_repr(_inner_node.child[0])

                if _inner_node.child[1] is not None:
                    right_repr = _recursive_repr(_inner_node.child[1])

                # return AND OR and etc.
                if left_repr and right_repr:
                    if action_type == GrammarType.FilterAnd:
                        return f"{left_repr} AND {right_repr}"
                    else:
                        return f"{left_repr} OR {right_repr}"
                # if right is None, means (AND WHERE HAVING)
                elif left_repr:
                    return left_repr
                elif right_repr:
                    return right_repr
                else:
                    return None
            # plain or subquery
            else:
                if action_type == GrammarType.FilterNotLike:
                    template = "{} NOT LIKE "
                elif action_type == GrammarType.FilterLike:
                    template = "{} LIKE "
                elif (
                    action_type == GrammarType.FilterEqual
                    or action_type == GrammarType.FilterEqualNes
                ):
                    template = "{} = "
                elif (
                    action_type == GrammarType.FilterGreater
                    or action_type == GrammarType.FilterGreaterNes
                ):
                    template = "{} > "
                elif (
                    action_type == GrammarType.FilterLess
                    or action_type == GrammarType.FilterLessNes
                ):
                    template = "{} < "
                elif (
                    action_type == GrammarType.FilterGeq
                    or action_type == GrammarType.FilterGeqNes
                ):
                    template = "{} >= "
                elif (
                    action_type == GrammarType.FilterLeq
                    or action_type == GrammarType.FilterLeqNes
                ):
                    template = "{} <= "
                elif (
                    action_type == GrammarType.FilterNeq
                    or action_type == GrammarType.FilterNeqNes
                ):
                    template = "{} != "
                elif (
                    action_type == GrammarType.FilterBetween
                    or action_type == GrammarType.FilterBetweenNes
                ):
                    template = "{} BETWEEN 1 AND "
                elif action_type == GrammarType.FilterInNes:
                    template = "{} IN "
                elif action_type == GrammarType.FilterNotInNes:
                    template = "{} NOT IN "
                else:
                    raise ValueError(
                        f"Error on Filter processing: not filter type: {len(action_type)}"
                    )

                assert len(_inner_node.child) >= 1

                if _inner_node.child[0] is not None:
                    assert isinstance(_inner_node.child[0].grammar_rule, A)

                    agg_repr = self._process_agg(_inner_node.child[0], component)

                    # judge as HAVING
                    if len(_inner_node.child) == 1:
                        return template.format(agg_repr) + "1"
                    # sub query
                    elif len(_inner_node.child) == 2:
                        assert isinstance(_inner_node.child[0].grammar_rule, A)
                        assert isinstance(_inner_node.child[1].grammar_rule, Root)
                        agg_repr = self._process_agg(_inner_node.child[0], component)
                        # sub-query start, allocate a new mapping dictionary
                        root_repr = self._process_root(
                            _inner_node.child[1], is_subquery=True
                        )
                        return template.format(agg_repr) + f"( {root_repr} )"
                    else:
                        raise ValueError(
                            f"Error on Filter processing: not supported child number: {len(_inner_node.child)}"
                        )
                else:
                    return None

        where_clause = _recursive_repr(where_root_node)
        having_clause = _recursive_repr(having_root_node)

        if where_clause:
            repr["where"] = f"WHERE {where_clause}"

        if having_clause:
            repr["having"] = f"HAVING {having_clause}"

        if not where_clause and not having_clause:
            raise ValueError(
                "There is no WHERE and HAVING, but there is an Filter Node."
            )

    @staticmethod
    def _process_agg(
        node: Optional[GrammarRuleTreeNode],
        mapping: Dict[str, List[GrammarRuleTreeNode]],
    ) -> str:
        """
        Process column, table and aggregation, return the representation
        :return: representation of aggregation
        """
        # process aggregation, column and table
        assert isinstance(node.grammar_rule, A)
        # C and T
        assert len(node.child) == 2
        # TODO: assum C is always before T
        assert isinstance(node.child[0].grammar_rule, C)
        assert isinstance(node.child[1].grammar_rule, T)
        col_name = node.child[0].grammar_rule.rule_id
        tab_name = node.child[1].grammar_rule.rule_id

        # TODO: we use Node instead of table name to keep consistent
        used_tab_names = [node.grammar_rule.rule_id for node in mapping["from"]]
        if tab_name not in used_tab_names:
            mapping["from"].append(node.child[1])

        # add tab_name into mapping
        action_type = node.grammar_rule.rule_id
        if action_type == GrammarType.ANone:
            if col_name == "*":
                return "*"
            else:
                return f"{tab_name}.{col_name}"
        else:
            template = ""
            if action_type == GrammarType.ACount:
                template = "count({})"
            elif action_type == GrammarType.AAvg:
                template = "avg({})"
            elif action_type == GrammarType.ASum:
                template = "sum({})"
            elif action_type == GrammarType.AMin:
                template = "min({})"
            elif action_type == GrammarType.AMax:
                template = "max({})"
            # * means direct return
            if col_name == "*":
                return template.format(col_name)
            else:
                return template.format(f"{tab_name}.{col_name}")

    def _build_graph(self):
        """
        Build the graph using primary/foregin key. Applied for multi-table scenario.
        :return:
        """
        # edges for building graph to find the shorted path
        relations = []
        foreign_pairs = []

        def _get_tab_col_name(full_name):
            name_parts = full_name.split(":")
            # the first one is type
            name_type = name_parts[0]
            if name_type != "column":
                return None, None
            col_type = name_parts[1]
            # TODO: single direction, make the key only FOREIGN
            if col_type not in ["foreign", "primary"]:
                return None, None
            # fetch the column name
            tab_name = name_parts[2]
            col_name = name_parts[3]
            return tab_name, col_name

        for (
            key_full_name
        ) in self.db_context.knowledge_graph.neighbors_with_table.keys():
            # get key name
            key_tab_name, key_col_name = _get_tab_col_name(key_full_name)
            if key_col_name is None:
                continue
            linked_value_set = self.db_context.knowledge_graph.neighbors_with_table[
                key_full_name
            ]
            for val_full_name in linked_value_set:
                val_tab_name, val_col_name = _get_tab_col_name(val_full_name)
                if val_col_name is None:
                    continue

                # else add them into graph
                relations.append((key_tab_name, val_tab_name))
                foreign_pairs.append(
                    (key_tab_name, key_col_name, val_tab_name, val_col_name)
                )

        return Graph(relations), foreign_pairs


if __name__ == "__main__":
    query = "Select * from class JOIN prof where age >= 43 and class.prof = prof.name"
    print(query)
    table_column1 = TableColumn("name", "name", "text", is_primary_key=True)
    table_column2 = TableColumn("location", "location", "text")
    table_column5 = TableColumn("prof", "prof", "text", foreign_key=["prof.name"])
    table_column3 = TableColumn("age", "age", "number")
    table_column4 = TableColumn("name", "name", "text", is_primary_key=True)
    table1 = Table("class", "class")
    table1.set_columns([table_column1, table_column2, table_column5])
    table2 = Table("prof", "prof")
    table2.set_columns([table_column4, table_column3])
    database_schema = DatabaseSchema("class", [table1, table2])
    converter = SQLConverter(database_schema)
    grammar = converter.convert_to_grammar_rules(query)
    print(grammar)

    action_converter = ActionConverter(database_schema)
    sql = action_converter.translate_to_sql([repr(rule) for rule in grammar])
    print(sql)
