# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from typing import Dict
from typing import List, Optional

from core.knowledge_base.database_schema import DatabaseSchema

Keywords = [
    "limit",
    "des",
    "asc",
    "and",
    "or",
    "sum",
    "min",
    "max",
    "avg",
    "none",
    "=",
    "!=",
    "<",
    ">",
    "<=",
    ">=",
    "between",
    "like",
    "not_like",
    "in",
    "not_in",
    "intersect",
    "union",
    "except",
    "none",
    "count",
    "ins",
]


class SpecialSymbol:
    copy_delimiter = " [COPY] "


class GrammarAction(object):
    grammar_dict = {}

    def __init__(self):
        self.ins_id = None
        self.rule = None

    def get_next_action(self, is_sketch: bool = False):
        actions = []
        # a rule always first mentions the action you are currently performing
        # 1: to remove this from the list of next actions
        for x in self.rule.split(" ")[1:]:
            if x not in Keywords:
                rule_type = eval(x)
                if is_sketch:
                    if rule_type is not A and rule_type is not T:
                        actions.append(rule_type)
                else:
                    actions.append(rule_type)
        return actions

    def __repr__(self):
        space_index = self.rule.find(" ")
        return f"{self.rule[:space_index]} -> {self.rule[space_index + 1:]}"

    def is_global(self):
        """
        Actions are global means they fit for the whole dataset, while others only
        fit for specific instances
        :return:
        """
        if self.__class__ in [C, T, Segment]:
            return False
        else:
            return True

    def __lt__(self, other):
        return self.__repr__() < other.__repr__()

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        return self.__repr__() == other.__repr__()

    @staticmethod
    def from_str(action_repr: str):
        """
        Build an action object from string
        :param action_repr: the representation of action
        :return: Action object
        """
        # the from_str ONLY can be used in non-copy scenario
        lhs, rhs = action_repr.split(" -> ")
        # eval class object
        cls_obj = eval(lhs)
        if cls_obj in [C, T]:
            return cls_obj(rhs)
        else:
            # find the rule id
            rule_str = " ".join([lhs, rhs])
            grammar_dict: Dict = cls_obj.grammar_dict
            rule_id = list(grammar_dict.keys())[
                list(grammar_dict.values()).index(rule_str)
            ]
            return cls_obj(rule_id)

    @property
    def is_nonterminal(self):
        """
        Here we use a simple but not robust method to judge whether self is a nonterminal action (Select),
        or a terminal action (C/T)
        :return:
        """
        if isinstance(self.ins_id, int):
            return True
        else:
            return False

    @property
    def nonterminal(self):
        return self.__class__.__name__


class ActionTreeNode(object):
    def __init__(self, action: GrammarAction):
        self.action = action
        self.child: List[Optional[ActionTreeNode]] = []
        # drop self
        if isinstance(self.action.ins_id, int):
            all_child = self.action.grammar_dict[self.action.ins_id].split(" ")[1:]
        else:
            all_child = []
        for child_name in all_child:
            if child_name not in Keywords:
                # placeholder
                self.child.append(None)

    def full_in_child(self) -> bool:
        """
        test if an action could be inserted into self's child, if fail, return false; otherwise, return true.
        :return:
        """
        # if is a non terminal
        if None in self.child:
            return False
        # successfully add the child, return true.
        return True

    def add_child(self, action_node):
        ind = self.child.index(None)
        self.child[ind] = action_node

    def get_tree_action(self) -> List[GrammarAction]:
        if self.action.is_nonterminal:
            sub_tree = [self.action]
            # FIXME: here we use a simple method to extract all subtrees from current root node:
            #  call all nodes' get_sub_tree. A better way is to backtrack and construct all subtrees
            #  using dynamic programming.
            for child in self.child:
                sub_tree.extend(child.get_tree_action())
            return sub_tree
        else:
            return [self.action]


class GrammarType:
    """
    Filter Grammar Type
    """

    FilterBetween = 1
    FilterEqual = 2
    FilterGreater = 3
    FilterLess = 4
    FilterGeq = 5
    FilterLeq = 6
    FilterNeq = 7
    FilterInNes = 8
    FilterNotInNes = 9
    FilterLike = 10
    FilterNotLike = 11
    FilterIs = 12
    FilterExist = 13

    # TODO: in and like does not have a nested version
    FilterNotNes = 14
    FilterBetweenNes = 15
    FilterEqualNes = 16
    FilterGreaterNes = 17
    FilterLessNes = 18
    FilterGeqNes = 19
    FilterLeqNes = 20
    FilterNeqNes = 21
    FilterIsNes = 22
    FilterExistNes = 23

    FilterAnd = 24
    FilterOr = 25
    # FilterNone = 26

    """
    Statement Grammar Type
    """
    StateInter = 1
    StateUnion = 2
    StateExcept = 3
    StateNone = 4

    """
    Root Grammar Type
    """
    RootSFO = 1
    RootSO = 2
    RootSF = 3
    RootS = 4

    RootJSFO = 5
    RootJSO = 6
    RootJSF = 7
    RootJS = 8

    """
    Select Grammar Type depends on the length of A
    """

    """
    Join Grammar Type depends on the length of A
    """

    """
    A Grammar Type
    """
    ANone = 1
    AMax = 2
    AMin = 3
    ACount = 4
    ASum = 5
    AAvg = 6

    """
    Order Grammar Type
    """
    OrderNone = 1
    OrderAsc = 2
    OrderDes = 3
    OrderAscLim = 4
    OrderDesLim = 5


class Grammar(object):
    def __init__(self, database_schema: DatabaseSchema):
        self.database_schema = database_schema

        self.rules = []

        self.build_production_map(Statement)
        self.build_production_map(Root)
        self.build_production_map(Join)
        self.build_production_map(Select)
        self.build_production_map(A)
        self.build_production_map(Filter)
        self.build_production_map(Order)

        self.local_grammar = self.build_instance_production()

    @classmethod
    def build_ast_tree(cls, action_seq: List[GrammarAction]):
        # action is the depth-first traversal
        node_queue: List[ActionTreeNode] = []
        root_node = None
        seq_len = len(action_seq)
        for i in range(seq_len):
            # build tree node
            tree_node = ActionTreeNode(action_seq[i])
            if i == 0:
                root_node = tree_node
            # try to append current node into the first element of node queue
            else:
                cur_node = node_queue[-1]
                # cannot insert, pop the least node
                while cur_node.full_in_child():
                    # break the first node
                    node_queue.pop(-1)
                    # update current node
                    cur_node = node_queue[-1]
                cur_node.add_child(tree_node)
            node_queue.append(tree_node)
        return root_node

    @classmethod
    def extract_all_subtree(cls, action_seq: List[GrammarAction]) -> List:
        """
        Given the root node of ast tree, return all the valid subtrees
        :return:
        """
        nonterminal_node_list: List[ActionTreeNode] = []
        # store root node into queue
        node_queue: List[ActionTreeNode] = []
        seq_len = len(action_seq)
        for i in range(seq_len):
            # build tree node
            tree_node = ActionTreeNode(action_seq[i])
            # try to append current node into the first element of node queue
            if i == 0:
                pass
            # try to append current node into the first element of node queue
            else:
                cur_node = node_queue[-1]
                # cannot insert, pop the least node
                while cur_node.full_in_child():
                    # break the first node
                    node_queue.pop(-1)
                    # update current node
                    cur_node = node_queue[-1]
                cur_node.add_child(tree_node)

            node_queue.append(tree_node)
            # add note into node list
            if tree_node.action.is_nonterminal:
                nonterminal_node_list.append(tree_node)
        # build tree end, get all subtrees
        subtree_list = [node.get_tree_action() for node in nonterminal_node_list]
        return subtree_list

    def build_production_map(self, cls):
        """Record the rules of class cls into class of Action."""
        # (note) the values could provide a fixed order
        # only when the dictionary is built on
        prod_ids = cls.grammar_dict.keys()
        for prod_id in prod_ids:
            cls_obj = cls(prod_id)
            self.rules.append(cls_obj)

    def build_instance_production(self):
        """Instance all possible column and table rules using the database schema."""
        tables = self.database_schema.tables

        local_grammars = [T(table.name) for table in tables]

        all_columns = set()
        for table in tables:
            all_columns.update([C(column.name) for column in table.columns])

        column_grammars = list(all_columns)
        local_grammars.extend(column_grammars)

        # convert into set and sorted
        local_grammars = set(local_grammars)
        # sorted local grammars
        local_grammars = sorted(local_grammars)

        return local_grammars

    @property
    def global_grammar(self):
        return sorted(self.rules)

    @staticmethod
    def default_sql_clause() -> Dict:
        default_sql = {
            "orderBy": [],
            "from": {"table_units": [["table_unit", 1]], "conds": []},
            "union": None,
            "except": None,
            "groupBy": None,
            "limit": None,
            "intersect": None,
            "where": [],
            "having": [],
            "select": [False, [[3, [0, [0, 5, False], None]]]],
        }
        return default_sql


class Statement(GrammarAction):
    grammar_dict = {
        GrammarType.StateInter: "Statement intersect Root Root",
        GrammarType.StateUnion: "Statement union Root Root",
        GrammarType.StateExcept: "Statement except Root Root",
        GrammarType.StateNone: "Statement Root",
    }

    def __init__(self, id_c):
        super().__init__()
        self.ins_id = id_c
        self.production = self.grammar_dict[id_c]


class Root(GrammarAction):
    grammar_dict = {
        GrammarType.RootSFO: "Root Select Filter Order",
        GrammarType.RootSF: "Root Select Filter",
        GrammarType.RootSO: "Root Select Order",
        GrammarType.RootS: "Root Select",
        GrammarType.RootJSFO: "Root Join Select Filter Order",
        GrammarType.RootJSF: "Root Join Select Filter",
        GrammarType.RootJSO: "Root Join Select Order",
        GrammarType.RootJS: "Root Join Select",
    }

    def __init__(self, id_c: int):
        super().__init__()
        self.ins_id = id_c
        self.production = self.grammar_dict[id_c]


class Select(GrammarAction):
    grammar_dict = {
        0: "Select A",
        1: "Select A A",
        2: "Select A A A",
        3: "Select A A A A",
        4: "Select A A A A A",
        5: "Select A A A A A A",
    }

    def __init__(self, id_c):
        super().__init__()
        self.ins_id = id_c
        self.production = self.grammar_dict[id_c]


class Join(GrammarAction):
    grammar_dict = {
        0: "Join A",
        # 1: 'Join A A'
    }

    def __init__(self, id_c):
        super().__init__()
        self.ins_id = id_c
        self.production = self.grammar_dict[id_c]


class A(GrammarAction):
    grammar_dict = {
        GrammarType.ANone: "A none C T",
        GrammarType.AMax: "A max C T",
        GrammarType.AMin: "A min C T",
        GrammarType.ACount: "A count C T",
        GrammarType.ASum: "A sum C T",
        GrammarType.AAvg: "A avg C T",
    }

    def __init__(self, id_c):
        super().__init__()
        self.ins_id = id_c
        self.production = self.grammar_dict[id_c]


class Filter(GrammarAction):
    # TODO: why not directly predict the number of Filters
    grammar_dict = {
        GrammarType.FilterAnd: "Filter Filter and Filter",
        GrammarType.FilterOr: "Filter Filter or Filter",
        GrammarType.FilterEqual: "Filter = A",
        GrammarType.FilterGreater: "Filter > A",
        GrammarType.FilterLess: "Filter < A",
        GrammarType.FilterGeq: "Filter >= A",
        GrammarType.FilterLeq: "Filter <= A",
        GrammarType.FilterNeq: "Filter != A",
        GrammarType.FilterBetween: "Filter between A",
        # TODO: like/not_like only apply to string type
        GrammarType.FilterLike: "Filter like A",
        GrammarType.FilterNotLike: "Filter not_like A",
        GrammarType.FilterEqualNes: "Filter = A Root",
        GrammarType.FilterGreaterNes: "Filter > A Root",
        GrammarType.FilterLessNes: "Filter < A Root",
        GrammarType.FilterGeqNes: "Filter >= A Root",
        GrammarType.FilterLeqNes: "Filter <= A Root",
        GrammarType.FilterNeqNes: "Filter != A Root",
        GrammarType.FilterBetweenNes: "Filter between A Root",
        GrammarType.FilterInNes: "Filter in A Root",
        GrammarType.FilterNotInNes: "Filter not_in A Root",
    }

    def __init__(self, id_c):
        super().__init__()
        self.ins_id = id_c
        self.production = self.grammar_dict[id_c]


class Order(GrammarAction):
    grammar_dict = {
        GrammarType.OrderAsc: "Order asc A",
        GrammarType.OrderDes: "Order des A",
        GrammarType.OrderAscLim: "Order asc A limit",
        GrammarType.OrderDesLim: "Order des A limit",
    }

    def __init__(self, ins_id):
        super().__init__()
        self.ins_id = ins_id
        self.production = self.grammar_dict[ins_id]


class C(GrammarAction):
    def __init__(self, ins_id: str):
        super().__init__()
        # TODO: here we lower it because the col -> id (entities_names) in SparcWorld is the lower key-value pair.
        self.ins_id = ins_id.lower()
        self.production = f"C {self.ins_id}"


class T(GrammarAction):
    def __init__(self, ins_id: str):
        super().__init__()
        self.ins_id = ins_id.lower()
        self.production = f"T {self.ins_id}"


class Segment(GrammarAction):
    """
    segment action appears only in the training post-processing. it is used to copy segment-level precedent SQL
    """

    def __init__(self, copy_ins_action: List[GrammarAction], copy_ins_idx: List[int]):
        super().__init__()
        self.copy_ins_action = copy_ins_action
        # copy ins idx has been padded
        self.copy_ins_idx = copy_ins_idx
        self.production = f"Copy {self.ins_id}"

    def __repr__(self):
        repr_str = SpecialSymbol.copy_delimiter + SpecialSymbol.copy_delimiter.join(
            [str(action) for action in self.copy_ins_action]
        )
        return repr_str

    # the nonterminal is the first one
    @property
    def nonterminal(self):
        # get the terminal of the first action string
        first_action = self.copy_ins_action[0]
        return first_action.nonterminal
