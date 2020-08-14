# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from typing import Dict, Text
from typing import List, Optional

from rasa.core.knowledge_base.schema.database_schema import DatabaseSchema

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


class GrammarRule(object):
    """
    A GrammarRule describes something like
      Root -> Select Filter
    or
      Select -> Agg Agg
    """

    grammar_dict = {}

    def __init__(self):
        self.rule_id = None
        self.rule = None

    def get_next_action(self, is_sketch: bool = False) -> List["GrammarRule"]:
        rules = []

        # a rule always first mentioned the current rule, e.g. Root Select Filter
        # '1:' to remove this from the list of next actions

        for x in self.rule.split(" ")[1:]:
            if x not in Keywords:
                rule_type = eval(x)
                if is_sketch:
                    if rule_type is not A and rule_type is not T:
                        rules.append(rule_type)
                else:
                    rules.append(rule_type)
        return rules

    def __repr__(self):
        space_index = self.rule.find(" ")
        return f"{self.rule[:space_index]} -> {self.rule[space_index + 1:]}"

    def is_global(self) -> bool:
        """GrammarRules are global means they fit for the whole dataset, while others
        only fit for specific instances.
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
    def from_nonterminal(rule_string: Text) -> "GrammarRule":
        return eval(rule_string)

    @staticmethod
    def from_string(rule_string: Text) -> "GrammarRule":
        """Create a GrammarRule from a string.

        The string looks like 'Root -> Select Filter'

        Args:
            rule_string: the string representing the grammar rule

        Returns:
            A GrammarRule
        """
        # the rule_string ONLY can be used in non-copy scenario
        lhs, rhs = rule_string.split(" -> ")

        # eval class object
        rule_type = eval(lhs)
        if rule_type in [C, T]:
            return rule_type(rhs)

        # find the rule id
        rule_str = " ".join([lhs, rhs])
        grammar_dict = rule_type.grammar_dict
        rule_id = list(grammar_dict.keys())[list(grammar_dict.values()).index(rule_str)]
        return rule_type(rule_id)

    @property
    def is_nonterminal(self) -> bool:
        """Determine whether this rule is nonterminal or not.

        A nonterminal rule has an instance id of type int.

        Returns:
            True, if this rule is a terminal rule, False, otherwise.
        """
        if isinstance(self.rule_id, int):
            return True
        else:
            return False

    @property
    def nonterminal(self) -> Text:
        return self.__class__.__name__


class GrammarRuleTreeNode(object):
    def __init__(self, grammar_rule: GrammarRule):
        self.grammar_rule = grammar_rule
        self.child: List[Optional[GrammarRuleTreeNode]] = []

        # drop self
        if isinstance(self.grammar_rule.rule_id, int):
            all_child = self.grammar_rule.grammar_dict[self.grammar_rule.rule_id].split(
                " "
            )[1:]
        else:
            all_child = []

        for child_name in all_child:
            if child_name not in Keywords:
                # placeholder
                self.child.append(None)

    def full_in_child(self) -> bool:
        """Test if a grammar rule could be inserted into self's child, if fail,
        return false; otherwise, return true.
        """
        # if is a non terminal
        if None in self.child:
            return False

        # successfully add the child, return true.
        return True

    def add_child(self, action_node) -> None:
        ind = self.child.index(None)
        self.child[ind] = action_node

    def get_tree_action(self) -> List[GrammarRule]:
        if self.grammar_rule.is_nonterminal:
            sub_tree = [self.grammar_rule]
            # FIXME: here we use a simple method to extract all subtrees from current root node:
            #  call all nodes' get_sub_tree. A better way is to backtrack and construct all subtrees
            #  using dynamic programming.
            for child in self.child:
                sub_tree.extend(child.get_tree_action())
            return sub_tree
        else:
            return [self.grammar_rule]


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
    def build_syntax_tree(cls, rule_sequence: List[GrammarRule]):
        # action is the depth-first traversal
        node_queue: List[GrammarRuleTreeNode] = []
        root_node = None
        seq_len = len(rule_sequence)
        for i in range(seq_len):
            # build tree node
            tree_node = GrammarRuleTreeNode(rule_sequence[i])
            if i == 0:
                root_node = tree_node
            # try to append current node into the first element of node queue
            else:
                current_node = node_queue[-1]
                # cannot insert, pop the least node
                while current_node.full_in_child():
                    # break the first node
                    node_queue.pop(-1)
                    # update current node
                    current_node = node_queue[-1]
                current_node.add_child(tree_node)
            node_queue.append(tree_node)

        return root_node

    @classmethod
    def extract_all_subtree(cls, action_seq: List[GrammarRule]) -> List:
        """
        Given the root node of syntax tree, return all the valid subtrees
        :return:
        """
        nonterminal_node_list: List[GrammarRuleTreeNode] = []
        # store root node into queue
        node_queue: List[GrammarRuleTreeNode] = []
        seq_len = len(action_seq)
        for i in range(seq_len):
            # build tree node
            tree_node = GrammarRuleTreeNode(action_seq[i])
            # try to append current node into the first element of node queue
            if i == 0:
                pass
            # try to append current node into the first element of node queue
            else:
                current_node = node_queue[-1]
                # cannot insert, pop the least node
                while current_node.full_in_child():
                    # break the first node
                    node_queue.pop(-1)
                    # update current node
                    current_node = node_queue[-1]
                current_node.add_child(tree_node)

            node_queue.append(tree_node)
            # add note into node list
            if tree_node.grammar_rule.is_nonterminal:
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


class Statement(GrammarRule):
    grammar_dict = {
        GrammarType.StateInter: "Statement intersect Root Root",
        GrammarType.StateUnion: "Statement union Root Root",
        GrammarType.StateExcept: "Statement except Root Root",
        GrammarType.StateNone: "Statement Root",
    }

    def __init__(self, rule_id):
        super().__init__()
        self.rule_id = rule_id
        self.rule = self.grammar_dict[rule_id]


class Root(GrammarRule):
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

    def __init__(self, rule_id: int):
        super().__init__()
        self.rule_id = rule_id
        self.rule = self.grammar_dict[rule_id]


class Select(GrammarRule):
    grammar_dict = {
        0: "Select A",
        1: "Select A A",
        2: "Select A A A",
        3: "Select A A A A",
        4: "Select A A A A A",
        5: "Select A A A A A A",
    }

    def __init__(self, rule_id):
        super().__init__()
        self.rule_id = rule_id
        self.rule = self.grammar_dict[rule_id]


class Join(GrammarRule):
    grammar_dict = {
        0: "Join A",
        # 1: 'Join A A'
    }

    def __init__(self, rule_id):
        super().__init__()
        self.rule_id = rule_id
        self.rule = self.grammar_dict[rule_id]


class A(GrammarRule):
    grammar_dict = {
        GrammarType.ANone: "A none C T",
        GrammarType.AMax: "A max C T",
        GrammarType.AMin: "A min C T",
        GrammarType.ACount: "A count C T",
        GrammarType.ASum: "A sum C T",
        GrammarType.AAvg: "A avg C T",
    }

    def __init__(self, rule_id):
        super().__init__()
        self.rule_id = rule_id
        self.rule = self.grammar_dict[rule_id]


class Filter(GrammarRule):
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

    def __init__(self, rule_id):
        super().__init__()
        self.rule_id = rule_id
        self.rule = self.grammar_dict[rule_id]


class Order(GrammarRule):
    grammar_dict = {
        GrammarType.OrderAsc: "Order asc A",
        GrammarType.OrderDes: "Order des A",
        GrammarType.OrderAscLim: "Order asc A limit",
        GrammarType.OrderDesLim: "Order des A limit",
    }

    def __init__(self, rule_id):
        super().__init__()
        self.ins_id = rule_id
        self.rule = self.grammar_dict[rule_id]


class C(GrammarRule):
    def __init__(self, rule_id: str):
        super().__init__()
        # TODO: here we lower it because the col -> id (entities_names) in SparcWorld is the lower key-value pair.
        self.rule_id = rule_id.lower()
        self.rule = f"C {self.rule_id}"


class T(GrammarRule):
    def __init__(self, rule_id: str):
        super().__init__()
        self.rule_id = rule_id.lower()
        self.rule = f"T {self.rule_id}"


class Segment(GrammarRule):
    """
    segment action appears only in the training post-processing. it is used to copy segment-level precedent SQL
    """

    def __init__(self, copy_ins_action: List[GrammarRule], copy_ins_idx: List[int]):
        super().__init__()
        self.copy_ins_action = copy_ins_action
        # copy ins idx has been padded
        self.copy_ins_idx = copy_ins_idx
        self.production = f"Copy {self.rule_id}"

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
