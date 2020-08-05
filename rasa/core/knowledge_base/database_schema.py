from collections import defaultdict
from typing import List, Optional, Text, Dict, Set

from sqlalchemy.util import OrderedSet

from core.knowledge_base.knowledge_graph_filed import KnowledgeGraph


class Table:
    """Representing the table."""

    def __init__(self, name: str, text: str, columns: Optional[List["TableColumn"]]):
        self.name = name
        self.text = text
        self.columns = columns or []

        for column in columns:
            column.refer_table = self


class TableColumn:
    """Representing the column of table."""

    def __init__(
        self,
        name: str,
        text: str,
        column_type: str,
        is_primary_key: bool = False,
        refer_table: Optional["Table"] = None,
        foreign_key: Optional[List[str]] = None,
    ):
        self.name = name
        self.text = text
        self.column_type = column_type
        self.is_primary_key = is_primary_key
        self.foreign_key = foreign_key or []
        self.refer_table = refer_table

    def __str__(self):
        return f"{self.name}"


class DatabaseSchema:
    def __init__(
        self,
        name: Text,
        tables: Optional[List[Table]],
        columns: Optional[List[TableColumn]],
    ):
        self.name = name
        self.tables = tables or []
        self.columns = columns or []
        self.knowledge_graph = self.get_db_knowledge_graph()

    def id_to_tables(self) -> Dict[int, Table]:
        return {i: table for i, table in enumerate(self.tables)}

    def tables_to_id(self) -> Dict[int, Table]:
        return {table.name: i for i, table in enumerate(self.tables)}

    def id_to_columns(self) -> Dict[int, TableColumn]:
        map = {i: column for i, column in enumerate(self.columns, start=1)}
        map[0] = TableColumn("*", "*", "text")
        return map

    def columns_to_id(self) -> Dict[int, TableColumn]:
        map = {
            column.refer_table.name + "." + column.name: i
            for i, column in enumerate(self.columns, start=1)
        }
        map.update({column.name: i for i, column in enumerate(self.columns, start=1)})
        map["*"] = 0
        return map

    def columns_names_of_table(self, table_name):
        for table in self.tables:
            if table.name != table_name:
                continue

            return [column.name for column in table.columns]

    @staticmethod
    def entity_key_for_column(column: TableColumn) -> str:
        if column.foreign_key is not None:
            column_type = "foreign"
        elif column.is_primary_key:
            column_type = "primary"
        else:
            column_type = column.column_type
        # FIXME: here we assume the same column name always returns the same text & entity
        return f"column:{column_type.lower()}:{column.name.lower()}"

    @staticmethod
    def entity_key_for_column_with_table(table_name: str, column: TableColumn) -> str:
        if column.foreign_key is not None:
            column_type = "foreign"
        elif column.is_primary_key:
            column_type = "primary"
        else:
            column_type = column.column_type
        # FIXME: here we assume the same column name always returns the same text & entity
        return (
            f"column:{column_type.lower()}:{table_name.lower()}:{column.name.lower()}"
        )

    def get_db_knowledge_graph(self) -> KnowledgeGraph:
        entities: Set[str] = set()
        # TODO: here we use two different neighbors graph: the first is used to extract potential features;
        #  the second is used to build join graph;
        neighbors = defaultdict(OrderedSet)
        neighbors_with_table = defaultdict(OrderedSet)
        entity_text: Dict[str, str] = {}

        for table in self.tables:
            table_key = f"table:{table.name.lower()}"
            if table_key not in entities:
                entities.add(table_key)
            entity_text[table_key] = table.text

            for column in self.columns:
                entity_key = self.entity_key_for_column(column)
                entity_key_with_table = self.entity_key_for_column_with_table(
                    table.name, column
                )
                if entity_key not in entities:
                    entities.add(entity_key)
                entity_text[entity_key] = column.text

                neighbors[entity_key].add(table_key)
                neighbors[table_key].add(entity_key)
                neighbors_with_table[entity_key_with_table].add(table_key)
                neighbors_with_table[table_key].add(entity_key_with_table)

        # sort entities in alpha-beta order. Now entities is a List
        entities: List[str] = sorted(list(entities))

        # loop again after we have gone through all columns to link foreign keys columns
        for table in self.tables:
            for column in table.columns:
                if column.foreign_key is None:
                    continue

                for foreign_key in column.foreign_key:
                    other_column_table, other_column_name = foreign_key.split(":")

                    # must have exactly one by design
                    other_column = [
                        col
                        for col in self.columns
                        if col.name == other_column_name
                        and col.refer_table.name == other_column_table
                    ][0]

                    entity_key = self.entity_key_for_column_with_table(
                        table.name, column
                    )
                    other_entity_key = self.entity_key_for_column_with_table(
                        other_column_table, other_column
                    )

                    # model the relation between column and column
                    neighbors_with_table[entity_key].add(other_entity_key)
                    neighbors_with_table[other_entity_key].add(entity_key)

        kg = KnowledgeGraph(
            entities, dict(neighbors), dict(neighbors_with_table), entity_text
        )

        return kg
