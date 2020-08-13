from collections import defaultdict
from typing import List, Optional, Text, Dict, Set, Any

from sqlalchemy.util import OrderedSet

from rasa.core.knowledge_base.schema.knowledge_graph_filed import KnowledgeGraph


class Table:
    """Representing the table."""

    def __init__(self, name: str, alias: str):
        self.name = name
        self.alias = alias
        self.columns = []

    def set_columns(self, columns: List["TableColumn"]) -> None:
        self.columns.extend(columns)

        for column in self.columns:
            column.refer_table = self

    def get_full_column_names(self) -> List[Text]:
        return [f"{self.name}.{column.name}" for column in self.columns]


class TableColumn:
    """Representing the column of table."""

    def __init__(
        self,
        name: str,
        alias: str,
        column_type: str,
        is_primary_key: bool = False,
        refer_table: Optional["Table"] = None,
        foreign_key: Optional[List[str]] = None,
    ):
        self.name = name
        self.alias = alias
        self.column_type = column_type
        self.is_primary_key = is_primary_key
        self.foreign_key = foreign_key or []
        self.refer_table = refer_table

    def __str__(self):
        return f"{self.name}"

    def full_name(self):
        return f"{self.refer_table.name}.{self.name}"


class DatabaseSchema:
    def __init__(self, name: Text, tables: Optional[List[Table]]):
        self.name = name
        self.tables = tables or []
        self.columns = []
        for table in tables:
            self.columns.append(TableColumn("*", "*", "any", refer_table=table))
            for column in table.columns:
                self.columns.append(column)

        self.knowledge_graph = self.get_db_knowledge_graph()

        self.id_to_tables = {i: table for i, table in enumerate(self.tables)}
        self.id_to_columns = {i: column for i, column in enumerate(self.columns)}

    def table_names_to_id(self) -> Dict[int, Table]:
        return {table.name: id for id, table in self.id_to_tables.items()}

    def column_names_to_id(self) -> Dict[int, TableColumn]:
        map = {column.full_name(): id for id, column in self.id_to_columns.items()}
        map.update({column.name: id for id, column in self.id_to_columns.items()})
        return map

    def columns_names_of_table(self, table_name: Text) -> List[Text]:
        for table in self.tables:
            if table.name == table_name:
                return [column.name for column in table.columns]
        return []

    @staticmethod
    def entity_key_for_column(column: TableColumn) -> Text:
        if column.foreign_key is not None:
            column_type = "foreign"
        elif column.is_primary_key:
            column_type = "primary"
        else:
            column_type = column.column_type
        return f"column:{column_type.lower()}:{column.name.lower()}"

    @staticmethod
    def entity_key_for_column_with_table(table_name: Text, column: TableColumn) -> Text:
        if column.foreign_key is not None:
            column_type = "foreign"
        elif column.is_primary_key:
            column_type = "primary"
        else:
            column_type = column.column_type
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
            entity_text[table_key] = table.alias

            for column in self.columns:
                entity_key = self.entity_key_for_column(column)
                entity_key_with_table = self.entity_key_for_column_with_table(
                    table.name, column
                )
                if entity_key not in entities:
                    entities.add(entity_key)
                entity_text[entity_key] = column.alias

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
                    other_column_table, other_column_name = foreign_key.split(".")

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

        return KnowledgeGraph(
            entities, dict(neighbors), dict(neighbors_with_table), entity_text
        )

    @classmethod
    def from_dict(cls, database_schema_dict: Dict[Text, Any]) -> "DatabaseSchema":
        database_name = database_schema_dict["name"]
        tables = []
        for table_dict in database_schema_dict["tables"]:
            table_name = table_dict["name"]

            table = Table(table_name, table_name)

            columns = []
            for column_dict in table_dict["columns"]:
                primary_key = False
                if "primary_key" in column_dict:
                    primary_key = column_dict["primary_key"]
                foreign_key = []
                if "foreign_key" in column_dict:
                    foreign_key = [column_dict["foreign_key"]]

                columns.append(
                    TableColumn(
                        column_dict["name"],
                        column_dict["name"],
                        column_dict["type"],
                        primary_key,
                        foreign_key=foreign_key,
                    )
                )

            table.set_columns(columns)
            tables.append(table)

        return DatabaseSchema(database_name, tables)

    def to_dict(self) -> Dict[Text, Any]:
        database_schema_dict = {"name": self.name, "tables": []}

        for table in self.tables:
            table_dict = {"name": table.name, "columns": []}
            for column in table.columns:
                column_dict = {"name": column.name, "type": column.column_type}
                if column.is_primary_key:
                    column_dict["primary_key"] = True
                if column.foreign_key:
                    column_dict["foreign_key"] = column.foreign_key[0]

                table_dict["columns"].append(column_dict)

            database_schema_dict["tables"].append(table_dict)

        return database_schema_dict
