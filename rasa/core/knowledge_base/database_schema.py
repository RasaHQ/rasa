from typing import List, Optional, Text


class Table:
    """Representing the table."""

    def __init__(self, name: str, text: str, columns: List["TableColumn"]):
        self.name = name
        self.text = text
        self.columns = columns


class TableColumn:
    """Representing the column of table."""

    def __init__(
        self,
        name: str,
        text: str,
        column_type: str,
        is_primary_key: bool,
        refer_table: "Table",
        foreign_key: Optional[List[str]],
    ):
        self.name = name
        self.text = text
        self.column_type = column_type
        self.is_primary_key = is_primary_key
        self.foreign_key = foreign_key
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
