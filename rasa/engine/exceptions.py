class GraphRunError(Exception):
    """Exception class for errors originating when running a graph."""


class GraphComponentException(Exception):
    """Exception class for errors originating within a `GraphComponent`."""


class GraphSchemaException(Exception):
    """Represents errors when dealing with `GraphSchema`s."""


class GraphSchemaValidationException(Exception):
    """Indicates that the given graph schema is invalid."""
