"""MCP utilities."""


def mcp_server_exists(mcp_server: str) -> bool:
    """Check if an MCP server exists in the configured endpoints.

    Args:
        mcp_server: The name of the MCP server to check.

    Returns:
        True if the MCP server exists, False otherwise.
    """
    from rasa.core.config.configuration import Configuration

    endpoints = Configuration.get_instance().endpoints
    if (mcp_server_list := endpoints.mcp_servers) is None:
        return False

    mcp_server_names = [server.name for server in mcp_server_list]
    return mcp_server in mcp_server_names
