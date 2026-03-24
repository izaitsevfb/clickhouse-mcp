#!/usr/bin/env python3
"""
PyTorch HUD MCP server entry point
"""

import os

# When CLICKHOUSE_MCP_NO_PROXY=1, strip HTTP(S)_PROXY env vars before any
# network libraries are imported.  This is useful in corporate environments
# where a system-level proxy (e.g. x2pagentd) intercepts HTTPS traffic with
# a TLS library too old to negotiate with ClickHouse Cloud.
if os.environ.get("CLICKHOUSE_MCP_NO_PROXY", "").strip() in ("1", "true", "yes"):
    for _k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ.pop(_k, None)

from clickhouse_mcp.mcp_server import mcp

def main():
    """Main entry point for the application"""
    print("Starting PyTorch Clickhouse MCP server...")
    mcp.run()

if __name__ == "__main__":
    main()