import clickhouse_connect.driver.ctypes
import clickhouse_connect.driver.types
import clickhouse_connect.driverc
from . import DEFAULT_BEDROCK_MODEL, DEFAULT_REGION
from .vector_search import load_faiss_index, vector_search, get_default_index_path
from langchain_community.vectorstores import FAISS
from langchain_aws import BedrockEmbeddings
import clickhouse_connect
import json
import datetime
from typing import Optional, Any, Dict, List
import clickhouse_connect.common
import clickhouse_connect.driver
import clickhouse_connect.driver.client
import clickhouse_connect.driver.exceptions
import clickhouse_connect.driver.query
from mcp.server.fastmcp import FastMCP, Context
import os
import time
import requests
import sqlfluff
import tempfile

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

# Import required modules for vector search


# Create an MCP server
mcp = FastMCP("PyTorch ClickHouse MCP")

clickhouse_client: Optional[clickhouse_connect.driver.client.Client] = None

# Maximum response size in bytes - set to a conservative size for context window efficiency
MAX_RESPONSE_SIZE = 10 * 1024  # 10KB

# datetime serializer for JSON


def datetime_serializer(obj):
    if isinstance(obj, datetime.datetime):
        return obj.isoformat()
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    raise TypeError(f"Type {type(obj)} not serializable")


def clickhouse_response_to_json(res: clickhouse_connect.driver.query.QueryResult) -> str:
    """Convert ClickHouse query result to JSON string.

    Args:
        res (clickhouse_connect.driver.query.QueryResult): The ClickHouse query result

    Returns:
        str: The JSON string representation of the query result
    """
    if res is None or res.result_rows is None:
        return "No data returned from the query."

    return {"column_names": res.column_names, "result_rows": res.result_rows}


def get_clean_error_string(clickhouse_error_msg: str) -> str:
    """Clean the ClickHouse error message for better readability.

    Args:
        clickhouse_error_msg (str): The raw ClickHouse error message

    Returns:
        str: The cleaned error message
    """
    # Remove unnecessary details from the error message - everything before "received ClickHouse error"
    if not "received ClickHouse error" in clickhouse_error_msg:
        return clickhouse_error_msg
    return "Clickhouse error " + clickhouse_error_msg.split("received ClickHouse error")[-1].strip()


def get_clickhouse_client() -> clickhouse_connect.driver.client.Client:
    """Get the ClickHouse client instance.

    Returns:
        clickhouse_connect.Client: The ClickHouse client.

    Raises:
        ValueError: If any required environment variables are missing
        Exception: If connection to ClickHouse fails
    """
    global clickhouse_client
    if clickhouse_client is None:
        # Check for required environment variables
        host = os.getenv("CLICKHOUSE_HOST")
        port = os.getenv("CLICKHOUSE_PORT")
        username = os.getenv("CLICKHOUSE_USER")
        password = os.getenv("CLICKHOUSE_PASSWORD")

        if not all([host, port, username, password]):
            missing = []
            if not host:
                missing.append("CLICKHOUSE_HOST")
            if not port:
                missing.append("CLICKHOUSE_PORT")
            if not username:
                missing.append("CLICKHOUSE_USER")
            if not password:
                missing.append("CLICKHOUSE_PASSWORD")
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}")

        clickhouse_client = clickhouse_connect.get_client(
            host=host,
            port=port,
            username=username,
            password=password,
            database="default",
            secure=True
        )
    return clickhouse_client


def safe_json_dumps(data: Any, indent: int = 2, max_size: int = MAX_RESPONSE_SIZE) -> str:
    """Safely serialize data to JSON with strict size limit.

    Args:
        data: The data to serialize
        indent: Indentation level for pretty printing
        max_size: Maximum response size in bytes

    Returns:
        JSON string, truncated if needed with a warning message
    """
    # Always use indentation for readability
    json_str = json.dumps(data, indent=indent)

    # Return as-is if under the size limit
    if len(json_str) <= max_size:
        return json_str

    # Hard truncate with a clear error message
    warning_msg = (
        "\n\n<RESPONSE TRUNCATED>\n"
        "The response exceeds the maximum size limit. Please use more specific parameters or pagination.\n"
    )

    # Calculate safe truncation size
    trunc_size = max_size - len(warning_msg)
    if trunc_size < 200:  # Ensure we have some minimal content
        trunc_size = 200

    # Return truncated JSON with error message
    return json_str[:trunc_size] + warning_msg


@mcp.tool()
def readme_howto_use_clickhouse_tools() -> str:
    """Returns a guide on how to use the ClickHouse tools.
    """
    return (
        """
        Clickhouse is a column based database used in PyTorch CI.
        It is used to store and query test results and other data.

        This tool allows you to run queries, explain queries, and get the schema of the ClickHouse database.

        Supported tools:

        - `run_clickhouse_query` Run an actual query and return result + timing. Use the measure_performance flag for detailed metrics. Note that the query can't have a ; at the end.
        - `get_clickhouse_schema` Get the schema of a ClickHouse table.
        - `get_query_execution_stats` Get the list of slow queries from the ClickHouse system table.
        - `get_clickhouse_tables` Get the list of tables from a specific database or all databases. Use the database parameter to specify a database (default is 'default') or set databases='all' to query tables from all available databases (default, benchmark, misc).
        - `explain_clickhouse_query` Explain a ClickHouse query using multiple optional EXPLAIN types (PLAN, PIPELINE, ESTIMATE) and return the combined results.
        - `get_query_details` Get the ClickHouse query by its name. Also optionally return the list of parameters from params.json and the performance samples from the ClickHouse system table.
        - `semantic_search_docs` Perform a semantic search over ClickHouse documentation.
        - `lint_clickhouse_query` Lint a ClickHouse SQL query using SQLFluff.


        Use this to create and run queries if user asks things like: 'what's the slowest query? How can I query X in ClickHouse?'
        """)


@mcp.tool()
def run_clickhouse_query(query: str, inline_result_limit_bytes: int = 1024, measure_performance: bool = False) -> Dict[str, Any]:
    """Runs a ClickHouse query and returns the result as a JSON (and optionally a file).
    Args:
        query (str): The ClickHouse query to execute
        inline_result_limit_bytes (int): Maximum bytes for inline result rows (default: 1024, max: 10240)
        measure_performance (bool): Whether to measure and log precise query performance (increases function call wall time,
            but gives precise runtime and memory usage metrics)

    Returns:
        result['time']: The time taken to execute the query
        result['result_rows']: Array of result rows that fit within the byte limit (at least one row is always returned)
        result['total_result_rows_n']: Total number of rows returned by the query
        result['columns']: The number of columns returned by the query
        result['error']: An error message if the query failed
        result['hash']: The hash of the query result
        result['performance']: (Optional) Detailed performance metrics if measure_performance is True
        result['result_file']: (Optional, if enabled) A file containing the result of the query as a JSON string

    """
    start_time = time.time()  # Start timing the query execution

    # Enforce hard max limit of 10KB
    inline_result_limit_bytes = min(inline_result_limit_bytes, 10240)

    if not query or not query.strip():
        return {
            "error": "Error: Empty query provided",
        }

    try:
        client = get_clickhouse_client()
    except ValueError as e:
        # Handle missing environment variables
        return {
            "error": f"Configuration error: {str(e)}",
        }
    except Exception as e:
        # Handle connection errors
        return {
            "error": f"Connection error: Failed to connect to ClickHouse server: {str(e)}",
        }

    try:
        query = query.replace("\\n", "\n").strip()

        if query.endswith(";"):
            query = query[:-1].strip()

        if measure_performance:
            # disable caching to get accurate performance measurements
            settings = "settings enable_filesystem_cache = 0, use_query_cache = false"
            query += f" {settings}"

        res: Optional[clickhouse_connect.driver.query.QueryResult] = client.query(
            query)
        end_time = time.time()  # End timing the query execution

        column_names = res.column_names
        # dump to json, with converting datetime to string
        json_result = json.dumps(
            res.result_rows, indent=2, default=datetime_serializer)

        # Check if tmp file generation is disabled
        disable_tmp_files = os.getenv(
            "CLICKHOUSE_DISABLE_TMP_FILES", "false").lower() == "true"
        filename = None

        if not disable_tmp_files:
            # Save to a temporary file - generate the filename to be unique
            filename = f"/tmp/clickhouse_query_result_{res.query_id}.json"

            try:
                with open(filename, "w") as f:
                    f.write(json_result)  # Write the JSON result to the file
            except IOError as e:
                return {
                    "time": end_time - start_time,
                    "columns": column_names,
                    "error": f"File system error: Failed to write result to file: {str(e)}",
                }

        json_data = json.loads(json_result)

        # Limit result rows by byte size
        limited_rows = []
        current_size = 0
        size_limit_exceeded = False

        for row in json_data:
            row_json = json.dumps(row, default=datetime_serializer)
            row_size = len(row_json.encode('utf-8'))

            if not limited_rows or current_size + row_size <= inline_result_limit_bytes:
                limited_rows.append(row)
                current_size += row_size
            else:
                size_limit_exceeded = True
                break

        # Build the base result
        result = {
            "time": end_time - start_time,
            "result_rows": limited_rows,
            "total_result_rows_n": len(json_data),
            "columns": column_names,
            "query_id": res.query_id,
        }

        # Only include result_file if tmp file was created
        if filename is not None:
            result["result_file"] = filename

        if size_limit_exceeded:
            result["warning"] = (
                f"Inline result rows limited to {len(limited_rows)} rows due to size limit of {inline_result_limit_bytes} bytes. "
                "`result_file` has the full results." if filename else "Request fewer rows or increase the limit to get more results."
            )

        # If performance measurement is enabled, fetch the query log data
        if measure_performance:
            perf_result = None
            # wait for 1 minute max
            for _ in range(60 // 5):
                try:
                    # Wait a moment to ensure query_log gets populated
                    time.sleep(5)

                    # Query the system.query_log table for detailed performance metrics
                    # Only use columns we have confirmed access to
                    perf_query = f"""
                    SELECT
                        event_time,
                        query_duration_ms,
                        memory_usage
                    FROM system.query_log
                    WHERE query_id = '{res.query_id}'
                      AND type = 'QueryFinish'
                    LIMIT 1
                    """

                    perf_result = client.query(perf_query)

                    if perf_result and perf_result.result_rows:
                        row = perf_result.result_rows[0]
                        result["performance"] = {
                            "duration_ms": row[1],
                            "memory_usage": row[2]
                        }
                        break
                except Exception as e:
                    result[
                        "performance_error"] = f"Failed to retrieve performance data: {str(e)}"
                    break
            if not perf_result or not perf_result.result_rows:
                result["performance_error"] = "Performance data search timed out"

        return result

    except clickhouse_connect.driver.exceptions.ClickHouseError as e:
        return {
            "time": time.time() - start_time,
            "error": f"Query error: {get_clean_error_string(str(e))}",
        }
    except Exception as e:
        return {
            "time": time.time() - start_time,
            "error": f"Unexpected error during query execution: {str(e)}",
        }


@mcp.tool()
def get_clickhouse_schema(table_name: str) -> str:
    """Get the schema of a ClickHouse table.

    Args:
        table_name (str): The name of the table to get the schema for

    Returns:
        str: The schema of the table as a JSON string with columns and create table statement
    """
    client = get_clickhouse_client()
    try:
        # Get table columns (name and type only)
        columns_result = client.query(f"DESCRIBE TABLE {table_name}")
        if columns_result is None or columns_result.result_rows is None or len(columns_result.result_rows) == 0:
            return "No data returned from the query."

        # Extract only name and type from the DESCRIBE result
        columns = [{"name": row[0], "type": row[1]}
                   for row in columns_result.result_rows]

        # Get CREATE TABLE statement
        create_table_result = client.query(f"SHOW CREATE TABLE {table_name}")
        create_table = create_table_result.result_rows[0][
            0] if create_table_result and create_table_result.result_rows else ""

        # Build the complete result
        schema_info = {
            "columns": columns,
            "create_table_statement": create_table
        }

        # Convert to JSON string with size limit
        json_result = safe_json_dumps(
            schema_info, indent=2, max_size=128 * 1024)
        return json_result
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def get_query_execution_stats(last_x_hours: int, limit: int = 10, query_name: Optional[str] = None) -> str:
    """Get the list of queries with their execution timings from the ClickHouse system table.
    Args:
        last_x_hours (int): The number of hours to look back for slow queries
        query_name (Optional[str]): Optional filter for specific query names
        limit (int): The maximum number of slow queries to return
    Returns:
        str: The name of the file containing the slow queries as a JSON string
    """
    query_name_filter = ""
    if query_name:
        # Ensure the query_id is sanitized to prevent SQL injection
        query_name_filter = f"""
        AND query_id LIKE '%{query_name}%' 
            """

    QUERY = f"""
    SELECT
        round(avg(query_duration_ms)) AS realTimeMSAvg,
        sum(query_duration_ms) as realTimeMSTotal,
        round(quantile(0.5)(query_duration_ms)) as realTimeMSP50,
        avg(memory_usage) as memoryBytesAvg,
        sum(memory_usage) as memoryBytesTotal,
        quantile(0.5)(memory_usage) as memoryBytesP50,
        count(*) as num,
        left(query_id, -37) as name
    FROM
        clusterAllReplicas(default, system.query_log)
    WHERE
        event_time >= now() - INTERVAL {last_x_hours} HOUR
        AND event_time < now()
        AND initial_user = 'hud_user'
        AND length(query_id) > 37
        AND type = 'QueryFinish'
        AND left(query_id, -37) != 'adhoc'
        {query_name_filter}
    GROUP BY
        name
    ORDER BY 
        realTimeMSAvg DESC
    LIMIT 
        {limit}
    """

    client = get_clickhouse_client()
    try:
        res = client.query(QUERY.format(last_x_hours=last_x_hours))
        if res is None or res.result_rows is None or len(res.result_rows) == 0:
            return "No data returned from the query."
        # Find top 5 slowest

        return safe_json_dumps(clickhouse_response_to_json(res), indent=2)
    except clickhouse_connect.driver.exceptions.ClickHouseError as e:
        return f"Clickhouse query error: {get_clean_error_string(str(e))}"
    except Exception as e:
        return f"Unexpected error: {e}"


@mcp.tool()
def explain_clickhouse_query(
    query: str,
    explain_plan: bool = False,
    explain_pipeline: bool = False,
    explain_estimate: bool = False
) -> Dict[str, Any]:
    """Explain a ClickHouse query using multiple optional EXPLAIN types.

    Note: expects a valid SELECT query, function prepends it with `EXPLAIN {query}` internally.

    Args:
        query (str): The ClickHouse query to explain
        explain_plan (bool): Whether to include EXPLAIN PLAN with actions=1 and indexes=1
        explain_pipeline (bool): Whether to include EXPLAIN PIPELINE with graph=1
        explain_estimate (bool): Whether to include EXPLAIN ESTIMATE

        simple EXPLAIN {query} is always run, plus any additional explain types requested

    Returns:
        Dict[str, Any]: Dictionary containing results from each requested EXPLAIN type
    """
    client = get_clickhouse_client()
    result = {}

    try:
        # Default EXPLAIN (query plan) - always run this
        try:
            res = client.query(f"EXPLAIN {query}")
            if res is not None and res.result_rows is not None and len(res.result_rows) > 0:
                result["default_explain"] = res.result_rows
        except Exception as e:
            result["default_explain_error"] = f"Error: {str(e)}"

        # EXPLAIN PLAN with actions and indexes
        if explain_plan:
            try:
                plan_res = client.query(
                    f"EXPLAIN PLAN actions=1, indexes=1 {query}")
                if plan_res is not None and plan_res.result_rows is not None and len(plan_res.result_rows) > 0:
                    result["explain_plan"] = plan_res.result_rows
            except Exception as e:
                result["explain_plan_error"] = f"Error: {str(e)}"

        # EXPLAIN PIPELINE with graph
        if explain_pipeline:
            try:
                pipe_res = client.query(f"EXPLAIN PIPELINE graph=1 {query}")
                if pipe_res is not None and pipe_res.result_rows is not None and len(pipe_res.result_rows) > 0:
                    result["explain_pipeline"] = pipe_res.result_rows
            except Exception as e:
                result["explain_pipeline_error"] = f"Error: {str(e)}"

        # EXPLAIN ESTIMATE
        if explain_estimate:
            try:
                est_res = client.query(f"EXPLAIN ESTIMATE {query}")
                if est_res is not None and est_res.result_rows is not None and len(est_res.result_rows) > 0:
                    result["explain_estimate"] = est_res.result_rows
                elif est_res is not None and est_res.column_names is not None:
                    # EXPLAIN ESTIMATE might return empty rows but with column names
                    result["explain_estimate"] = {
                        "columns": est_res.column_names,
                        "rows": [],
                        "note": "No estimate data returned, query might be too simple or not supported for estimation"
                    }
            except Exception as e:
                result["explain_estimate_error"] = f"Error: {str(e)}"

        # Return the dictionary result directly
        if not result:
            return {"error": "No explain data returned from any of the explain methods."}

        return result
    except Exception as e:
        return {"error": f"Error during explain operations: {str(e)}"}


@mcp.tool()
def get_clickhouse_tables(database: Optional[str] = "default", databases: Optional[str] = None) -> str:
    """Get the list of tables in the ClickHouse database.

    Args:
        database (Optional[str]): The database to query tables from. Defaults to 'default'.
        databases (Optional[str]): If set to 'all', returns tables from all available databases (default, benchmark, misc).
                                  This parameter takes precedence over the database parameter.

    Returns:
        str: The list of tables as a JSON string, grouped by database if multiple are requested
    """
    client = get_clickhouse_client()
    try:
        if databases == "all":
            # Query tables from all supported databases
            dbs_to_query = ["default", "benchmark", "misc"]
            all_tables = {}

            for db in dbs_to_query:
                res = client.query(f"SHOW TABLES FROM {db}")
                if res is not None and res.result_rows is not None and len(res.result_rows) > 0:
                    # Group tables by database
                    all_tables[db] = res.result_rows
                else:
                    all_tables[db] = []

            # Convert to JSON string with size limit
            json_result = safe_json_dumps(all_tables, indent=2)
            return json_result
        else:
            # Query tables from a single database
            res = client.query(f"SHOW TABLES FROM {database}")
            if res is None or res.result_rows is None or len(res.result_rows) == 0:
                return f"No tables found in database '{database}'."

            # Convert to JSON string with size limit
            json_result = safe_json_dumps(res.result_rows, indent=2)
            return json_result
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def get_query_details(query_name: str, include_params: Optional[bool] = True, include_performance_samples: int = 1) -> str:
    """ Get the ClickHouse query by its name. ALso optionally return the list of parameters from params.json and the performance samples from the ClickHouse system table.

    Args:
        query_name (str): The name of the query to get as referenced in pytorch/test-infra
        include_params (bool): Whether to include the parameters from params.json in the query
        include_performance_samples (int): The number of performance samples to include from the actual ClickHouse query execution (0 to disable). It contains the actual paramater values used in the query execution.
    Returns:
        str: A JSON string containing the query, parameters, and performance samples
    """
    result_json = {}

    query_url = f"https://raw.githubusercontent.com/pytorch/test-infra/refs/heads/main/torchci/clickhouse_queries/{query_name}/query.sql"
    params_url = f"https://raw.githubusercontent.com/pytorch/test-infra/refs/heads/main/torchci/clickhouse_queries/{query_name}/params.json"

    try:
        query = requests.get(query_url).text
        # Fix literal '\n' character sequences (2 chars) with actual newlines
        query = query.replace("\\n", "\n").strip()
        result_json["query"] = query
    except Exception as e:
        result_json["error"] = f"Error: Failed to get query from {query_url}. {e}"
        return json.dumps(result_json)

    # get the params from the url if include_params is True
    if include_params:
        try:
            params = requests.get(params_url).text
            params_dict = json.loads(params)
            result_json["params"] = params_dict
        except Exception as e:
            result_json["error"] = f"Error: Failed to get params from {params_url}. {e}"

    # get the performance samples from the ClickHouse system table
    if include_performance_samples > 0:
        try:
            COLUMN_NAMES = ["event_time", "query_id",
                            "query_duration_ms", "memory_usage", "query"]
            performance_query = f"""
            SELECT {', '.join(COLUMN_NAMES)}
            FROM clusterAllReplicas(default, system.query_log)
            WHERE left(query_id, -37) = '{query_name}' 
            AND type = 'QueryFinish'
            ORDER BY event_time DESC
            LIMIT {include_performance_samples}
            """
            client = get_clickhouse_client()
            res = client.query(performance_query)
            if res and res.result_rows:
                result_json["performance_samples"] = [
                    dict(zip(COLUMN_NAMES, row)) for row in res.result_rows
                ]

        except Exception as e:
            result_json[
                "error"] = f"Error: Failed to get performance samples from ClickHouse. {e}"

    return json.dumps(result_json, default=datetime_serializer, indent=2)


# Vector store singleton
vector_store_instance: Optional[FAISS] = None


def get_vector_store() -> FAISS:
    """Get or initialize the vector store singleton.

    Returns:
        FAISS: The loaded vector store
    """
    global vector_store_instance

    if vector_store_instance is None:
        # Initialize embeddings
        embeddings = BedrockEmbeddings(
            region_name=DEFAULT_REGION,
            model_id=DEFAULT_BEDROCK_MODEL
        )

        # Load the FAISS index
        index_path = get_default_index_path()
        vector_store_instance = load_faiss_index(index_path, embeddings)

    return vector_store_instance


@mcp.tool()
def semantic_search_docs(
    query: str,
    page: int = 1,
    per_page: int = 3,
    limit: int = 300
) -> str:
    """Performs semantic search over ClickHouse documentation.

    Args:
        query: The search query text
        page: Page number (starting from 1)
        per_page: Number of results per page
        limit: Maximum character length for each result's content

    Returns:
        The formatted search results with markdown content
    """
    try:
        # Get the vector store singleton
        vector_store = get_vector_store()

        # Calculate pagination offsets
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page

        # We need to fetch enough results to cover the requested page
        num_results = end_idx

        # Perform the semantic search
        search_results = vector_search(vector_store, query, num_results)

        # Apply pagination
        paginated_results = search_results[start_idx:end_idx]

        # Prepare result text with clear delimiters
        results_text = f"Search results for: '{query}'\n\n"
        results_text += f"Page {page} of {(len(search_results) + per_page - 1) // per_page} "
        results_text += f"({len(search_results)} total results)\n\n"

        # Format each result with delimiters
        for i, result in enumerate(paginated_results):
            results_text += f"==== RESULT {i+1} ====\n"
            results_text += f"DOCUMENT: {result.metadata.get('document_title', 'Unknown')}\n"
            results_text += f"SECTION: {result.metadata.get('section_title', 'Unknown')}\n"
            results_text += f"SOURCE: {result.metadata.get('path', 'Unknown')}\n"
            results_text += "CONTENT:\n"

            # Include content with truncation if needed
            content = result.page_content
            if limit and len(content) > limit:
                content = content[:limit] + "...(truncated)"

            results_text += content + "\n"
            results_text += "==================\n\n"

        # Add pagination instructions
        if page > 1:
            results_text += "For previous results: Use page=" + \
                str(page-1) + "\n"
        if page < (len(search_results) + per_page - 1) // per_page:
            results_text += "For more results: Use page=" + str(page+1) + "\n"

        return results_text

    except FileNotFoundError:
        return "ERROR: FAISS index not found. Please make sure the index has been created."
    except Exception as e:
        return f"ERROR: Search failed: {str(e)}"


@mcp.tool()
def lint_clickhouse_query(
    query: str,
    rule_exclude: Optional[str] = None
) -> Dict[str, Any]:
    """Lint a ClickHouse SQL query using SQLFluff.

    Args:
        query (str): The ClickHouse SQL query to lint
        rule_exclude (Optional[str]): Comma-separated list of rules to exclude (e.g. "L001,L002")

    Returns:
        Dict containing:
        - 'status': Whether the query passed linting ('pass' or 'fail')
        - 'errors_count': Number of linting errors found
        - 'errors': List of linting errors with details
        - 'formatted_query': Formatted SQL query (only if different from input)
    """
    if not query or not query.strip():
        return {
            "status": "fail",
            "errors_count": 1,
            "errors": ["Empty query provided"],
            "formatted_query": None
        }

    try:
        # Set up config for sqlfluff
        config = {
            "dialect": "clickhouse"
        }
        if rule_exclude:
            config["exclude_rules"] = rule_exclude.split(",")

        # Create a temporary file for better error reporting
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sql', delete=False) as tmp_file:
            tmp_file.write(query)
            tmp_path = tmp_file.name

        try:
            # Lint the query
            linting_result = sqlfluff.lint(tmp_path, config=config)

            # Format the query
            formatted_result = sqlfluff.fix(tmp_path, config=config)
            formatted_query = formatted_result.get("fix_str", query)

            # Process the linting results
            errors = []
            for violation in linting_result:
                errors.append({
                    "rule": violation.rule_code(),
                    "description": violation.description(),
                    "line": violation.line_no,
                    "position": violation.line_pos,
                    "context": violation.line_str if hasattr(violation, "line_str") else None
                })

            # Return structured results
            is_passing = len(errors) == 0

            # Only include formatted_query if it's different from input and there were errors
            formatted_query_output = None
            if not is_passing and formatted_query != query:
                formatted_query_output = formatted_query

            return {
                "status": "pass" if is_passing else "fail",
                "errors_count": len(errors),
                "errors": errors,
                "formatted_query": formatted_query_output
            }
        finally:
            # Clean up the temporary file
            os.unlink(tmp_path)

    except Exception as e:
        return {
            "status": "error",
            "errors_count": 1,
            "errors": [f"Linting error: {str(e)}"],
            "formatted_query": None
        }


# Run the server if executed directly
if __name__ == "__main__":
    mcp.run()
