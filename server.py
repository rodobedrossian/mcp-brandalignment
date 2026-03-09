import os, sys, re, time
from contextlib import contextmanager
from typing import List, Tuple, Dict, Optional

from dotenv import load_dotenv
load_dotenv()

try:
    from mcp.server.fastmcp import FastMCP
    # Nota: devolvemos dicts MCP en vez de types.TextContent para evitar validaciones
except Exception as e:
    print("Error importando 'mcp'. Instalá mcp[fastmcp].", file=sys.stderr)
    raise

import mysql.connector

# RAG integration (optional)
try:
    import requests
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    print("Warning: 'requests' not available. RAG enrichment will be disabled.", file=sys.stderr)

try:
    from scripts.notion_semantic_layer import get_notion_semantic_layer_content
    SEMANTIC_LAYER_AVAILABLE = True
except Exception as e:
    SEMANTIC_LAYER_AVAILABLE = False
    print(f"Warning: Semantic-layer integration unavailable: {e}", file=sys.stderr)


# === Config obligatoria ===
def need(name: str) -> str:
    v = os.getenv(name, "")
    if not v:
        print(f"FALTA variable obligatoria: {name}", file=sys.stderr)
        sys.exit(1)
    return v

MYSQL_HOST = need("MYSQL_HOST")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = need("MYSQL_USER")
MYSQL_PASSWORD = need("MYSQL_PASSWORD")
MYSQL_DATABASE = need("MYSQL_DATABASE")

# === Config opcional ===
MCP_MAX_ROWS = int(os.getenv("MCP_MAX_ROWS", "200"))
MCP_QUERY_TIMEOUT = int(os.getenv("MCP_QUERY_TIMEOUT", "15"))
ALLOWED_TABLES = [t.strip() for t in os.getenv("MCP_ALLOWED_TABLES", "").split(",") if t.strip()]
DENIED_COLUMNS = {c.strip().lower() for c in os.getenv("MCP_DENIED_COLUMNS", "").split(",") if c.strip()}

# === RAG Configuration (optional) ===
RAG_ENABLED = os.getenv("RAG_ENABLED", "true").lower() == "true"
RAG_API_URL = os.getenv("RAG_API_URL", "http://localhost:8000")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))
RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.5"))
RAG_TIMEOUT = int(os.getenv("RAG_TIMEOUT", "5"))  # seconds

app = FastMCP("mysql-mcp-local")

def get_conn():
    cn = mysql.connector.connect(
        host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER,
        password=MYSQL_PASSWORD, database=MYSQL_DATABASE,
        connection_timeout=10
    )
    try:
        cur = cn.cursor()
        cur.execute("SET SESSION TRANSACTION READ ONLY")
        cur.close()
    except Exception:
        pass
    return cn

@contextmanager
def mysql_cursor():
    cn = get_conn()
    try:
        try:
            cur = cn.cursor()
            # Si el servidor MySQL lo soporta: timeout por consulta (ms)
            cur.execute(f"SET SESSION MAX_EXECUTION_TIME={(MCP_QUERY_TIMEOUT * 1000)}")
            cur.close()
        except Exception:
            pass
        cur = cn.cursor()
        yield cn, cur
    finally:
        try:
            cur.close()
        except Exception:
            pass
        try:
            cn.close()
        except Exception:
            pass

def _mask_row(cols: List[str], row: Tuple) -> List[str]:
    out = []
    for col, val in zip(cols, row):
        out.append("***REDACTED***" if col.lower() in DENIED_COLUMNS else ("" if val is None else str(val)))
    return out

def _format_tsv(headers: List[str], rows: List[List[str]]) -> str:
    lines = ["\t".join(headers)]
    for r in rows:
        lines.append("\t".join(r))
    return "\n".join(lines)

# ==================== RAG INTEGRATION ====================

def _get_rag_context(entity_type: str, entity_id: str, entity_name: Optional[str] = None) -> Optional[str]:
    """
    Get Notion context for an entity (brand, product, or seller).
    
    Args:
        entity_type: Type of entity ("brand", "product", or "seller")
        entity_id: Entity ID (numeric ID or string identifier)
        entity_name: Optional entity name (for better context search)
    
    Returns:
        Formatted context string with Notion information, or None if unavailable.
    """
    if not RAG_ENABLED or not RAG_AVAILABLE:
        return None
    
    try:
        # Use entity name if available, otherwise use entity_id
        # URL encode the search_id to handle special characters
        from urllib.parse import quote
        search_id = entity_name if entity_name else str(entity_id)
        encoded_id = quote(str(search_id), safe='')
        
        # Build URL
        url = f"{RAG_API_URL}/context/{entity_type}/{encoded_id}"
        params = {
            "top_k": RAG_TOP_K,
            "min_score": RAG_MIN_SCORE,
            "namespace": "default",
            "use_rerank": "true"
        }
        
        # Make request with timeout
        response = requests.get(url, params=params, timeout=RAG_TIMEOUT)
        response.raise_for_status()
        
        data = response.json()
        context_items = data.get("context", [])
        
        if not context_items:
            return None
        
        # Format context as readable text
        context_lines = [f"=== Notion Context for {entity_type} '{search_id}' ==="]
        for i, item in enumerate(context_items, 1):
            title = item.get("title", "Untitled")
            text = item.get("text", "")
            item_url = item.get("url", "")
            score = item.get("score", 0)
            
            # Truncate text if too long (keep first 500 chars)
            if len(text) > 500:
                text = text[:500] + "..."
            
            context_lines.append(f"\n[{i}] {title} (score: {score:.2f})")
            if item_url:
                context_lines.append(f"    URL: {item_url}")
            context_lines.append(f"    {text}")
        
        return "\n".join(context_lines)
        
    except requests.exceptions.RequestException as e:
        # Log warning but don't fail the query
        print(f"Warning: RAG service unavailable: {e}", file=sys.stderr)
        return None
    except Exception as e:
        # Log error but don't fail the query
        print(f"Warning: Error getting RAG context: {e}", file=sys.stderr)
        return None


def _get_brand_name(brand_id: int) -> Optional[str]:
    """Get brand name from database."""
    try:
        with mysql_cursor() as (_cn, cur):
            cur.execute("SELECT name FROM brand WHERE id = %s", (brand_id,))
            row = cur.fetchone()
            return row[0] if row else None
    except Exception:
        return None


def _get_brand_context(brand_id: int, brand_name: Optional[str] = None) -> Optional[str]:
    """Get Notion context for a brand."""
    # Get brand name from database if not provided
    if not brand_name:
        brand_name = _get_brand_name(brand_id)
    return _get_rag_context("brand", str(brand_id), brand_name)


def _enrich_results_with_brand_context(results: list[dict], brand_id: int, log_tool_name: Optional[str] = None, log_inputs: Optional[dict] = None) -> list[dict]:
    """
    Enrich query results with brand context from Notion.
    
    Args:
        results: List of MCP result dicts (typically from _exec_tsv)
        brand_id: Brand ID to get context for
        log_tool_name: Optional tool name for logging
        log_inputs: Optional inputs dict for logging
    
    Returns:
        Results with brand context appended as additional text content.
    """
    if not RAG_ENABLED or not RAG_AVAILABLE:
        return results
    
    if not results or len(results) == 0:
        return results
    
    # Get brand context
    brand_context = _get_brand_context(brand_id)
    
    has_rag_context = False
    if brand_context:
        # Append context as additional text content
        results.append({"type": "text", "text": f"\n\n{brand_context}"})
        has_rag_context = True
    
    return results


def _enrich_sql_results_with_context(results: list[dict]) -> list[dict]:
    """
    Automatically enrich SQL query results with Notion context by analyzing the results.
    
    Analyzes the TSV results to find entities (brand_id, product_id, seller_id) and
    enriches with relevant Notion context.
    
    Args:
        results: List of MCP result dicts (typically from run_sql)
    
    Returns:
        Results with context appended as additional text content.
    """
    if not RAG_ENABLED or not RAG_AVAILABLE:
        return results
    
    if not results or len(results) == 0:
        return results
    
    # Get the TSV text from results
    tsv_text = results[0].get("text", "") if results else ""
    if not tsv_text or "\n" not in tsv_text:
        return results
    
    # Parse TSV to find entities
    lines = tsv_text.strip().split("\n")
    if len(lines) < 2:  # Need at least header + 1 data row
        return results
    
    headers = [h.strip().lower() for h in lines[0].split("\t")]
    
    # Find entity columns
    brand_id_idx = None
    brand_name_idx = None
    product_id_idx = None
    seller_id_idx = None
    
    for i, header in enumerate(headers):
        if header in ["brand_id", "brandid"]:
            brand_id_idx = i
        elif header in ["brand_name", "brandname", "brand"]:
            brand_name_idx = i
        elif header in ["product_id", "productid"]:
            product_id_idx = i
        elif header in ["seller_id", "sellerid"]:
            seller_id_idx = i
    
    # Collect unique entities from results
    brand_ids = set()
    product_ids = set()
    seller_ids = set()
    brand_names = {}  # Map brand_id to brand_name
    
    for line in lines[1:]:
        if not line.strip():
            continue
        cols = line.split("\t")
        
        # Extract brand_id
        if brand_id_idx is not None and len(cols) > brand_id_idx:
            brand_id_val = cols[brand_id_idx].strip()
            if brand_id_val and brand_id_val.isdigit():
                brand_ids.add(int(brand_id_val))
                # Store brand name if available
                if brand_name_idx is not None and len(cols) > brand_name_idx:
                    brand_name_val = cols[brand_name_idx].strip()
                    if brand_name_val:
                        brand_names[int(brand_id_val)] = brand_name_val
        
        # Extract product_id
        if product_id_idx is not None and len(cols) > product_id_idx:
            product_id_val = cols[product_id_idx].strip()
            if product_id_val and product_id_val.isdigit():
                product_ids.add(int(product_id_val))
        
        # Extract seller_id
        if seller_id_idx is not None and len(cols) > seller_id_idx:
            seller_id_val = cols[seller_id_idx].strip()
            if seller_id_val:
                seller_ids.add(seller_id_val)
    
    # Get context for found entities
    context_sections = []
    
    # Get brand context (prioritize brands)
    for brand_id in list(brand_ids)[:3]:  # Limit to first 3 brands to avoid too much context
        brand_name = brand_names.get(brand_id)
        context = _get_brand_context(brand_id, brand_name)
        if context:
            context_sections.append(context)
    
    # Get product context (if no brands found)
    if not brand_ids and product_ids:
        for product_id in list(product_ids)[:2]:  # Limit to first 2 products
            context = _get_rag_context("product", str(product_id))
            if context:
                context_sections.append(context)
    
    # Get seller context (if no brands or products found)
    if not brand_ids and not product_ids and seller_ids:
        for seller_id in list(seller_ids)[:2]:  # Limit to first 2 sellers
            context = _get_rag_context("seller", seller_id)
            if context:
                context_sections.append(context)
    
    # Append context to results
    if context_sections:
        context_text = "\n\n" + "\n\n".join(context_sections)
        results.append({"type": "text", "text": context_text})
    
    return results


def _append_semantic_layer_context(results: list[dict], topic: Optional[str]) -> list[dict]:
    """
    Append semantic-layer context from Notion based on the provided topic.

    Args:
        results: Existing MCP results (list of dicts with type/text).
        topic: High-level topic to use when searching Notion. If empty/None, no-op.

    Returns:
        Results list with semantic-layer context appended when available.
    """
    if not topic:
        return results
    if not SEMANTIC_LAYER_AVAILABLE:
        return results

    try:
        context = get_notion_semantic_layer_content(topic.strip())
    except Exception as exc:
        print(f"Warning: Unable to fetch Notion semantic layer for topic '{topic}': {exc}", file=sys.stderr)
        return results

    if not context:
        return results

    enriched = list(results)  # shallow copy to avoid mutating callers
    semantic_text = f"---\nSemantic Layer Context ({topic.strip()}):\n\n{context}"
    enriched.append({"type": "text", "text": semantic_text})
    return enriched


def _finalize_tool_output(results: list[dict], topic: Optional[str]) -> list[dict]:
    """
    Run all post-processing enrichments required before returning tool output.

    Currently this appends semantic-layer context when available.
    """
    return _append_semantic_layer_context(results, topic)

def _deny_non_select(sql: str) -> None:
    s = sql.strip().lower()
    if not (s.startswith("select ") or s.startswith("with ")):
        raise ValueError("Solo se permiten SELECT/CTE.")
    forbidden = [" update ", " delete ", " insert ", " drop ", " alter ", " truncate ", " create "]
    s2 = f" {s} "
    for bad in forbidden:
        if bad in s2:
            raise ValueError(f"Comando no permitido: {bad.strip().upper()}")

def _enforce_allowed_tables(sql: str) -> None:
    if not ALLOWED_TABLES:
        return
    tokens = re.findall(r"[`\"]?([a-zA-Z0-9_]+)[`\"]?(?:\s*\.\s*[`\"]?([a-zA-Z0-9_]+)[`\"]?)?", sql)
    probable = set()
    for a, b in tokens:
        probable.add((b or a).lower())
    non_tables = {
        "select","from","where","join","on","group","by","order","limit","offset","and","or","asc","desc",
        "count","sum","avg","min","max","distinct","as","with","over","partition","case","when","then","else","end",
        "inner","left","right","full","outer","union","all","having","like","in","is","null","not","between","exists"
    }
    probable = {t for t in probable if t not in non_tables}
    allowed = {x.lower() for x in ALLOWED_TABLES}
    not_allowed = {t for t in probable if t not in allowed}
    if not_allowed:
        raise ValueError(
            f"Tablas no permitidas detectadas: {', '.join(sorted(not_allowed))}. "
            f"Permitidas: {', '.join(ALLOWED_TABLES)}"
        )

def _append_limit(sql: str, max_rows: int) -> str:
    s = sql.strip().rstrip(";")
    if re.search(r"\blimit\b\s+\d+", s, flags=re.IGNORECASE):
        return s
    return f"{s} LIMIT {max_rows}"

# ---------------- Tools ----------------

@app.tool()
def health() -> list[dict]:
    """Devuelve un contenido MCP válido."""
    try:
        with mysql_cursor() as (_cn, cur):
            cur.execute("SELECT 1")
            _ = cur.fetchall()
        return [{"type": "text", "text": "OK"}]
    except Exception as e:
        return [{"type": "text", "text": f"ERROR: {e}"}]

@app.tool()
def describe_schema() -> list[dict]:
    with mysql_cursor() as (_cn, cur):
        cur.execute("""
            SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE()
            ORDER BY TABLE_NAME, ORDINAL_POSITION
        """)
        rows = cur.fetchall()
    lines = []
    for t, c, d in rows:
        red = " [REDACTED]" if c.lower() in DENIED_COLUMNS else ""
        lines.append(f"{t}.{c} {d}{red}")
    text = "\n".join(lines) if lines else "(Sin columnas)"
    return [{"type": "text", "text": text}]

@app.tool()
def run_sql(query: str, max_rows: int = MCP_MAX_ROWS, topic: str | None = None) -> list[dict]:
    if not query or not isinstance(query, str):
        error_msg = "Error: 'query' es obligatorio (string)."
        return [{"type": "text", "text": error_msg}]
    if max_rows <= 0:
        max_rows = MCP_MAX_ROWS

    _deny_non_select(query)
    _enforce_allowed_tables(query)
    final_sql = _append_limit(query, max_rows)

    t0 = time.time()
    try:
        with mysql_cursor() as (_cn, cur):
            cur.execute(final_sql)
            cols = [d[0] for d in cur.description] if cur.description else []
            data = cur.fetchall() if cols else []
    except mysql.connector.Error as e:
        error_msg = f"Error MySQL: {e}"
        return [{"type": "text", "text": error_msg}]
    except Exception as e:
        error_msg = f"Error: {e}"
        return [{"type": "text", "text": error_msg}]

    rows = [_mask_row(cols, r) for r in data]
    out = _format_tsv(cols, rows) if cols else "(Sin resultados)"
    meta = f"-- rows={len(rows)} cols={len(cols)} time_s={time.time()-t0:.3f} limit={max_rows}"
    results = [{"type": "text", "text": f"{out}\n\n{meta}"}]
    
    # Enrich with RAG context if enabled
    # Automatically detects entities (brand_id, product_id, seller_id) in results
    enriched_results = _enrich_sql_results_with_context(results)

    return _finalize_tool_output(enriched_results, topic)

    # ==================== BUSINESS TOOLS (brand-aware, seller_id aligned) ====================

# If you don't already have these helpers in your file, keep them. Otherwise, skip this block.
def _mcp_text(text: str) -> list[dict]:
    return [{"type": "text", "text": text}]

def _exec_tsv(sql: str, params: list, headers_override: list[str] | None = None) -> list[dict]:
    """
    Execute SQL and return results as TSV (tab-separated values) in MCP-compatible format.

    **Purpose**
    Standardized helper for executing parameterized SQL safely and producing a text-based
    table response that can be easily parsed or displayed by AI agents.

    **Returns**
      A list with one element:
        [{"type": "text", "text": "<TSV_DATA>"}]

    **TSV Format**
      - First line: column headers (tab-separated)
      - Following lines: one data row per line (tab-separated)
      - No trailing metadata (pure tabular output)

    **Parameters**
      - sql (str): The SQL query to execute (with placeholders if needed)
      - params (list): Positional parameters for the query
      - headers_override (list[str] | None): Optional fixed headers if query columns vary

    **Notes**
      - Errors are caught and returned as text content, not exceptions.
      - Columns and rows are masked/redacted if defined in DENIED_COLUMNS.
      - This function is used by all business tools to produce consistent, agent-friendly outputs.
    """
    try:
        with mysql_cursor() as (_cn, cur):
            cur.execute(sql, params)
            cols = [d[0] for d in cur.description] if cur.description else []
            data = cur.fetchall() if cols else []
    except Exception as e:
        return _mcp_text(f"Error executing SQL: {e}")

    cols_out = headers_override if headers_override else cols
    rows = [_mask_row(cols_out, r) if cols_out else [] for r in data]
    tsv = _format_tsv(cols_out, rows) if cols_out else "(No results)"
    return _mcp_text(tsv)

    # ==================== LOOKUP HELPERS (semantic translation) ====================
# These allow using natural names (e.g., "GoPro", "Germany", "TechWorld") instead
# of numeric IDs in queries for brand_id, country_id, and seller_id.
#
# The matching is case-insensitive and uses partial matching for convenience.
# For example, "gopro", "GoPro", or "GOPRO" will all resolve to the same brand_id.
# ===============================================================================
def resolve_brand_id(name_or_id: str) -> int:
    """Return brand_id given brand name (case-insensitive). If numeric, return as-is."""
    if isinstance(name_or_id, int) or (isinstance(name_or_id, str) and name_or_id.isdigit()):
        return int(name_or_id)
    sql = "SELECT id FROM brand WHERE LOWER(name) = LOWER(%s) LIMIT 1"
    with mysql_cursor() as (_cn, cur):
        cur.execute(sql, [name_or_id.strip()])
        row = cur.fetchone()
        if not row:
            raise ValueError(f"No brand found matching '{name_or_id}'")
        return row[0]


def resolve_country_id(name_or_id: str) -> int:
    """Return country_id given country name (case-insensitive). If numeric, return as-is."""
    if isinstance(name_or_id, int) or (isinstance(name_or_id, str) and name_or_id.isdigit()):
        return int(name_or_id)
    sql = "SELECT id FROM country WHERE LOWER(name) = LOWER(%s) LIMIT 1"
    with mysql_cursor() as (_cn, cur):
        cur.execute(sql, [name_or_id.strip()])
        row = cur.fetchone()
        if not row:
            raise ValueError(f"No country found matching '{name_or_id}'")
        return row[0]


def resolve_seller_id(name_or_id: str) -> int:
    """Return seller_id given seller name (case-insensitive). If numeric, return as-is."""
    if isinstance(name_or_id, int) or (isinstance(name_or_id, str) and name_or_id.isdigit()):
        return int(name_or_id)
    sql = "SELECT id FROM seller WHERE LOWER(name) = LOWER(%s) LIMIT 1"
    with mysql_cursor() as (_cn, cur):
        cur.execute(sql, [name_or_id.strip()])
        row = cur.fetchone()
        if not row:
            raise ValueError(f"No seller found matching '{name_or_id}'")
        return row[0]


# ==================== DISCOVERY TOOLS ====================
# These tools help AI agents discover available entities (brands, countries, sellers)
# and generate common date ranges without needing to know the data beforehand.
# Use these tools FIRST when you don't know the specific IDs or need to explore.
# =========================================================

@app.tool()
def list_brands(name_pattern: str | None = None, max_rows: int = 50) -> list[dict]:
    """
    List available brands with their IDs, optionally filtered by name pattern.
    
    When to use:
      - Discover which brands exist in the system
      - Find the brand_id for a known brand name
      - Explore brands matching a pattern (e.g., all brands starting with "Go")
    
    Parameters:
      - name_pattern (str|None): Optional SQL LIKE pattern (e.g., "GoPro%", "%Tech%")
      - max_rows (int): Limit results (default 50)
    
    Returns (TSV columns):
      - brand_id: Numeric brand identifier (use this in analytics tools)
      - brand_name: Human-readable brand name
    
    Sort order:
      - brand_name ASC (alphabetical)
    
    Example usage:
      - List all: name_pattern=None
      - Find GoPro: name_pattern="GoPro%"
      - Search: name_pattern="%tech%"
    """
    sql = "SELECT id, name FROM brand"
    params = []
    if name_pattern:
        sql += " WHERE name LIKE %s"
        params.append(name_pattern)
    sql += " ORDER BY name LIMIT %s"
    params.append(max_rows)
    return _exec_tsv(sql, params, ["brand_id", "brand_name"])


@app.tool()
def list_countries(name_pattern: str | None = None, max_rows: int = 50) -> list[dict]:
    """
    List available countries/marketplaces with their IDs.
    
    When to use:
      - Discover which countries/marketplaces are in the system
      - Find the country_id for analytics queries
      - Understand geographic coverage
    
    Parameters:
      - name_pattern (str|None): Optional SQL LIKE pattern (e.g., "United%", "%States%")
      - max_rows (int): Limit results (default 50)
    
    Returns (TSV columns):
      - country_id: Numeric country identifier (use this in analytics tools)
      - country_name: Human-readable country/marketplace name
    
    Sort order:
      - country_name ASC (alphabetical)
    
    Notes:
      - Common IDs might be: US, Canada, Mexico, UK, Germany, etc.
      - Use the returned country_id in all analytics tools
    """
    sql = "SELECT id, name FROM country"
    params = []
    if name_pattern:
        sql += " WHERE name LIKE %s"
        params.append(name_pattern)
    sql += " ORDER BY name LIMIT %s"
    params.append(max_rows)
    return _exec_tsv(sql, params, ["country_id", "country_name"])


@app.tool()
def list_sellers(
    name_pattern: str | None = None,
    brand_id: int | None = None,
    country_id: int | None = None,
    only_authorized: bool = False,
    max_rows: int = 100
) -> list[dict]:
    """
    List sellers, optionally filtered by name, brand, country, or authorization status.
    
    When to use:
      - Discover which sellers are active in the system
      - Find seller_id for a known seller name
      - Identify authorized sellers for a brand/country
      - Explore seller population before detailed analysis
    
    Parameters:
      - name_pattern (str|None): Optional SQL LIKE pattern for seller name
      - brand_id (int|None): Filter to sellers authorized for this brand
      - country_id (int|None): Filter to sellers in this country/marketplace
      - only_authorized (bool): If True, show only authorized sellers (requires brand_id and country_id)
      - max_rows (int): Limit results (default 100)
    
    Returns (TSV columns):
      - seller_id: Seller identifier (canonical key)
      - seller_name: Human-readable seller name
      - is_authorized: 1 if authorized for brand/country, 0 otherwise (only when brand_id and country_id provided)
    
    Sort order:
      - seller_name ASC (alphabetical)
    
    Notes:
      - For authorization filtering, both brand_id and country_id must be provided
      - Authorization is derived from seller_brand_authorized table
    """
    if only_authorized and (not brand_id or not country_id):
        return _mcp_text("Error: only_authorized=True requires both brand_id and country_id")
    
    if brand_id and country_id:
        # Include authorization status
        sql = """
        SELECT DISTINCT 
          s.id AS seller_id,
          s.name AS seller_name,
          CASE WHEN sba.seller_id IS NOT NULL THEN 1 ELSE 0 END AS is_authorized
        FROM seller s
        LEFT JOIN seller_brand_authorized sba
          ON sba.seller_id = s.id
         AND sba.brand_id = %s
         AND sba.country_id = %s
        WHERE 1=1
        """
        params = [brand_id, country_id]
        if name_pattern:
            sql += " AND s.name LIKE %s"
            params.append(name_pattern)
        if only_authorized:
            sql += " AND sba.seller_id IS NOT NULL"
        sql += " ORDER BY s.name LIMIT %s"
        params.append(max_rows)
        return _exec_tsv(sql, params, ["seller_id", "seller_name", "is_authorized"])
    else:
        # Simple seller list
        sql = "SELECT id, name FROM seller WHERE 1=1"
        params = []
        if name_pattern:
            sql += " AND name LIKE %s"
            params.append(name_pattern)
        sql += " ORDER BY name LIMIT %s"
        params.append(max_rows)
        return _exec_tsv(sql, params, ["seller_id", "seller_name"])


@app.tool()
def suggest_date_range(period: str = "last_7_days") -> list[dict]:
    """
    Generate common date ranges for analysis queries.
    
    When to use:
      - Need standardized date ranges for analytics
      - Don't know current date or want relative windows
      - Ensure consistent date formats (YYYY-MM-DD)
    
    Parameters:
      - period (str): One of:
        * "last_7_days" - Previous 7 complete days
        * "last_30_days" - Previous 30 complete days
        * "last_90_days" - Previous 90 complete days
        * "this_week" - Current calendar week (Mon-Sun)
        * "last_week" - Previous calendar week
        * "this_month" - Current calendar month
        * "last_month" - Previous calendar month
        * "this_quarter" - Current calendar quarter
        * "this_year" - Current calendar year
        * "yesterday" - Previous complete day
        * "today" - Current day so far
    
    Returns (TSV columns):
      - period_name: Description of the period
      - since: Start date (inclusive) in YYYY-MM-DD format
      - until: End date (exclusive) in YYYY-MM-DD format
      - description: Human-readable explanation
    
    Notes:
      - All ranges use half-open intervals [since, until)
      - Times are in UTC unless configured otherwise
      - Use the returned since/until directly in analytics tools
    
    Example usage:
      1. Call suggest_date_range(period="last_30_days")
      2. Use returned since and until in product_overview or other tools
    """
    from datetime import datetime, timedelta, date
    
    today = date.today()
    now = datetime.now()
    
    ranges = {
        "last_7_days": {
            "since": (today - timedelta(days=7)).strftime("%Y-%m-%d"),
            "until": today.strftime("%Y-%m-%d"),
            "description": "Previous 7 complete days"
        },
        "last_30_days": {
            "since": (today - timedelta(days=30)).strftime("%Y-%m-%d"),
            "until": today.strftime("%Y-%m-%d"),
            "description": "Previous 30 complete days"
        },
        "last_90_days": {
            "since": (today - timedelta(days=90)).strftime("%Y-%m-%d"),
            "until": today.strftime("%Y-%m-%d"),
            "description": "Previous 90 complete days"
        },
        "yesterday": {
            "since": (today - timedelta(days=1)).strftime("%Y-%m-%d"),
            "until": today.strftime("%Y-%m-%d"),
            "description": "Previous complete day"
        },
        "today": {
            "since": today.strftime("%Y-%m-%d"),
            "until": (today + timedelta(days=1)).strftime("%Y-%m-%d"),
            "description": "Current day so far"
        },
        "this_week": {
            "since": (today - timedelta(days=today.weekday())).strftime("%Y-%m-%d"),
            "until": (today + timedelta(days=1)).strftime("%Y-%m-%d"),
            "description": "Current calendar week (Mon-Sun)"
        },
        "last_week": {
            "since": (today - timedelta(days=today.weekday() + 7)).strftime("%Y-%m-%d"),
            "until": (today - timedelta(days=today.weekday())).strftime("%Y-%m-%d"),
            "description": "Previous calendar week"
        },
        "this_month": {
            "since": today.replace(day=1).strftime("%Y-%m-%d"),
            "until": (today + timedelta(days=1)).strftime("%Y-%m-%d"),
            "description": "Current calendar month"
        },
        "last_month": {
            "since": (today.replace(day=1) - timedelta(days=1)).replace(day=1).strftime("%Y-%m-%d"),
            "until": today.replace(day=1).strftime("%Y-%m-%d"),
            "description": "Previous calendar month"
        },
        "this_quarter": {
            "since": date(today.year, ((today.month - 1) // 3) * 3 + 1, 1).strftime("%Y-%m-%d"),
            "until": (today + timedelta(days=1)).strftime("%Y-%m-%d"),
            "description": "Current calendar quarter"
        },
        "this_year": {
            "since": date(today.year, 1, 1).strftime("%Y-%m-%d"),
            "until": (today + timedelta(days=1)).strftime("%Y-%m-%d"),
            "description": "Current calendar year"
        }
    }
    
    if period not in ranges:
        available = ", ".join(sorted(ranges.keys()))
        return _mcp_text(f"Invalid period '{period}'. Available: {available}")
    
    r = ranges[period]
    tsv = f"period_name\tsince\tuntil\tdescription\n{period}\t{r['since']}\t{r['until']}\t{r['description']}"
    return _mcp_text(tsv)


# ==================== BUSINESS TOOLS ====================
# These tools provide brand compliance and seller monitoring analytics.
# All tools are scoped to brand_id + country_id for security.
#
# WORKFLOW GUIDE FOR AI AGENTS:
# Step 1: DISCOVER - Use discovery tools to find IDs
#   - list_brands()     → Find brand_id
#   - list_countries()  → Find country_id
#   - list_sellers()    → Find seller_id or check authorization
#   - suggest_date_range() → Get standardized date ranges
#
# Step 2: ASSESS - High-level health check
#   - product_overview()        → Product health across the brand
#   - brand_buybox_and_compliance()  → Overall brand KPI scorecard and buybox and compliance analysis
#
# Step 3: INVESTIGATE - Specific deep dives
#   - seller_behavior()         → Individual seller violations & behavior
#   - violation_trend()         → Time series of violations
#   - buybox_loss_attribution() → Root cause of Buy Box issues
#
# Step 4: MONITOR - Ongoing surveillance
#   - restock_monitor()              → Track inventory movements
#   - alert_new_suspicious_sellers() → New unauthorized sellers
#   - risk_matrix()                  → Prioritize enforcement actions
#
# KEY CONCEPTS:
# - DATE FORMAT: All since/until params expect 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'
# - WINDOWING: Use half-open ranges [since, until) to avoid overlaps
# - AUTHORIZATION: Always derived via seller_brand_authorized by (seller_id, brand_id, country_id)
# - IDENTIFIERS: asin and model_number are joined from vw_product_identifiers_summary
# =========================================================

# 1) PRODUCT OVERVIEW (adds asin/model_number)
@app.tool()
def product_overview(
    brand_id: int,
    country_id: int,
    since: str,
    until: str,
    max_rows: int = 200,
    topic: str | None = None,
) -> list[dict]:
    """
    Get a comprehensive health overview of products for a brand in a specific country.

    When to use:
      - Initial brand health assessment for a given window
      - Prioritizing ASINs with the most MAP pressure or Buy Box gaps
      - Understanding seller competition per product

    Parameters:
      - brand_id (int): Brand to analyze (filters via product.brand_id)
      - country_id (int): Marketplace/country id
      - since (str): Start of window, inclusive (YYYY-MM-DD or with time)
      - until (str): End of window, exclusive (YYYY-MM-DD or with time)
      - max_rows (int): Row limit (default 200)

    Returns (TSV columns):
      - product_id, asin, model_number: Product identifiers (asin/model_number via vw_product_identifiers_summary)
      - total_sellers: Distinct sellers observed in the window
      - unauthorized_sellers: Distinct sellers not present in seller_brand_authorized for (brand_id, country_id)
      - buybox_unavailable_rate: Fraction of time Buy Box was unavailable (0–1)
      - map_violations: Count of MAP events (from offer_listing map_violation_id)
      - avg_price: Average observed price across the window
      - avg_map_delta: Average (current_map_price - price) for violating rows

    Sort order:
      - map_violations DESC (worst offenders first)

    Notes:
      - Authorization is derived using seller_brand_authorized on (seller_id, brand_id, country_id)
      - Uses CTEs to aggregate first, then joins aggregates (small joins, high performance)
    """
    sql = """
    WITH filtered_offers AS (
      SELECT 
        ol.product_id,
        ol.country_id,
        COUNT(DISTINCT ol.seller_id) AS total_sellers,
        SUM(CASE WHEN sba.seller_id IS NULL THEN 1 ELSE 0 END) AS unauthorized_sellers,
        ROUND(AVG(ol.price), 2) AS avg_price
      FROM offer_listing ol
      INNER JOIN product p
        ON p.id = ol.product_id
       AND p.brand_id = %s
      LEFT JOIN seller_brand_authorized sba
        ON sba.brand_id  = p.brand_id
       AND sba.country_id = ol.country_id
       AND sba.seller_id  = ol.seller_id
      WHERE 
        ol.country_id = %s
        AND ol.entry_date >= %s
        AND ol.entry_date <  %s
      GROUP BY
        ol.product_id,
        ol.country_id
    ),
    cte_map_violations AS (
      SELECT
        ol.product_id,
        ol.country_id,
        ROUND(AVG(ol.current_map_price - ol.price), 2) AS avg_map_delta,
        COUNT(DISTINCT ol.id) AS map_violations
      FROM offer_listing ol
      INNER JOIN product p
        ON p.id = ol.product_id
       AND p.brand_id = %s
      WHERE 
        ol.country_id = %s
        AND ol.entry_date >= %s
        AND ol.entry_date <  %s
        AND ol.map_violation_id IS NOT NULL
      GROUP BY 
        ol.product_id,
        ol.country_id
    ),
    cte_buybox_stats AS (
      SELECT
        bh.product_id,
        bh.country_id,
        ROUND(AVG(bh.buybox_unavailable), 3) AS buybox_unavailable_rate
      FROM buybox_hourly_statistic bh
      INNER JOIN product p
        ON p.id = bh.product_id
       AND p.brand_id = %s
      WHERE 
        bh.country_id = %s
        AND bh.entry_date >= %s
        AND bh.entry_date <  %s
      GROUP BY
        bh.product_id,
        bh.country_id
    )
    SELECT
      fo.product_id,
      vpis.asin,
      vpis.model_number,
      fo.total_sellers,
      fo.unauthorized_sellers,
      COALESCE(bb.buybox_unavailable_rate, 0) AS buybox_unavailable_rate,
      COALESCE(mv.map_violations, 0) AS map_violations,
      fo.avg_price,
      mv.avg_map_delta
    FROM filtered_offers fo
    LEFT JOIN cte_buybox_stats bb
      ON bb.product_id = fo.product_id
     AND bb.country_id = fo.country_id
    LEFT JOIN cte_map_violations mv
      ON mv.product_id = fo.product_id
     AND mv.country_id = fo.country_id
    LEFT JOIN vw_product_identifiers_summary vpis
      ON vpis.brand_id   = %s
     AND vpis.country_id = fo.country_id
     AND vpis.product_id = fo.product_id
    ORDER BY map_violations DESC
    LIMIT %s
    """
    params = [
        brand_id, country_id, since, until,                # filtered_offers
        brand_id, country_id, since, until,                # cte_map_violations
        brand_id, country_id, since, until,                # cte_buybox_stats
        brand_id,                                          # vpis join
        max_rows
    ]
    headers = ["product_id","asin","model_number","total_sellers","unauthorized_sellers",
               "buybox_unavailable_rate","map_violations","avg_price","avg_map_delta"]
    results = _exec_tsv(sql, params, headers)
    # Enrich with brand context from Notion
    enriched = _enrich_results_with_brand_context(
        results, 
        brand_id,
        log_tool_name="product_overview",
        log_inputs={"brand_id": brand_id, "country_id": country_id, "since": since, "until": until, "max_rows": max_rows}
    )
    return _finalize_tool_output(enriched, topic)

# 2) SELLER BEHAVIOR (no product_id in final select)
@app.tool()
def seller_behavior(
    brand_id: int,
    country_id: int,
    since: str,
    until: str,
    seller_id: str | None = None,
    max_rows: int = 200,
    topic: str | None = None,
) -> list[dict]:
    """
    Analyze individual seller behavior patterns for compliance and enforcement focus.

    When to use:
      - Investigating a specific seller’s history (violations, Buy Box presence, restocks)
      - Ranking top violators across all sellers in the brand window
      - Understanding a seller’s pricing pressure (avg_map_delta)

    Parameters:
      - brand_id (int): Brand to analyze (via product.brand_id)
      - country_id (int): Marketplace/country id
      - since (str): Start of window, inclusive
      - until (str): End of window, exclusive
      - seller_id (str|None): Optional — restrict to one seller
      - max_rows (int): Row limit (default 200)

    Returns (TSV columns):
      - seller_name, seller_id: Seller identity (id is the canonical key)
      - violations: Count of MAP violations (offer_listing.map_violation_id IS NOT NULL)
      - avg_map_delta: Average $ below MAP when violating
      - buybox_appearances: Times the seller held the Buy Box in the window
      - restocks: Number of restock events observed (seller_restocks)

    Sort order:
      - violations DESC, restocks DESC, buybox_appearances DESC

    Notes:
      - Authorization is NOT shown here; use risk_matrix or alert_new_suspicious_sellers if you need it.
      - Aggregations done in CTEs; no large joins.
    """
    sql = """
    WITH auth_sellers AS (
      SELECT DISTINCT sba.seller_id
      FROM seller_brand_authorized sba
      WHERE sba.brand_id = %s AND sba.country_id = %s
    ),
    filtered_offers AS (
      SELECT
        ol.seller_id,
        COALESCE(ol.seller_name, '(unknown)') AS seller_name,
        SUM(CASE WHEN ol.is_buybox_owner = 1 THEN 1 ELSE 0 END) AS buybox_appearances,
        SUM(CASE WHEN ol.map_violation_id IS NOT NULL THEN 1 ELSE 0 END) AS violations,
        ROUND(AVG(CASE WHEN ol.map_violation_id IS NOT NULL THEN (ol.current_map_price - ol.price) END), 2) AS avg_map_delta
      FROM offer_listing ol
      INNER JOIN product p
        ON p.id = ol.product_id AND p.brand_id = %s
      WHERE ol.country_id = %s AND ol.entry_date >= %s AND ol.entry_date < %s
      /**/  {seller_filter_offers}
      GROUP BY ol.seller_id, COALESCE(ol.seller_name, '(unknown)')
    ),
    restocks AS (
      SELECT sr.seller_id, COUNT(*) AS restocks
      FROM seller_restocks sr
      INNER JOIN product p ON p.id = sr.product_id AND p.brand_id = %s
      WHERE sr.country_id = %s AND sr.restock_date >= %s AND sr.restock_date < %s
      /**/  {seller_filter_restocks}
      GROUP BY sr.seller_id
    )
    SELECT
      fo.seller_name,
      fo.seller_id,
      fo.violations,
      fo.avg_map_delta,
      fo.buybox_appearances,
      COALESCE(r.restocks, 0) AS restocks
    FROM filtered_offers fo
    LEFT JOIN restocks r ON r.seller_id = fo.seller_id
    ORDER BY fo.violations DESC, r.restocks DESC, fo.buybox_appearances DESC
    LIMIT %s
    """
    seller_filter_offers = "AND ol.seller_id = %s" if seller_id else ""
    seller_filter_restocks = "AND sr.seller_id = %s" if seller_id else ""
    sql = sql.format(seller_filter_offers=seller_filter_offers, seller_filter_restocks=seller_filter_restocks)

    params = [
        brand_id, country_id,           # auth_sellers
        brand_id, country_id, since, until,   # filtered_offers (brand,country,dates)
    ]
    if seller_id: params.append(seller_id)
    params += [
        brand_id, country_id, since, until,   # restocks
    ]
    if seller_id: params.append(seller_id)
    params += [max_rows]

    headers = ["seller_name","seller_id","violations","avg_map_delta","buybox_appearances","restocks"]
    results = _exec_tsv(sql, params, headers)
    # Enrich with brand context from Notion
    enriched = _enrich_results_with_brand_context(
        results,
        brand_id,
        log_tool_name="seller_behavior",
        log_inputs={
            "brand_id": brand_id,
            "country_id": country_id,
            "since": since,
            "until": until,
            "seller_id": seller_id,
            "max_rows": max_rows,
        },
    )
    return _finalize_tool_output(enriched, topic)

# 3) VIOLATION TREND (no product_id in final select)
@app.tool()
def violation_trend(
    brand_id: int,
    country_id: int,
    since: str,
    until: str,
    grain: str = "day",  # "day" | "hour"
    topic: str | None = None,
) -> list[dict]:
    """
    Time series analysis of MAP violations over time for a brand in one country.

    When to use:
      - Detecting violation spikes or patterns
      - Monitoring whether enforcement is improving compliance
      - Correlating violations with external events or campaigns

    Parameters:
      - brand_id (int): Brand to analyze (via product.brand_id)
      - country_id (int): Marketplace/country id
      - since (str): Start of window, inclusive
      - until (str): End of window, exclusive
      - grain (str): Aggregation level — "day" for daily trends, "hour" for intraday patterns

    Returns (TSV columns; ordered chronologically):
      - bucket: Time period (YYYY-MM-DD or YYYY-MM-DD HH:00:00)
      - total_violations: Count of violations in the period
      - avg_map_delta: Average price gap vs MAP for violating rows in the period

    Notes:
      - Source is offer_listing (uses map_violation_id), no heavy join to map_violation table required.
      - Uses half-open windowing to avoid overlap across adjacent runs.
    """
    if grain not in ("day","hour"):
        return _mcp_text("Invalid grain. Use 'day' or 'hour'.")

    bucket_expr = "DATE(ol.entry_date)" if grain == "day" else "DATE_FORMAT(ol.entry_date, '%Y-%m-%d %H:00:00')"

    sql = f"""
    WITH filtered_violations AS (
      SELECT
        {bucket_expr} AS bucket,
        COUNT(DISTINCT ol.id) AS total_violations,
        ROUND(AVG(ol.current_map_price - ol.price), 2) AS avg_map_delta
      FROM offer_listing ol
      INNER JOIN product p ON p.id = ol.product_id AND p.brand_id = %s
      WHERE ol.country_id = %s AND ol.entry_date >= %s AND ol.entry_date < %s
        AND ol.map_violation_id IS NOT NULL
      GROUP BY {bucket_expr}
    )
    SELECT bucket, total_violations, avg_map_delta
    FROM filtered_violations
    ORDER BY bucket ASC
    """
    params = [brand_id, country_id, since, until]
    headers = ["bucket","total_violations","avg_map_delta"]
    results = _exec_tsv(sql, params, headers)
    # Enrich with brand context from Notion
    enriched = _enrich_results_with_brand_context(
        results,
        brand_id,
        log_tool_name="violation_trend",
        log_inputs={
            "brand_id": brand_id,
            "country_id": country_id,
            "since": since,
            "until": until,
            "grain": grain,
        },
    )
    return _finalize_tool_output(enriched, topic)

# 4) RESTOCK MONITOR (adds asin/model_number)
@app.tool()
def restock_monitor(
    brand_id: int,
    country_id: int,
    since: str,
    until: str,
    seller_id: str | None = None,
    product_id: int | None = None,
    only_unauthorized: bool = False,
    max_rows: int = 200,
    topic: str | None = None,
) -> list[dict]:
    """
    List recent restock events for a brand, enriched with product identifiers (asin, model_number).

    When to use:
      - Track inventory movements and restock cadence
      - Investigate whether unauthorized sellers keep replenishing
      - Tie restocks to subsequent Buy Box or pricing behavior (in separate tools)

    Parameters:
      - brand_id (int): Brand to analyze (via product.brand_id)
      - country_id (int): Marketplace/country id
      - since (str): Start of window, inclusive (uses seller_restocks.restock_date)
      - until (str): End of window, exclusive
      - seller_id (str|None): Optional — filter to a seller
      - product_id (int|None): Optional — filter to a product
      - only_unauthorized (bool): If True, return only sellers not in seller_brand_authorized
      - max_rows (int): Row limit (default 200)

    Returns (TSV columns):
      - restock_date, seller_id, product_id, asin, model_number, country_id
      - is_authorized: 1 if seller is in seller_brand_authorized for (brand_id, country_id), else 0
      - fullfiled_by, restock_amount, restock_price, map_price, stock_before, stock_after, priority

    Sort order:
      - restock_date DESC (most recent first)

    Example usage:
      - Monitor unauthorized restocks last week: brand_id=5, country_id=12, since='2025-11-01', until='2025-11-08', only_unauthorized=True
    """
    sql = """
    WITH auth_sellers AS (
      SELECT DISTINCT sba.seller_id
      FROM seller_brand_authorized sba
      WHERE sba.brand_id = %s AND sba.country_id = %s
    ),
    filtered_restocks AS (
      SELECT
        sr.restock_date,
        sr.seller_id,
        sr.product_id,
        sr.country_id,
        CASE WHEN a.seller_id IS NOT NULL THEN 1 ELSE 0 END AS is_authorized,
        sr.fullfiled_by,
        sr.restock_amount,
        sr.restock_price,
        sr.map_price,
        sr.stock_before,
        sr.stock_after,
        sr.priority
      FROM seller_restocks sr
      INNER JOIN product p ON p.id = sr.product_id AND p.brand_id = %s
      LEFT JOIN auth_sellers a ON a.seller_id = sr.seller_id
      WHERE sr.country_id = %s
        AND sr.restock_date >= %s AND sr.restock_date < %s
        /**/ {seller_filter}
        /**/ {product_filter}
        /**/ {unauth_filter}
    )
    SELECT
      fr.restock_date,
      fr.seller_id,
      fr.product_id,
      vpis.asin,
      vpis.model_number,
      fr.country_id,
      fr.is_authorized,
      fr.fullfiled_by,
      fr.restock_amount,
      fr.restock_price,
      fr.map_price,
      fr.stock_before,
      fr.stock_after,
      fr.priority
    FROM filtered_restocks fr
    LEFT JOIN vw_product_identifiers_summary vpis
      ON vpis.brand_id   = %s
     AND vpis.country_id = fr.country_id
     AND vpis.product_id = fr.product_id
    ORDER BY fr.restock_date DESC
    LIMIT %s
    """
    seller_filter  = "AND sr.seller_id  = %s" if seller_id  else ""
    product_filter = "AND sr.product_id = %s" if product_id else ""
    unauth_filter  = "AND a.seller_id IS NULL" if only_unauthorized else ""
    sql = sql.format(seller_filter=seller_filter, product_filter=product_filter, unauth_filter=unauth_filter)

    params = [brand_id, country_id, brand_id, country_id, since, until]
    if seller_id:  params.append(seller_id)
    if product_id: params.append(product_id)
    params += [brand_id, max_rows]

    headers = ["restock_date","seller_id","product_id","asin","model_number","country_id","is_authorized",
               "fullfiled_by","restock_amount","restock_price","map_price","stock_before","stock_after","priority"]
    results = _exec_tsv(sql, params, headers)
    # Enrich with brand context from Notion
    enriched = _enrich_results_with_brand_context(
        results,
        brand_id,
        log_tool_name="restock_monitor",
        log_inputs={
            "brand_id": brand_id,
            "country_id": country_id,
            "since": since,
            "until": until,
            "seller_id": seller_id,
            "product_id": product_id,
            "only_unauthorized": only_unauthorized,
            "max_rows": max_rows,
        },
    )
    return _finalize_tool_output(enriched, topic)

# 5) RISK MATRIX (no product_id in final select)
@app.tool()
def risk_matrix(
    brand_id: int,
    country_id: int,
    since: str = "",
    until: str = "",
    max_rows: int = 20,
    auth_status: str = "all",  # "all" | "authorized" | "unauthorized"
    topic: str | None = None,
) -> list[dict]:
    """
    Prioritize risky sellers combining violations, Buy Box behavior, restock persistence, and current inventory.

    When to use:
      - Weekly/daily enforcement prioritization.
      - Identifying "dormant" threats (sellers with high stock but no recent restocks).
      - Building a short-list for legal or ops follow-up based on inventory volume.

    Interpretation Guide for Results:
      - risk_score: The primary sort key. Higher = higher priority for enforcement.
      - total_current_stock: Total units currently held in FBA inventory.
      - total_restocked_units: Measures the volume of recent supply leaks.
      - restocks_unauth: replenishment by sellers NOT in the authorized list.

    Parameters:
      - brand_id (int): Brand to analyze.
      - country_id (int): Marketplace/country id.
      - since (str|None): Start of window (YYYY-MM-DD). Defaults to 7 days ago.
      - until (str|None): End of window (YYYY-MM-DD). Defaults to today.
      - max_rows (int): Row limit (default 20).
      - auth_status (str): Filter by "all", "authorized", or "unauthorized" (default "all").

    Returns (TSV columns):
      - seller_name: Seller identity.
      - seller_id: Seller identifier.
      - is_authorized: 1 if authorized, 0 otherwise.
      - violations: MAP violations count.
      - avg_map_delta: Average $ below MAP when violating.
      - total_current_stock: Total units currently in FBA inventory.
      - restock_events: Count of distinct FBA replenishment arrivals in window.
      - total_restocked_units: Total units moved into FBA during the window.
      - total_restocked_value: Est. dollar value of restocked inventory.
      - buybox_appearances: BB wins/appearances in the window.
      - unauth_buybox_hits: BB wins by unauthorized sellers.
      - risk_score: Composite prioritization score

    Risk Score Formula:
      - violations × 2.0
      - restocks_unauth × 1.5
      - unauth_buybox_hits × 0.75
      - total_restocked_units × 0.25
      - total_current_stock × 0.1
    """
    from datetime import date, timedelta
    if not until:
        until = date.today().strftime("%Y-%m-%d")
    if not since:
        try:
            until_dt = date.fromisoformat(until)
        except ValueError:
            until_dt = date.fromisoformat(until.split()[0])
        since = (until_dt - timedelta(days=7)).strftime("%Y-%m-%d")

    sql = """
    WITH auth_sellers AS (
      SELECT DISTINCT sba.seller_id
      FROM seller_brand_authorized sba
      WHERE sba.brand_id = %s AND sba.country_id = %s
    ),
    cte_offers_seller AS (
      SELECT
        ol.seller_id,
        COALESCE(ol.seller_name, '(unknown)') AS seller_name,
        SUM(CASE WHEN ol.is_buybox_owner = 1 THEN 1 ELSE 0 END) AS buybox_appearances,
        SUM(CASE WHEN ol.is_buybox_owner = 1 AND a.seller_id IS NULL THEN 1 ELSE 0 END) AS unauth_buybox_hits,
        SUM(CASE WHEN ol.map_violation_id IS NOT NULL THEN 1 ELSE 0 END) AS violations,
        ROUND(AVG(CASE WHEN ol.map_violation_id IS NOT NULL THEN (ol.current_map_price - ol.price) END), 2) AS avg_map_delta
      FROM last_offer_listing ol
      INNER JOIN product p ON p.id = ol.product_id AND p.brand_id = %s
      LEFT JOIN auth_sellers a ON a.seller_id = ol.seller_id
      WHERE ol.country_id = %s AND ol.entry_date >= %s AND ol.entry_date < %s
      GROUP BY ol.seller_id, COALESCE(ol.seller_name, '(unknown)')
    ),
    cte_restocks_seller AS (
      SELECT sr.seller_id,
             COUNT(*) AS restock_events,
             SUM(CASE WHEN a.seller_id IS NULL THEN 1 ELSE 0 END) AS restocks_unauth,
             SUM(sr.restock_amount) as total_restocked_units,
             SUM(sr.restock_amount * sr.restock_price) as total_restocked_value
      FROM seller_restocks sr
      INNER JOIN product p ON p.id = sr.product_id AND p.brand_id = %s
      LEFT JOIN auth_sellers a ON a.seller_id = sr.seller_id
      WHERE sr.country_id = %s AND sr.restock_date >= %s AND sr.restock_date < %s AND sr.fullfiled_by = 'fba'
      GROUP BY sr.seller_id
    ),
    cte_inventory_seller AS (
      SELECT lsi.seller_id,
             SUM(lsi.stock) AS total_current_stock
      FROM last_seller_inventory lsi
      INNER JOIN product p ON p.id = lsi.product_id AND p.brand_id = %s
      WHERE lsi.country_id = %s
      GROUP BY lsi.seller_id
    ),
    cte_authorized_flag AS (
      SELECT o.seller_id,
             CASE WHEN a.seller_id IS NOT NULL THEN 1 ELSE 0 END AS is_authorized_for_brand_country
      FROM (SELECT DISTINCT seller_id FROM cte_offers_seller) o
      LEFT JOIN auth_sellers a ON a.seller_id = o.seller_id
    ),
    final_agg AS (
      SELECT
        o.seller_name,
        o.seller_id,
        f.is_authorized_for_brand_country AS is_authorized,
        o.violations,
        o.avg_map_delta,
        COALESCE(i.total_current_stock, 0) AS total_current_stock,
        COALESCE(r.restock_events, 0)  AS restock_events,
        COALESCE(r.total_restocked_units, 0) AS total_restocked_units,
        COALESCE(r.total_restocked_value, 0) AS total_restocked_value,
        o.buybox_appearances,
        o.unauth_buybox_hits,
        (o.violations * 2.0
         + COALESCE(r.restocks_unauth, 0) * 1.5
         + o.unauth_buybox_hits * 0.75
         + COALESCE(r.total_restocked_units, 0) * 0.25
         + COALESCE(i.total_current_stock, 0) * 0.1) AS risk_score
      FROM cte_offers_seller o
      LEFT JOIN cte_restocks_seller r ON r.seller_id = o.seller_id
      LEFT JOIN cte_inventory_seller i ON i.seller_id = o.seller_id
      LEFT JOIN cte_authorized_flag f ON f.seller_id = o.seller_id
    )
    SELECT * FROM final_agg
    WHERE 1=1
    /**/ {auth_filter}
    ORDER BY risk_score DESC
    LIMIT %s
    """
    auth_filter = ""
    if auth_status == "authorized":
        auth_filter = "AND is_authorized = 1"
    elif auth_status == "unauthorized":
        auth_filter = "AND is_authorized = 0"
    
    sql = sql.format(auth_filter=auth_filter)

    params = [
        brand_id, country_id,           # auth_sellers
        brand_id, country_id, since, until, # cte_offers_seller
        brand_id, country_id, since, until, # cte_restocks_seller
        brand_id, country_id,           # cte_inventory_seller
        max_rows                        # limit
    ]
    headers = [
        "seller_name", "seller_id", "is_authorized", "violations", "avg_map_delta", 
        "total_current_stock", "restock_events", "total_restocked_units", 
        "total_restocked_value", "buybox_appearances", "unauth_buybox_hits", "risk_score"
    ]
    results = _exec_tsv(sql, params, headers)
    # Enrich with brand context from Notion
    enriched = _enrich_results_with_brand_context(
        results,
        brand_id,
        log_tool_name="risk_matrix",
        log_inputs={
            "brand_id": brand_id,
            "country_id": country_id,
            "since": since,
            "until": until,
            "max_rows": max_rows,
            "auth_status": auth_status
        },
    )
    return _finalize_tool_output(enriched, topic)

# 6) BUYBOX LOSS ATTRIBUTION (adds asin/model_number)
@app.tool()
def buybox_loss_attribution(
    brand_id: int,
    country_id: int,
    since: str,
    until: str,
    product_id: int | None = None,
    max_rows: int = 400,
    topic: str | None = None,
) -> list[dict]:
    """
    Attribute Buy Box loss to root causes per product: unavailability, unauthorized BB wins, and MAP pressure.
    Enrich with asin and model_number for downstream reporting.

    When to use:
      - Distinguish between structural availability issues vs. competitive unauthorized wins
      - Explain Buy Box underperformance to brand stakeholders
      - Feed remediation playbooks (pricing enforcement vs. catalog/ops fixes)

    Parameters:
      - brand_id (int): Brand to analyze
      - country_id (int): Marketplace/country id
      - since (str): Start of window, inclusive
      - until (str): End of window, exclusive
      - product_id (int|None): Optional — restrict to a single product
      - max_rows (int): Row limit (default 400)

    Returns (TSV columns):
      - product_id, asin, model_number
      - buybox_unavailable_rate: Fraction of hours with BB unavailable (0–1)
      - unauth_buybox_rate: % of BB events won by unauthorized sellers (0–1)
      - map_violations: Violation count from offer_listing
      - avg_map_delta: Average $ below MAP when violating

    Sort order:
      - map_violations DESC, unauth_buybox_rate DESC

    Notes:
      - Uses three independent CTEs (buybox stats, BB ownership split, violations), then joins aggregates.
      - Authorization derived via seller_brand_authorized (seller_id, brand_id, country_id).
    """
    sql = """
    WITH auth_sellers AS (
      SELECT DISTINCT sba.seller_id
      FROM seller_brand_authorized sba
      WHERE sba.brand_id = %s AND sba.country_id = %s
    ),
    cte_buybox_stats AS (
      SELECT
        bh.product_id,
        bh.country_id,
        ROUND(AVG(bh.buybox_unavailable), 3) AS buybox_unavailable_rate
      FROM buybox_hourly_statistic bh
      INNER JOIN product p ON p.id = bh.product_id AND p.brand_id = %s
      WHERE bh.country_id = %s AND bh.entry_date >= %s AND bh.entry_date < %s
        /**/ {prod_filter_bh}
      GROUP BY bh.product_id, bh.country_id
    ),
    cte_offer_buybox AS (
      SELECT
        ol.product_id,
        ol.country_id,
        SUM(CASE WHEN ol.is_buybox_owner = 1 THEN 1 ELSE 0 END) AS total_buybox_events,
        SUM(CASE WHEN ol.is_buybox_owner = 1 AND a.seller_id IS NULL THEN 1 ELSE 0 END) AS unauth_buybox_events
      FROM offer_listing ol
      INNER JOIN product p ON p.id = ol.product_id AND p.brand_id = %s
      LEFT JOIN auth_sellers a ON a.seller_id = ol.seller_id
      WHERE ol.country_id = %s AND ol.entry_date >= %s AND ol.entry_date < %s
        /**/ {prod_filter_ol}
      GROUP BY ol.product_id, ol.country_id
    ),
    cte_map_violations AS (
      SELECT
        ol.product_id,
        ol.country_id,
        COUNT(*) AS map_violations,
        ROUND(AVG(ol.current_map_price - ol.price), 2) AS avg_map_delta
      FROM offer_listing ol
      INNER JOIN product p ON p.id = ol.product_id AND p.brand_id = %s
      WHERE ol.country_id = %s AND ol.entry_date >= %s AND ol.entry_date < %s
        AND ol.map_violation_id IS NOT NULL
        /**/ {prod_filter_mv}
      GROUP BY ol.product_id, ol.country_id
    )
    SELECT
      bs.product_id,
      vpis.asin,
      vpis.model_number,
      bs.buybox_unavailable_rate,
      ROUND(COALESCE(ob.unauth_buybox_events, 0) / NULLIF(ob.total_buybox_events, 0), 3) AS unauth_buybox_rate,
      COALESCE(mv.map_violations, 0) AS map_violations,
      mv.avg_map_delta
    FROM cte_buybox_stats bs
    LEFT JOIN cte_offer_buybox ob
      ON ob.product_id = bs.product_id AND ob.country_id = bs.country_id
    LEFT JOIN cte_map_violations mv
      ON mv.product_id = bs.product_id AND mv.country_id = bs.country_id
    LEFT JOIN vw_product_identifiers_summary vpis
      ON vpis.brand_id   = %s
     AND vpis.country_id = bs.country_id
     AND vpis.product_id = bs.product_id
    ORDER BY map_violations DESC, unauth_buybox_rate DESC
    LIMIT %s
    """
    prod_filter = "AND {alias}.product_id = %s"
    sql = sql.format(
        prod_filter_bh = (prod_filter.format(alias="bh") if product_id else ""),
        prod_filter_ol = (prod_filter.format(alias="ol") if product_id else ""),
        prod_filter_mv = (prod_filter.format(alias="ol") if product_id else "")
    )
    params = [brand_id, country_id, brand_id, country_id, since, until]  # auth + bh
    if product_id: params.append(product_id)
    params += [brand_id, country_id, since, until]  # ol
    if product_id: params.append(product_id)
    params += [brand_id, country_id, since, until]  # mv
    if product_id: params.append(product_id)
    params += [brand_id, max_rows]                  # vpis + limit

    headers = ["product_id","asin","model_number","buybox_unavailable_rate","unauth_buybox_rate","map_violations","avg_map_delta"]
    results = _exec_tsv(sql, params, headers)
    # Enrich with brand context from Notion
    enriched = _enrich_results_with_brand_context(
        results,
        brand_id,
        log_tool_name="buybox_loss_attribution",
        log_inputs={
            "brand_id": brand_id,
            "country_id": country_id,
            "since": since,
            "until": until,
            "product_id": product_id,
            "max_rows": max_rows,
        },
    )
    return _finalize_tool_output(enriched, topic)

# 7) BRAND BUYBOX AND COMPLIANCE (no product_id in final select)

@app.tool()
def brand_buybox_and_compliance(
    brand_id: int,
    country_id: int,
    since: str,
    until: str,
    topic: str | None = None,
) -> list[dict]:
    """
    Brand-level Buy Box distribution (Unavailable, Amazon, Authorized, Unauthorized) and product compliance.

    When to use:
      - Align MCP results with dashboard cards (the 4-way Buy Box shares that sum to 100%).
      - Produce a brand summary for a given time window and marketplace.
      - Feed “executive summary” reports with availability, BB ownership mix, and compliance.

    Parameters:
      - brand_id (int): Brand to analyze (filters via product.brand_id)
      - country_id (int): Country/marketplace id
      - since (str): Start of window, inclusive (YYYY-MM-DD or with time)
      - until (str): End of window, exclusive (YYYY-MM-DD or with time)

    Returns (TSV columns):
      - brand_name: Human-readable brand name
      - availability_pct: (1 - Unavailable share) × 100
      - authorized_pct: % of hours Buy Box held by authorized sellers
      - unauthorized_pct: % of hours Buy Box held by unauthorized sellers
      - amazon_pct: % of hours Buy Box held by Amazon (inferred via name/merchant id patterns)
      - total_products: Distinct products for the brand
      - compliant_products: Products with zero MAP violations in the window
      - compliance_pct: compliant_products / total_products × 100
      - compliance_index: Weighted composite = Availability 20% + Authorized 40% + Compliance 40%

    Notes:
      - This uses hourly data from buybox_hourly_statistic and its owner table.
      - Amazon ownership is inferred when seller name contains 'amazon' or merchant id looks like 'amzn'/'A1…'.
      - 4 buckets are mutually exclusive; their shares sum to 100%.
      - Product compliance is product-based (share of products with zero violations in window).
    """
    sql = """
    WITH auth_sellers AS (
      SELECT DISTINCT sba.seller_id
      FROM seller_brand_authorized sba
      WHERE sba.brand_id = %s
        AND sba.country_id = %s
    ),

    -- 1) Hourly Buy Box classification (mutually exclusive buckets)
    hourly_bb AS (
      SELECT
        bh.product_id,
        bh.country_id,
        bh.entry_date,
        CASE
          WHEN bh.buybox_unavailable = 1 THEN 'unavailable'
          WHEN LOWER(COALESCE(bho.buybox_owner_seller_name,'')) LIKE '%%amazon%%'
            OR LOWER(COALESCE(bho.buybox_owner_merchant_id,'')) LIKE '%%amzn%%'
            OR LOWER(COALESCE(bho.buybox_owner_merchant_id,'')) LIKE 'a1%%'
          THEN 'amazon'
          WHEN bho.is_buybox_owner_authorized_seller = 1 THEN 'authorized'
          ELSE 'unauthorized'
        END AS bb_bucket
      FROM buybox_hourly_statistic bh
      INNER JOIN product p
        ON p.id = bh.product_id
       AND p.brand_id = %s
      LEFT JOIN buybox_hourly_statistic_owner bho
        ON bho.buybox_hourly_statistic_id = bh.id
      WHERE
        bh.country_id = %s
        AND bh.entry_date >= %s
        AND bh.entry_date <  %s
    ),

    bb_share AS (
      -- Four-way Buy Box share (sums to 1.0)
      SELECT
        ROUND(AVG(CASE WHEN bb_bucket = 'unavailable' THEN 1 ELSE 0 END), 6) AS pct_unavailable,
        ROUND(AVG(CASE WHEN bb_bucket = 'amazon'      THEN 1 ELSE 0 END), 6) AS pct_amazon,
        ROUND(AVG(CASE WHEN bb_bucket = 'authorized'  THEN 1 ELSE 0 END), 6) AS pct_authorized,
        ROUND(AVG(CASE WHEN bb_bucket = 'unauthorized'THEN 1 ELSE 0 END), 6) AS pct_unauthorized
      FROM hourly_bb
    ),

    -- 2) Product-based compliance (% products with zero violations in window)
    violating_products AS (
      SELECT DISTINCT ol.product_id
      FROM offer_listing ol
      INNER JOIN product p
        ON p.id = ol.product_id
       AND p.brand_id = %s
      WHERE
        ol.country_id = %s
        AND ol.entry_date >= %s
        AND ol.entry_date <  %s
        AND ol.map_violation_id IS NOT NULL
    ),
    brand_products AS (
      SELECT DISTINCT p.id AS product_id
      FROM product p
      WHERE p.brand_id = %s
    ),
    compliance AS (
      SELECT
        (SELECT COUNT(*) FROM brand_products) AS total_products,
        (SELECT COUNT(*) FROM brand_products bp
          WHERE bp.product_id NOT IN (SELECT product_id FROM violating_products)
        ) AS compliant_products
    )

    SELECT
      b.name AS brand_name,
      ROUND((1 - s.pct_unavailable) * 100, 2) AS availability_pct,
      ROUND(s.pct_authorized * 100, 2)       AS authorized_pct,
      ROUND(s.pct_unauthorized * 100, 2)     AS unauthorized_pct,
      ROUND(s.pct_amazon * 100, 2)           AS amazon_pct,
      c.total_products,
      c.compliant_products,
      ROUND(c.compliant_products / NULLIF(c.total_products, 0) * 100, 2) AS compliance_pct,
      ROUND((
        (1 - s.pct_unavailable) * 0.2 +
        s.pct_authorized * 0.4 +
        (c.compliant_products / NULLIF(c.total_products, 0)) * 0.4
      ) * 100, 2) AS compliance_index
    FROM bb_share s
    CROSS JOIN compliance c
    JOIN brand b ON b.id = %s
    """
    params = [
        # auth_sellers
        brand_id, country_id,
        # hourly_bb
        brand_id, country_id, since, until,
        # violating_products
        brand_id, country_id, since, until,
        # brand_products
        brand_id,
        # final brand join
        brand_id,
    ]
    headers = [
        "brand_name",
        "availability_pct",
        "authorized_pct",
        "unauthorized_pct",
        "amazon_pct",
        "total_products",
        "compliant_products",
        "compliance_pct",
        "compliance_index",
    ]
    results = _exec_tsv(sql, params, headers)
    # Enrich with brand context from Notion
    enriched = _enrich_results_with_brand_context(
        results,
        brand_id,
        log_tool_name="brand_buybox_and_compliance",
        log_inputs={
            "brand_id": brand_id,
            "country_id": country_id,
            "since": since,
            "until": until,
        },
    )
    return _finalize_tool_output(enriched, topic)

# 8) ALERT NEW SUSPICIOUS SELLERS (no product_id in final select)
@app.tool()
def alert_new_suspicious_sellers(
    brand_id: int,
    country_id: int,
    since: str,
    until: str,
    max_rows: int = 100,
    topic: str | None = None,
) -> list[dict]:
    """
    Identify newly seen unauthorized sellers and rank them by suspicious behavior.

    When to use:
      - Daily/weekly monitoring to catch newcomers quickly
      - Prioritize outreach/enforcement against sellers grabbing BB or violating MAP
      - Build a watchlist for ops/legal

    Parameters:
      - brand_id (int): Brand to analyze
      - country_id (int): Marketplace/country id
      - since (str): Start of window, inclusive
      - until (str): End of window, exclusive
      - max_rows (int): Row limit (default 100)

    Returns (TSV columns):
      - seller_name, seller_id
      - first_seen: First observed offer in window
      - distinct_products_listed: Product breadth
      - buybox_appearances: BB wins/appearances in the window
      - violations: MAP violation count (offer_listing-derived)
      - min_price, max_price, avg_price: Pricing context
      - is_unauthorized: 1 if not in seller_brand_authorized for brand/country
      - restocks_last_10d: Recent restock activity count
      - suspicion_score: Composite score (your weights)

    Suspicion Score Formula (your latest):
      (is_unauthorized * 2)
    + (violations * 0.5)
    + (buybox_appearances * 1.5)
    + (restocks_last_10d * 0.5)

    Sort order:
      - suspicion_score DESC, first_seen DESC

    Example usage:
      - brand_id=5, country_id=12, since='2025-11-01', until='2025-11-10'
    """
    sql = """
    WITH auth_sellers AS (
      SELECT DISTINCT sba.seller_id
      FROM seller_brand_authorized sba
      WHERE sba.brand_id = %s AND sba.country_id = %s
    ),
    cte_recent_offers AS (
      SELECT
        ol.seller_id,
        COALESCE(ol.seller_name, '(unknown)') AS seller_name,
        MIN(ol.entry_date) AS first_seen,
        COUNT(DISTINCT ol.product_id) AS distinct_products_listed,
        SUM(CASE WHEN ol.is_buybox_owner = 1 THEN 1 ELSE 0 END) AS buybox_appearances,
        ROUND(AVG(ol.price), 2) AS avg_price,
        ROUND(MIN(ol.price), 2) AS min_price,
        ROUND(MAX(ol.price), 2) AS max_price,
        SUM(CASE WHEN ol.map_violation_id IS NOT NULL THEN 1 ELSE 0 END) AS violations
      FROM offer_listing ol
      INNER JOIN product p ON p.id = ol.product_id AND p.brand_id = %s
      WHERE ol.country_id = %s AND ol.entry_date >= %s AND ol.entry_date < %s
      GROUP BY ol.seller_id, COALESCE(ol.seller_name, '(unknown)')
    ),
    cte_flagged_sellers AS (
      SELECT
        ro.seller_id,
        ro.seller_name,
        ro.first_seen,
        ro.distinct_products_listed,
        ro.buybox_appearances,
        ro.avg_price,
        ro.min_price,
        ro.max_price,
        ro.violations,
        CASE WHEN a.seller_id IS NULL THEN 1 ELSE 0 END AS is_unauthorized
      FROM cte_recent_offers ro
      LEFT JOIN auth_sellers a ON a.seller_id = ro.seller_id
    ),
    cte_restock_activity AS (
      SELECT sr.seller_id, COUNT(*) AS restocks_last_10d
      FROM seller_restocks sr
      INNER JOIN product p ON p.id = sr.product_id AND p.brand_id = %s
      WHERE sr.country_id = %s AND sr.restock_date >= %s AND sr.restock_date < %s
      GROUP BY sr.seller_id
    )
    SELECT
      f.seller_name,
      f.seller_id,
      f.first_seen,
      f.distinct_products_listed,
      f.buybox_appearances,
      f.violations,
      f.min_price,
      f.max_price,
      f.avg_price,
      f.is_unauthorized,
      COALESCE(r.restocks_last_10d, 0) AS restocks_last_10d,
      (f.is_unauthorized * 2
       + f.violations * 0.5
       + f.buybox_appearances * 1.5
       + COALESCE(r.restocks_last_10d, 0) * 0.5) AS suspicion_score
    FROM cte_flagged_sellers f
    LEFT JOIN cte_restock_activity r ON r.seller_id = f.seller_id
    WHERE f.is_unauthorized = 1
    ORDER BY suspicion_score DESC, f.first_seen DESC
    LIMIT %s
    """
    params = [brand_id, country_id, brand_id, country_id, since, until, brand_id, country_id, since, until, max_rows]
    headers = ["seller_name","seller_id","first_seen","distinct_products_listed","buybox_appearances","violations",
               "min_price","max_price","avg_price","is_unauthorized","restocks_last_10d","suspicion_score"]
    results = _exec_tsv(sql, params, headers)
    # Enrich with brand context from Notion
    enriched = _enrich_results_with_brand_context(
        results,
        brand_id,
        log_tool_name="alert_new_suspicious_sellers",
        log_inputs={
            "brand_id": brand_id,
            "country_id": country_id,
            "since": since,
            "until": until,
            "max_rows": max_rows,
        },
    )
    return _finalize_tool_output(enriched, topic)

# 9) BUYBOX OPPORTUNITY SUMMARY
@app.tool()
def buybox_opportunity_summary(
    brand_id: int,
    country_id: int,
    topic: str | None = None,
) -> list[dict]:
    """
    High-level KPI summary of Buy Box opportunities (External vs Internal issues).

    When to use:
      - Quick health check of Buy Box status for a brand.
      - Quantifying the scale of price suppressions (external) vs price gaps (internal).

    Parameters:
      - brand_id (int): Brand identifier.
      - country_id (int): Marketplace identifier.

    Returns (TSV columns):
      - external_issues_count: Count of products with Buy Box suppressed due to external market prices.
      - internal_issues_count: Count of products where brand/authorized sellers are uncompetitive.
      - out_of_stock_count: Products with no active inventory.
      - support_ticket_count: Products flagged as needing manual intervention with Amazon.
    """
    sql = """
    SELECT 
      external_issues_count,
      internal_issues_count,
      out_of_stock_count,
      support_ticket_count
    FROM vw_buybox_opportunities_summary
    WHERE brand_id = %s AND country_id = %s
    """
    params = [brand_id, country_id]
    headers = ["external_issues_count", "internal_issues_count", "out_of_stock_count", "support_ticket_count"]
    results = _exec_tsv(sql, params, headers)
    enriched = _enrich_results_with_brand_context(results, brand_id, log_tool_name="buybox_opportunity_summary")
    return _finalize_tool_output(enriched, topic)


# 10) BUYBOX OPPORTUNITY LIST
@app.tool()
def buybox_opportunity_list(
    brand_id: int,
    country_id: int,
    issue_type: str = "all",  # "all" | "external" | "internal"
    max_rows: int = 20,
    topic: str | None = None,
) -> list[dict]:
    """
    List prioritized Buy Box opportunities for a brand.

    When to use:
      - Identifying the most valuable products to fix (based on stock value).
      - Finding products where external price matching is most active.

    Parameters:
      - brand_id (int): Brand identifier.
      - country_id (int): Marketplace identifier.
      - issue_type (str): "all", "external" (suppressed), or "internal" (uncompetitive).
      - max_rows (int): Row limit (default 20).

    Returns (TSV columns):
      - asin, product_description
      - issue_type: 'External' or 'Internal'
      - featured_offer_price: Current winning price on Amazon.
      - your_price: The brand's active price.
      - map_price: Minimum Advertised Price.
      - total_stock_value: Total value of inventory held by affected sellers.
      - active_exact_matches: Number of external listings matching target price.
      - priority: System-assigned priority level.
    """
    issue_filter = ""
    if issue_type == "external":
        issue_filter = "AND o.has_external_issue = 1"
    elif issue_type == "internal":
        issue_filter = "AND o.has_internal_issue = 1"

    sql = f"""
    SELECT 
      o.asin,
      LEFT(o.product_description, 50) AS product_description,
      CASE 
        WHEN o.has_external_issue = 1 THEN 'External'
        WHEN o.has_internal_issue = 1 THEN 'Internal'
        ELSE 'None'
      END AS issue_type,
      o.featured_offer_price,
      o.your_price,
      o.map_price,
      COALESCE(s.total_stock_value, 0) AS total_stock_value,
      o.active_exact_matches_count AS active_exact_matches,
      o.priority
    FROM vw_buybox_opportunities o
    LEFT JOIN (
      SELECT product_id, country_id, SUM(stock_value) as total_stock_value
      FROM vw_buybox_opportunity_amazon_sellers
      GROUP BY product_id, country_id
    ) s ON s.product_id = o.product_id AND s.country_id = o.country_id
    WHERE o.brand_id = %s AND o.country_id = %s
    {issue_filter}
    ORDER BY total_stock_value DESC, active_exact_matches DESC
    LIMIT %s
    """
    params = [brand_id, country_id, max_rows]
    headers = ["asin", "product_description", "issue_type", "featured_offer_price", "your_price", "map_price", "total_stock_value", "active_exact_matches", "priority"]
    results = _exec_tsv(sql, params, headers)
    enriched = _enrich_results_with_brand_context(results, brand_id, log_tool_name="buybox_opportunity_list")
    return _finalize_tool_output(enriched, topic)


# 11) BUYBOX EXTERNAL DRILLDOWN
@app.tool()
def buybox_external_drilldown(
    brand_id: int,
    country_id: int,
    asin: str,
    topic: str | None = None,
) -> list[dict]:
    """
    Find the external sources (Walmart, eBay, etc.) causing a Buy Box suppression.

    When to use:
      - Investigating "External Issues" found in the opportunity list.
      - Identifying specific sellers undercutting Amazon in the wider market.

    Parameters:
      - brand_id (int): Brand identifier.
      - country_id (int): Marketplace identifier.
      - asin (str): Product ASIN to investigate.

    Returns (TSV columns):
      - market: The external marketplace (e.g., Walmart).
      - seller_name: The seller on that marketplace.
      - price: The price offered externally.
      - seller_status: Authorized or Unauthorized.
      - exact_match_since: When this price was first seen matching the competitive target.
    """
    sql = """
    SELECT 
      market,
      seller_name,
      price,
      seller_status,
      exact_match_entry_date AS exact_match_since
    FROM vw_buybox_opportunity_iw_listings
    WHERE brand_id = %s AND country_id = %s AND asin = %s
    ORDER BY price ASC
    """
    params = [brand_id, country_id, asin]
    headers = ["market", "seller_name", "price", "seller_status", "exact_match_since"]
    results = _exec_tsv(sql, params, headers)
    enriched = _enrich_results_with_brand_context(results, brand_id, log_tool_name="buybox_external_drilldown")
    return _finalize_tool_output(enriched, topic)


# 12) BUYBOX INTERNAL DRILLDOWN
@app.tool()
def buybox_internal_drilldown(
    brand_id: int,
    country_id: int,
    asin: str,
    topic: str | None = None,
) -> list[dict]:
    """
    Analyze which authorized sellers are losing the Buy Box and why.

    When to use:
      - Investigating "Internal Issues" where Amazon sellers are uncompetitive.
      - Checking stock levels and pricing for authorized sellers.

    Parameters:
      - brand_id (int): Brand identifier.
      - country_id (int): Marketplace identifier.
      - asin (str): Product ASIN to investigate.

    Returns (TSV columns):
      - seller_name: Amazon seller identity.
      - seller_status: Authorized or Unauthorized.
      - landed_price: Total price (product + shipping).
      - stock_units: Current inventory levels.
      - is_buybox_owner: 1 if currently winning, 0 otherwise.
      - is_eligible: 1 if seller is eligible for BB.
    """
    sql = """
    SELECT 
      seller_name,
      seller_status,
      landed_price,
      stock_units,
      is_buybox_owner,
      is_eligible_for_buybox AS is_eligible
    FROM vw_buybox_opportunity_amazon_sellers
    WHERE brand_id = %s AND country_id = %s AND asin = %s
    ORDER BY is_buybox_owner DESC, landed_price ASC
    """
    params = [brand_id, country_id, asin]
    headers = ["seller_name", "seller_status", "landed_price", "stock_units", "is_buybox_owner", "is_eligible"]
    results = _exec_tsv(sql, params, headers)
    enriched = _enrich_results_with_brand_context(results, brand_id, log_tool_name="buybox_internal_drilldown")
    return _finalize_tool_output(enriched, topic)


# --------------------------------------

if __name__ == "__main__":
    app.run()
