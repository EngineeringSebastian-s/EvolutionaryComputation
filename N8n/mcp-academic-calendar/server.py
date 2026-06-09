"""
server.py
─────────────────────────────────────────────────────────────────────────────
Servidor MCP — Horarios Académicos · Politécnico Colombiano Jaime Isaza Cadavid
Transporte: SSE (Server-Sent Events) sobre Starlette / ASGI
─────────────────────────────────────────────────────────────────────────────
Endpoints expuestos:
  GET  /sse       → canal SSE que el cliente MCP (n8n) abre primero
  POST /messages  → canal de mensajes bidireccional del protocolo MCP

Herramienta disponible:
  consultar_calendario_academico  → scraping en tiempo real del sitio oficial
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx
from bs4 import BeautifulSoup
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import TextContent, Tool
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.routing import Mount, Route

# ─────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────
CALENDARIO_URL = "https://www.politecnicojic.edu.co/calendario-academico"

# Regex para identificar divs de periodo (ej. "2026-1", "2025-2-medellin")
PERIODO_RE = re.compile(r"^\d{4}-\d")

# Timeout generoso para el scraping (red universitaria puede ser lenta)
HTTP_TIMEOUT = httpx.Timeout(timeout=20.0)

# Headers que simulan un navegador real para evitar bloqueos por User-Agent
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "es-CO,es;q=0.9",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("mcp-horarios")


# ─────────────────────────────────────────────
# Lógica de scraping
# ─────────────────────────────────────────────

async def _fetch_html(url: str) -> str:
    """Descarga el HTML de la URL dada de forma asíncrona."""
    async with httpx.AsyncClient(
            headers=REQUEST_HEADERS,
            timeout=HTTP_TIMEOUT,
            follow_redirects=True,
    ) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.text


def _parse_calendario(html: str) -> list[dict[str, str]]:
    """
    Parsea el HTML del calendario académico y retorna una lista de eventos.

    Estructura esperada en el HTML:
      <div class="YYYY-N[-sufijo]">          ← contenedor de periodo
        <h2 class="toggle-titulo">…</h2>     ← nombre del periodo
        <div class="toggle-item">            ← bloque de categoría
          <div class="toggle-enlace"><h4>…</h4></div>          ← categoría
          <div class="toggle-seccion">
            <h4 class="toggle-seccion__titulo">…</h4>          ← evento
            <p  class="toggle-seccion__contenido">…</p>        ← fecha
          </div>
        </div>
      </div>
    """
    soup = BeautifulSoup(html, "html.parser")
    events: list[dict[str, str]] = []

    # El contenido útil vive dentro de [itemprop="articleBody"]
    article_body = soup.find("div", itemprop="articleBody")
    if not article_body:
        # Fallback: buscar directamente en todo el documento
        log.warning("No se encontró div[itemprop='articleBody']; buscando en documento completo.")
        article_body = soup

    for periodo_div in article_body.find_all("div", recursive=False):
        classes = periodo_div.get("class", [])

        # Filtrar únicamente divs cuya clase identifique un periodo académico
        if not any(PERIODO_RE.match(c) for c in classes):
            continue

        # Nombre del periodo
        h2 = periodo_div.find("h2", class_="toggle-titulo")
        periodo_nombre = h2.get_text(strip=True) if h2 else "Periodo sin nombre"

        # Iterar sobre cada bloque de categoría
        for item in periodo_div.find_all("div", class_="toggle-item"):

            # Nombre de la categoría
            enlace_div = item.find("div", class_="toggle-enlace")
            if enlace_div:
                cat_h4 = enlace_div.find("h4")
                categoria = cat_h4.get_text(strip=True) if cat_h4 else "General"
            else:
                categoria = "General"

            # Sección de eventos (puede haber más de una por item)
            for seccion in item.find_all("div", class_="toggle-seccion"):
                titulos = seccion.find_all("h4", class_="toggle-seccion__titulo")
                fechas = seccion.find_all("p", class_="toggle-seccion__contenido")

                # Emparejar titulo[i] con fecha[i]
                for titulo, fecha in zip(titulos, fechas):
                    events.append({
                        "periodo": periodo_nombre,
                        "categoria": categoria,
                        "evento": titulo.get_text(strip=True),
                        "fecha": fecha.get_text(strip=True),
                    })

    return events


# ─────────────────────────────────────────────
# Servidor MCP
# ─────────────────────────────────────────────

mcp_server = Server("horarios-academicos-poli")


@mcp_server.list_tools()
async def list_tools() -> list[Tool]:
    """Declara las herramientas disponibles en este servidor MCP."""
    return [
        Tool(
            name="consultar_calendario_academico",
            description=(
                "Consulta en tiempo real el calendario académico oficial del "
                "Politécnico Colombiano Jaime Isaza Cadavid. "
                "Devuelve una lista JSON con todos los eventos académicos estructurados: "
                "periodo, categoría, nombre del evento y fecha. "
                "Úsala cuando el usuario pregunte por fechas de clases, vacaciones, "
                "exámenes, matrículas, semana santa u otras actividades del calendario."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "periodo": {
                        "type": "string",
                        "description": (
                            "Opcional. Filtra los resultados por periodo académico "
                            "(ej. '2026-1', '2025-2'). Si se omite, devuelve todos los periodos."
                        ),
                    }
                },
                "required": [],
            },
        )
    ]


@mcp_server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Despacha la llamada a la herramienta solicitada."""

    if name != "consultar_calendario_academico":
        return [TextContent(
            type="text",
            text=json.dumps(
                {"error": f"Herramienta desconocida: '{name}'"},
                ensure_ascii=False,
            ),
        )]

    filtro_periodo: str | None = arguments.get("periodo")

    try:
        log.info("Descargando calendario académico desde %s", CALENDARIO_URL)
        html = await _fetch_html(CALENDARIO_URL)

        log.info("Parseando estructura de eventos…")
        events = _parse_calendario(html)

        if not events:
            return [TextContent(
                type="text",
                text=json.dumps(
                    {"advertencia": "No se encontraron eventos. La estructura del sitio pudo haber cambiado."},
                    ensure_ascii=False,
                ),
            )]

        # Aplicar filtro opcional de periodo
        if filtro_periodo:
            pattern = filtro_periodo.strip().lower()
            events = [
                e for e in events
                if pattern in e["periodo"].lower()
            ]
            log.info("Filtro '%s' aplicado: %d evento(s) encontrados.", filtro_periodo, len(events))
        else:
            log.info("Sin filtro: %d evento(s) encontrados.", len(events))

        # Serializar como JSON minificado (sin saltos de línea)
        payload = json.dumps(events, ensure_ascii=False, separators=(",", ":"))
        return [TextContent(type="text", text=payload)]

    except httpx.TimeoutException:
        msg = {
            "error": "Tiempo de espera agotado al intentar conectar con el sitio del Politécnico.",
            "sugerencia": "Intenta nuevamente en unos minutos.",
        }
        log.error("Timeout al descargar %s", CALENDARIO_URL)
        return [TextContent(type="text", text=json.dumps(msg, ensure_ascii=False))]

    except httpx.HTTPStatusError as exc:
        msg = {
            "error": f"El sitio respondió con HTTP {exc.response.status_code}.",
            "url": CALENDARIO_URL,
        }
        log.error("HTTP %s desde %s", exc.response.status_code, CALENDARIO_URL)
        return [TextContent(type="text", text=json.dumps(msg, ensure_ascii=False))]

    except Exception as exc:  # noqa: BLE001
        msg = {"error": f"Error inesperado durante el scraping: {type(exc).__name__}: {exc}"}
        log.exception("Error inesperado en consultar_calendario_academico")
        return [TextContent(type="text", text=json.dumps(msg, ensure_ascii=False))]


# ─────────────────────────────────────────────
# Aplicación ASGI (Starlette) con transporte SSE
# ─────────────────────────────────────────────

# SseServerTransport gestiona el estado de las sesiones SSE.
# El path "/messages" debe coincidir con el Mount en las rutas de Starlette.
sse_transport = SseServerTransport("/messages")


async def sse_endpoint(request: Request):
    """
    GET /sse
    El cliente MCP abre este endpoint primero.
    Starlette delega en el transporte SSE del SDK, que devuelve la respuesta SSE
    y conecta los streams de lectura/escritura al mcp_server.
    """
    async with sse_transport.connect_sse(
            request.scope,
            request.receive,
            request._send,  # noqa: SLF001
    ) as (read_stream, write_stream):
        await mcp_server.run(
            read_stream,
            write_stream,
            mcp_server.create_initialization_options(),
        )


# Starlette requiere la ruta /sse como Route normal
# y /messages montado como ASGI app del transport
app = Starlette(
    routes=[
        Route("/sse", endpoint=sse_endpoint),
        Mount("/messages", app=sse_transport.handle_post_message),
    ]
)

# ─────────────────────────────────────────────
# Punto de entrada directo (desarrollo)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8080,
        log_level="info",
    )
