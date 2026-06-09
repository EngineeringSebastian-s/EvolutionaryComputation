"""
server_bienestar.py
─────────────────────────────────────────────────────────────────────────────
Servidor MCP — Bienestar Estudiantil · Politécnico Colombiano Jaime Isaza Cadavid
Transporte : SSE (Server-Sent Events) sobre Starlette / ASGI
─────────────────────────────────────────────────────────────────────────────
Endpoints expuestos:
  GET  /sse       → canal SSE que el cliente MCP (n8n) abre al iniciar sesión
  POST /messages  → mensajes bidireccionales del protocolo MCP

Herramienta:
  consultar_bienestar_estudiantil  → scraping en tiempo real del sitio oficial
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
from bs4 import BeautifulSoup, NavigableString, Tag
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import TextContent, Tool
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.routing import Mount, Route

# ─────────────────────────────────────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────────────────────────────────────

BIENESTAR_URL = (
    "https://www.politecnicojic.edu.co/presentacion-bienestar-institucional"
)

HTTP_TIMEOUT = httpx.Timeout(timeout=20.0)

# User-Agent de navegador real para esquivar bloqueos por bot-detection
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "es-CO,es;q=0.9",
}

# Etiquetas que abren una nueva sección temática
HEADING_TAGS: frozenset[str] = frozenset({"h3", "h4"})

# Etiquetas que aportan contenido a la sección activa
CONTENT_TAGS: frozenset[str] = frozenset({"p", "ul", "ol"})

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("mcp-bienestar")


# ─────────────────────────────────────────────────────────────────────────────
# Scraping
# ─────────────────────────────────────────────────────────────────────────────

async def _fetch_html(url: str) -> str:
    """Descarga el HTML de la URL con httpx de forma completamente asíncrona."""
    async with httpx.AsyncClient(
        headers=REQUEST_HEADERS,
        timeout=HTTP_TIMEOUT,
        follow_redirects=True,
    ) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.text


def _parse_bienestar(html: str) -> list[dict[str, str]]:
    """
    Parsea el HTML de la página de bienestar y devuelve secciones estructuradas.

    Algoritmo de iteración (sibling-walk):
    ──────────────────────────────────────
    Todos los elementos relevantes son hijos directos de div[itemprop='articleBody'].
    Se recorren en orden lineal:

      ┌─ Elemento es <h3> o <h4>  ──► guarda la sección anterior (si existe)
      │                                y abre una nueva con ese texto como título.
      │
      └─ Elemento es <p>, <ul> o <ol>
           └─ hay sección activa  ──► extrae texto y lo agrega al "contenido".
                                      Para <ul>/<ol> une los <li> con "; ".

    Al finalizar el bucle se guarda la última sección (flush).
    Las secciones sin contenido textual se descartan.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Raíz del contenido útil
    body: Tag | None = soup.find("div", itemprop="articleBody")
    if not body:
        log.warning(
            "No se encontró div[itemprop='articleBody']; "
            "buscando en todo el documento."
        )
        body = soup  # type: ignore[assignment]

    sections: list[dict[str, str]] = []
    current: dict[str, str] | None = None  # sección activa

    for node in body.children:
        # Ignorar nodos de texto puro (saltos de línea, espacios)
        if isinstance(node, NavigableString):
            continue
        if not isinstance(node, Tag):
            continue

        tag = node.name  # str, ej. "h3", "p", "ul"

        # ── Nuevo encabezado → cierra sección anterior y abre una nueva ──────
        if tag in HEADING_TAGS:
            titulo = node.get_text(strip=True)
            if not titulo:
                continue  # h3/h4 vacíos (poco frecuente, pero defensivo)

            # Guardar sección anterior solo si tiene contenido real
            if current and current["contenido"].strip():
                sections.append(current)

            current = {"seccion": titulo, "contenido": ""}

        # ── Contenido de párrafo ──────────────────────────────────────────────
        elif tag == "p" and current is not None:
            texto = node.get_text(strip=True)
            if texto:
                # Concatenar con espacio si ya hay contenido previo
                sep = " " if current["contenido"] else ""
                current["contenido"] += sep + texto

        # ── Lista (ul / ol) → unir ítems con "; " ─────────────────────────────
        elif tag in {"ul", "ol"} and current is not None:
            items = [
                li.get_text(strip=True)
                for li in node.find_all("li")
                if li.get_text(strip=True)
            ]
            if items:
                # Prefijo descriptivo solo si la lista abre el contenido
                prefix = (
                    "Direcciones involucradas: "
                    if not current["contenido"]
                    else " Lista: "
                )
                current["contenido"] += prefix + "; ".join(items)

    # ── Flush: guardar la última sección activa ───────────────────────────────
    if current and current["contenido"].strip():
        sections.append(current)

    return sections


# ─────────────────────────────────────────────────────────────────────────────
# Servidor MCP
# ─────────────────────────────────────────────────────────────────────────────

mcp_server = Server("bienestar-estudiantil-poli")


@mcp_server.list_tools()
async def list_tools() -> list[Tool]:
    """Registra las herramientas disponibles en este servidor MCP."""
    return [
        Tool(
            name="consultar_bienestar_estudiantil",
            description=(
                "Consulta en tiempo real la oferta de Bienestar Estudiantil del "
                "Politécnico Colombiano Jaime Isaza Cadavid: programas de inclusión, "
                "cultura, deportes, permanencia, graduación y atención a población "
                "vulnerable. Devuelve un array JSON con secciones temáticas y su "
                "contenido. Úsala cuando el usuario pregunte por becas, apoyos "
                "económicos, actividades culturales o deportivas, servicios de salud "
                "o psicológicos, o cualquier programa de bienestar universitario."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "filtro": {
                        "type": "string",
                        "description": (
                            "Opcional. Palabra clave para filtrar secciones "
                            "(ej. 'cultura', 'deporte', 'vulnerable'). "
                            "Si se omite, devuelve todas las secciones."
                        ),
                    }
                },
                "required": [],
            },
        )
    ]


@mcp_server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Despacha la herramienta solicitada por el agente LLM."""

    if name != "consultar_bienestar_estudiantil":
        payload = json.dumps(
            {"error": f"Herramienta desconocida: '{name}'"},
            ensure_ascii=False,
        )
        return [TextContent(type="text", text=payload)]

    filtro: str | None = arguments.get("filtro")

    # ── Descarga ──────────────────────────────────────────────────────────────
    try:
        log.info("Descargando página de bienestar desde %s", BIENESTAR_URL)
        html = await _fetch_html(BIENESTAR_URL)

    except httpx.TimeoutException:
        msg = {
            "error": "Tiempo de espera agotado al conectar con el sitio del Politécnico.",
            "sugerencia": "Intenta nuevamente en unos minutos.",
        }
        log.error("Timeout al descargar %s", BIENESTAR_URL)
        return [TextContent(type="text", text=json.dumps(msg, ensure_ascii=False))]

    except httpx.HTTPStatusError as exc:
        msg = {
            "error": f"El sitio respondió con HTTP {exc.response.status_code}.",
            "url": BIENESTAR_URL,
        }
        log.error("HTTP %s desde %s", exc.response.status_code, BIENESTAR_URL)
        return [TextContent(type="text", text=json.dumps(msg, ensure_ascii=False))]

    except Exception as exc:  # noqa: BLE001
        msg = {"error": f"Error de red inesperado: {type(exc).__name__}: {exc}"}
        log.exception("Error de red en consultar_bienestar_estudiantil")
        return [TextContent(type="text", text=json.dumps(msg, ensure_ascii=False))]

    # ── Parseo ────────────────────────────────────────────────────────────────
    try:
        log.info("Parseando estructura de secciones de bienestar…")
        sections = _parse_bienestar(html)

        if not sections:
            return [TextContent(
                type="text",
                text=json.dumps(
                    {
                        "advertencia": (
                            "No se encontraron secciones de bienestar. "
                            "La estructura del sitio pudo haber cambiado."
                        )
                    },
                    ensure_ascii=False,
                ),
            )]

        # Filtro opcional por palabra clave (case-insensitive)
        if filtro:
            keyword = filtro.strip().lower()
            sections = [
                s for s in sections
                if keyword in s["seccion"].lower()
                or keyword in s["contenido"].lower()
            ]
            log.info(
                "Filtro '%s' aplicado: %d sección(es) encontradas.",
                filtro, len(sections),
            )
        else:
            log.info("Sin filtro: %d sección(es) encontradas.", len(sections))

        # JSON minificado (sin saltos de línea) para el LLM
        payload = json.dumps(sections, ensure_ascii=False, separators=(",", ":"))
        return [TextContent(type="text", text=payload)]

    except Exception as exc:  # noqa: BLE001
        msg = {"error": f"Error durante el parseo del HTML: {type(exc).__name__}: {exc}"}
        log.exception("Error de parseo en consultar_bienestar_estudiantil")
        return [TextContent(type="text", text=json.dumps(msg, ensure_ascii=False))]


# ─────────────────────────────────────────────────────────────────────────────
# Aplicación ASGI — transporte SSE
# ─────────────────────────────────────────────────────────────────────────────

# SseServerTransport gestiona el estado de sesiones SSE en memoria.
# El path "/messages" debe coincidir exactamente con el Mount de Starlette.
sse_transport = SseServerTransport("/messages")


async def sse_endpoint(request: Request):
    """
    GET /sse
    El cliente MCP (n8n) abre este endpoint al iniciar la sesión.
    El SDK devuelve una respuesta SSE y conecta los streams al mcp_server.
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


app = Starlette(
    routes=[
        Route("/sse", endpoint=sse_endpoint),
        Mount("/messages", app=sse_transport.handle_post_message),
    ]
)


# ─────────────────────────────────────────────────────────────────────────────
# Punto de entrada (modo desarrollo)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server_bienestar:app",
        host="0.0.0.0",
        port=8081,          # puerto distinto al servidor de horarios (8080)
        log_level="info",
    )
