"""
server_biblioteca.py
─────────────────────────────────────────────────────────────────────────────
Servidor MCP — Biblioteca Digital · Politécnico Colombiano Jaime Isaza Cadavid
Transporte : SSE (Server-Sent Events) sobre Starlette / ASGI
─────────────────────────────────────────────────────────────────────────────
Endpoints expuestos:
  GET  /sse       → canal SSE que el cliente MCP (n8n) abre al iniciar sesión
  POST /messages  → mensajes bidireccionales del protocolo MCP

Herramienta:
  consultar_servicios_biblioteca  → scraping en tiempo real del sitio oficial
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

BIBLIOTECA_URL = "https://www.politecnicojic.edu.co/servicos-biblioteca"
BASE_DOMAIN = "https://www.politecnicojic.edu.co"

HTTP_TIMEOUT = httpx.Timeout(timeout=20.0)

# User-Agent real para evitar bloqueos por bot-detection
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
log = logging.getLogger("mcp-biblioteca")


# ─────────────────────────────────────────────────────────────────────────────
# Scraping
# ─────────────────────────────────────────────────────────────────────────────

async def _fetch_html(url: str) -> str:
    """Descarga el HTML con httpx de forma asíncrona."""
    async with httpx.AsyncClient(
            headers=REQUEST_HEADERS,
            timeout=HTTP_TIMEOUT,
            follow_redirects=True,
    ) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.text


def _resolve_url(href: str) -> str:
    """
    Convierte una ruta relativa en URL absoluta usando BASE_DOMAIN.
    Si la URL ya es absoluta (empieza con http/https), la devuelve sin cambios.
    """
    if href.startswith(("http://", "https://")):
        return href
    # Ruta relativa: prepender el dominio base
    return BASE_DOMAIN + (href if href.startswith("/") else f"/{href}")


def _parse_servicios(html: str) -> list[dict[str, str]]:
    """
    Parsea el HTML de la página de servicios de biblioteca.

    Algoritmo sibling-walk — cuatro casos tratados en orden:
    ─────────────────────────────────────────────────────────
    Se itera sobre los hijos directos de div[itemprop='articleBody'].
    El estado activo es `current` (servicio en construcción).

    CASO 1 — <h3> SIN clase 'alert'
        Es el título de un nuevo servicio.
        Guarda el `current` anterior (si tiene contenido) y abre uno nuevo.

    CASO 2 — <h3 class="alert ..."> CON <a> adentro
        Es un enlace a un recurso (PDF, formato, etc.) relacionado con el
        servicio activo anterior. Se extrae el texto del enlace y su `href`,
        y se resuelve a URL absoluta si es relativa. Se concatena al
        `descripcion` del servicio actual como "Enlace: Texto (URL)".
        → Captura el PDF de Formación y el Formato FD-GB13 mencionados.

    CASO 3 — <p>
        Texto descriptivo del servicio activo. Se concatena con espacio.
        Los <b> o <strong> internos se aplanan con get_text().

    CASO 4 — <ol> o <ul>
        Lista de ítems (ej. los +50 convenios interbibliotecarios).
        Se extraen todos los <li> y se unen con ", " formando un string
        continuo. Prefijo " Lista: " si ya hay descripción, "Lista: " si
        es el primer contenido (raro pero defensivo).

    Nodos ignorados: NavigableString (saltos de línea, espacios entre tags).
    Al finalizar el bucle se hace flush del último servicio activo.
    Servicios sin descripción textual se descartan.
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

    services: list[dict[str, str]] = []
    current: dict[str, str] | None = None  # servicio activo

    for node in body.children:
        # Ignorar texto puro entre tags (whitespace, newlines)
        if isinstance(node, NavigableString):
            continue
        if not isinstance(node, Tag):
            continue

        tag: str = node.name
        classes: list[str] = node.get("class", [])

        # ── CASO 1: h3 regular → nuevo servicio ───────────────────────────
        if tag == "h3" and "alert" not in classes:
            titulo = node.get_text(strip=True)
            if not titulo:
                continue  # h3 vacío, ignorar

            # Guardar servicio anterior si tiene contenido real
            if current and current["descripcion"].strip():
                services.append(current)

            current = {"servicio": titulo, "descripcion": ""}

        # ── CASO 2: h3.alert con <a> → enlace al servicio actual ──────────
        elif tag == "h3" and "alert" in classes and current is not None:
            anchor = node.find("a")
            if anchor:
                link_text = anchor.get_text(strip=True)
                href = anchor.get("href", "").strip()
                if href:
                    full_url = _resolve_url(href)
                    fragment = f"Enlace: {link_text} ({full_url})"
                    sep = " " if current["descripcion"] else ""
                    current["descripcion"] += sep + fragment

        # ── CASO 3: párrafo → texto descriptivo ───────────────────────────
        elif tag == "p" and current is not None:
            # get_text aplana <b>, <strong>, <i>, etc.
            text = node.get_text(strip=True)
            if text:
                sep = " " if current["descripcion"] else ""
                current["descripcion"] += sep + text

        # ── CASO 4: lista → ítems unidos con ", " ─────────────────────────
        elif tag in {"ul", "ol"} and current is not None:
            items = [
                li.get_text(strip=True)
                for li in node.find_all("li")
                if li.get_text(strip=True)
            ]
            if items:
                # Separa la lista del texto previo para lectura clara
                prefix = " Lista: " if current["descripcion"] else "Lista: "
                current["descripcion"] += prefix + ", ".join(items)

    # ── Flush: guardar el último servicio activo ───────────────────────────
    if current and current["descripcion"].strip():
        services.append(current)

    return services


# ─────────────────────────────────────────────────────────────────────────────
# Servidor MCP
# ─────────────────────────────────────────────────────────────────────────────

mcp_server = Server("biblioteca-digital-poli")


@mcp_server.list_tools()
async def list_tools() -> list[Tool]:
    """Registra las herramientas disponibles en este servidor MCP."""
    return [
        Tool(
            name="consultar_servicios_biblioteca",
            description=(
                "Consulta en tiempo real los servicios, normativas y convenios "
                "del Sistema de Biblioteca del Politécnico Colombiano Jaime Isaza Cadavid. "
                "Devuelve un array JSON con cada servicio y su descripción completa: "
                "préstamo en sala, préstamo domiciliario, préstamo interbibliotecario, "
                "convenios con más de 50 instituciones, formación y capacitación, "
                "renovaciones y bases de datos disponibles. "
                "Úsala cuando el usuario pregunte cómo funciona el préstamo de libros, "
                "qué instituciones tienen convenio, cómo renovar un libro, "
                "cuántos libros puede llevar, cuánto tiempo dura el préstamo "
                "o qué recursos electrónicos ofrece la biblioteca."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "filtro": {
                        "type": "string",
                        "description": (
                            "Opcional. Palabra clave para filtrar servicios "
                            "(ej. 'convenio', 'préstamo', 'formación', 'renovación'). "
                            "Si se omite, devuelve todos los servicios."
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

    if name != "consultar_servicios_biblioteca":
        payload = json.dumps(
            {"error": f"Herramienta desconocida: '{name}'"},
            ensure_ascii=False,
        )
        return [TextContent(type="text", text=payload)]

    filtro: str | None = arguments.get("filtro")

    # ── Descarga ──────────────────────────────────────────────────────────────
    try:
        log.info("Descargando servicios de biblioteca desde %s", BIBLIOTECA_URL)
        html = await _fetch_html(BIBLIOTECA_URL)

    except httpx.TimeoutException:
        msg = {
            "error": "Tiempo de espera agotado al conectar con el sitio del Politécnico.",
            "sugerencia": "Intenta nuevamente en unos minutos.",
        }
        log.error("Timeout al descargar %s", BIBLIOTECA_URL)
        return [TextContent(type="text", text=json.dumps(msg, ensure_ascii=False))]

    except httpx.HTTPStatusError as exc:
        msg = {
            "error": f"El sitio respondió con HTTP {exc.response.status_code}.",
            "url": BIBLIOTECA_URL,
        }
        log.error("HTTP %s desde %s", exc.response.status_code, BIBLIOTECA_URL)
        return [TextContent(type="text", text=json.dumps(msg, ensure_ascii=False))]

    except Exception as exc:  # noqa: BLE001
        msg = {"error": f"Error de red inesperado: {type(exc).__name__}: {exc}"}
        log.exception("Error de red en consultar_servicios_biblioteca")
        return [TextContent(type="text", text=json.dumps(msg, ensure_ascii=False))]

    # ── Parseo ────────────────────────────────────────────────────────────────
    try:
        log.info("Parseando estructura de servicios de biblioteca…")
        services = _parse_servicios(html)

        if not services:
            return [TextContent(
                type="text",
                text=json.dumps(
                    {
                        "advertencia": (
                            "No se encontraron servicios. "
                            "La estructura del sitio pudo haber cambiado."
                        )
                    },
                    ensure_ascii=False,
                ),
            )]

        # Filtro opcional por palabra clave (case-insensitive, busca en ambos campos)
        if filtro:
            keyword = filtro.strip().lower()
            services = [
                s for s in services
                if keyword in s["servicio"].lower()
                   or keyword in s["descripcion"].lower()
            ]
            log.info(
                "Filtro '%s' aplicado: %d servicio(s) encontrados.",
                filtro, len(services),
            )
        else:
            log.info("Sin filtro: %d servicio(s) encontrados.", len(services))

        # JSON minificado (separadores sin espacios) para el LLM
        payload = json.dumps(services, ensure_ascii=False, separators=(",", ":"))
        return [TextContent(type="text", text=payload)]

    except Exception as exc:  # noqa: BLE001
        msg = {"error": f"Error durante el parseo del HTML: {type(exc).__name__}: {exc}"}
        log.exception("Error de parseo en consultar_servicios_biblioteca")
        return [TextContent(type="text", text=json.dumps(msg, ensure_ascii=False))]


# ─────────────────────────────────────────────────────────────────────────────
# Aplicación ASGI — transporte SSE
# ─────────────────────────────────────────────────────────────────────────────

# SseServerTransport mantiene el estado de sesiones SSE en memoria del proceso.
# El path "/messages" debe coincidir exactamente con el Mount de Starlette.
sse_transport = SseServerTransport("/messages")


async def sse_endpoint(request: Request):
    """
    GET /sse
    El cliente MCP (n8n) abre este endpoint al iniciar la sesión.
    El SDK gestiona el handshake y conecta los streams al mcp_server.
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
        "server_biblioteca:app",
        host="0.0.0.0",
        port=8082,  # puerto distinto a horarios (8080) y bienestar (8081)
        log_level="info",
    )
