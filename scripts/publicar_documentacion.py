"""Genera las páginas HTML de los artefactos públicos de la entrega."""
import argparse
from html import escape
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/RICK1130042/Netflix_clasificadores"
PAGES = [
    "README.md", "negocio.md", "contrato-prompt.md", "contrato-datos.md",
    "graficas.md", "acceso.md", "CLAUDE.md", "datos/README.md",
    "prompts/B1.md", "prompts/B2.md", "prompts/B3.md", "prompts/B4.md",
    "prompts/analisis-sdlc.md", "docs/entregable-3.md", "docs/entrega-final.md",
    "docs/evidencia-ci.md", "docs/observaciones-21-oct.md",
    "docs/design-thinking.md", "docs/auditoria-entrega.md",
    "investigacion/00-sdlc-genai.md", "investigacion/01-jerarquia.md",
    "investigacion/02-contraste.md", "investigacion/03-acceso.md",
    "investigacion/04-metricas.md",
    "Netflix_B2_Build_Test/contrato-datos.md",
    "Netflix_B2_Build_Test/investigacion/claude.md",
]


def output_name(path):
    return "index.html" if path == "README.md" else path[:-3].replace("/", "-") + ".html"


def rewrite(link, source, sha, image=False):
    url = urlsplit(link)
    if url.scheme or url.netloc or not url.path:
        return link
    dest = (source.parent / unquote(url.path)).resolve()
    target = dest.relative_to(ROOT).as_posix()
    anchor = f"#{url.fragment}" if url.fragment else ""
    if not dest.exists():
        raise ValueError(f"Enlace roto en {source.relative_to(ROOT)}: {link}")
    if target in PAGES:
        return output_name(target) + anchor
    if image:
        return f"https://raw.githubusercontent.com/RICK1130042/Netflix_clasificadores/{sha}/{target}"
    kind = "tree" if dest.is_dir() else "blob"
    return f"{REPO}/{kind}/{sha}/{target}{anchor}"


def build(directory, sha):
    renderer = MarkdownIt("commonmark", {"html": False}).enable("table")
    directory.mkdir(parents=True, exist_ok=True)
    for rel in PAGES:
        source = ROOT / rel
        raw = source.read_text()
        title = next((line[2:] for line in raw.splitlines() if line.startswith("# ")), rel)
        tokens = renderer.parse(raw)
        for token in tokens:
            for child in token.children or ():
                if child.type in {"link_open", "image"}:
                    attribute = "src" if child.type == "image" else "href"
                    child.attrSet(attribute, rewrite(child.attrGet(attribute), source, sha, child.type == "image"))
        body = renderer.renderer.render(tokens, renderer.options, {})
        (directory / output_name(rel)).write_text(f"""<!doctype html>
<html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>{escape(title)} · Netflix Churn</title>
<style>
body{{margin:0;background:#111318;color:#f5f6fa;font:1rem/1.65 system-ui}}
main{{max-width:68rem;margin:auto;padding:2rem 1.2rem;overflow-x:auto}}
a{{color:#83d1ff}} a:focus-visible{{outline:2px solid #fff}}
h1,h2{{line-height:1.25}} table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #788394;padding:.55rem;text-align:left;vertical-align:top}}
pre{{padding:1rem;background:#1b1f27;overflow:auto}} img{{max-width:100%}}
</style><main><nav><a href="index.html">Índice de la entrega</a></nav>
{body}<footer>Versión <a href="{REPO}/commit/{sha}">{sha[:12]}</a></footer></main></html>
""")
    (directory / ".nojekyll").touch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--sha", required=True)
    args = parser.parse_args()
    if not re.fullmatch("[0-9a-f]{40}", args.sha):
        parser.error("--sha requiere un SHA de Git de 40 dígitos hexadecimales")
    build(args.directory, args.sha)
