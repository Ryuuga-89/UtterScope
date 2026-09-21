"""HTML report renderer (Jinja2, self-contained)."""

from __future__ import annotations

from jinja2 import Environment, PackageLoader, StrictUndefined, select_autoescape

from utterscope.report.view import ReportView

_TEMPLATE_NAME = "report.html.j2"


def _environment() -> Environment:
    return Environment(
        loader=PackageLoader("utterscope.report", "templates"),
        autoescape=select_autoescape(enabled_extensions=("html", "j2", "xml")),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_html(view: ReportView) -> str:
    """Render ``report.html`` content (Japanese, interactive timeline)."""
    template = _environment().get_template(_TEMPLATE_NAME)
    return template.render(view=view)
