import re
from typing import Any


def normalize_template(template: str) -> str:
    """
    Normalize single-brace variables to Jinja2 double-brace format.
    {var} -> {{ var }}
    Preserves existing {{ var }} syntax.
    """
    pattern = r"\{(?!\{)(\w+)\}"

    def replacer(match):
        var_name = match.group(1)
        return f"{{{{ {var_name} }}}}"

    return re.sub(pattern, replacer, template)


def render_template(template: str, variables: Any) -> str:
    """
    Render a template that may use {var} or {{ var }} syntax.
    Automatically normalizes the template before rendering.
    """
    from jinja2 import Template

    if variables is None:
        variables = {}

    normalized = normalize_template(template)
    template_obj = Template(normalized)
    return template_obj.render(**variables)
