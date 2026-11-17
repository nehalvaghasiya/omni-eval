"""Stable user-facing CLI exit codes without tracebacks."""

import typer


def fail(exc: Exception) -> None:
    if isinstance(exc, PermissionError):
        code = 5
    elif isinstance(exc, (RuntimeError, ConnectionError, TimeoutError)):
        code = 3
    else:
        code = 2
    typer.echo(f"OmniEval: {exc}", err=True)
    raise typer.Exit(code=code)
