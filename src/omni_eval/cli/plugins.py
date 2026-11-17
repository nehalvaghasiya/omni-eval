"""Inspect plugin support without pretending installation is integration."""

import json

import typer

from omni_eval.cli.errors import fail as _failure
from omni_eval.plugins.catalog import Catalog
from omni_eval.plugins.prepare import Preparer


def plugin_list(dry_run: bool = False) -> None:
    for name, info in sorted(Catalog().list().items()):
        typer.echo(f"{name}: {info['status']}")


def plugin_show(name: str, dry_run: bool = False) -> None:
    try:
        typer.echo(json.dumps(Catalog().get(name), indent=2))
    except Exception as exc:
        _failure(exc)


def plugin_doctor(name: str, dry_run: bool = False) -> None:
    try:
        typer.echo(f"{name}: {Preparer().check(name)}")
    except Exception as exc:
        _failure(exc)


def plugin_setup(name: str, offline: bool = False, dry_run: bool = False) -> None:
    try:
        if dry_run:
            typer.echo(f"DEMO / SYNTHETIC setup check: {name}: {Preparer().check(name)}")
        else:
            typer.echo(Preparer().prepare(name, offline=offline))
    except Exception as exc:
        _failure(exc)
