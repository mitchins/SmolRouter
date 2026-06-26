"""Dashboard request filtering using a small fielded query syntax."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Iterable, Sequence


FIELD_ALIASES = {
    "host": "host",
    "ip": "host",
    "source": "host",
    "source_ip": "host",
    "provider": "provider",
    "service": "provider",
    "model": "model",
    "project": "project",
    "identity": "identity",
}


class DashboardFilterError(ValueError):
    """Raised when a dashboard filter query cannot be parsed."""

    def __init__(self, invalid_terms: Sequence[str], message: str):
        super().__init__(message)
        self.invalid_terms = list(invalid_terms)


@dataclass(frozen=True)
class FilterClause:
    field: str
    value: str


@dataclass(frozen=True)
class DashboardFilterQuery:
    raw: str
    clauses: tuple[FilterClause, ...]
    text_terms: tuple[str, ...]

    @property
    def active(self) -> bool:
        return bool(self.clauses or self.text_terms)

    def to_meta(self, matched_count: int | None = None, limit: int | None = None) -> dict:
        return {
            "query": self.raw,
            "active": self.active,
            "clauses": [{"field": clause.field, "value": clause.value} for clause in self.clauses],
            "terms": list(self.text_terms),
            "matched_count": matched_count,
            "limit": limit,
        }


def parse_dashboard_filter_query(raw_query: str | None) -> DashboardFilterQuery:
    """Parse the dashboard filter query.

    Syntax:
    - fielded clauses: host:1.2.3.4 provider:google model:gemma
    - project/identity clauses: project:my-project identity:facade_key:my-project
    - quoted values: provider:"google genai"
    - bare terms: matched across common dashboard fields
    """

    normalized_query = (raw_query or "").strip()
    if not normalized_query:
        return DashboardFilterQuery(raw="", clauses=(), text_terms=())

    try:
        tokens = shlex.split(normalized_query)
    except ValueError as exc:
        raise DashboardFilterError([], f"Invalid filter query: {exc}") from exc

    clauses: list[FilterClause] = []
    text_terms: list[str] = []
    invalid_terms: list[str] = []

    for token in tokens:
        if ":" not in token:
            text_terms.append(token)
            continue

        field_name, raw_value = token.split(":", 1)
        field = FIELD_ALIASES.get(field_name.strip().lower())
        value = raw_value.strip()
        if not field or not value:
            invalid_terms.append(token)
            continue

        clauses.append(FilterClause(field=field, value=value))

    if invalid_terms:
        supported_fields = ", ".join(sorted(set(FIELD_ALIASES.values())))
        invalid_list = ", ".join(invalid_terms)
        raise DashboardFilterError(
            invalid_terms,
            f"Invalid filter term(s): {invalid_list}. Supported fields: {supported_fields}.",
        )

    return DashboardFilterQuery(raw=normalized_query, clauses=tuple(clauses), text_terms=tuple(text_terms))


def filter_request_logs(logs: Iterable[object], query: DashboardFilterQuery) -> list[object]:
    """Filter request logs using AND semantics across all terms and clauses."""

    if not query.active:
        return list(logs)

    return [log for log in logs if matches_dashboard_filter(log, query)]


def matches_dashboard_filter(log: object, query: DashboardFilterQuery) -> bool:
    """Return True when a log entry satisfies the parsed filter query."""

    for clause in query.clauses:
        needle = clause.value.casefold()
        haystacks = [value.casefold() for value in _field_values(log, clause.field) if value]
        if not any(needle in value for value in haystacks):
            return False

    if not query.text_terms:
        return True

    searchable_values = [value.casefold() for value in _searchable_values(log) if value]
    return all(any(term.casefold() in value for value in searchable_values) for term in query.text_terms)


def _field_value_pair(*values: object) -> tuple[str, ...]:
    return tuple(_stringify(value) for value in values)


def _field_values(log: object, field: str) -> tuple[str, ...]:
    if field == "host":
        return _field_value_pair(getattr(log, "source_ip", None))

    if field == "provider":
        return _field_value_pair(getattr(log, "provider_id", None), getattr(log, "service_type", None))

    if field == "model":
        return _field_value_pair(getattr(log, "mapped_model", None), getattr(log, "original_model", None))

    if field == "project":
        identity_subject = getattr(log, "identity_subject_id", None)
        identity_kind = getattr(log, "identity_kind", None)
        canonical = f"{identity_kind}:{identity_subject}" if identity_kind and identity_subject else None
        return _field_value_pair(
            identity_subject,
            canonical,
            getattr(log, "identity_display_name", None),
        )

    if field == "identity":
        identity_kind = getattr(log, "identity_kind", None)
        identity_subject = getattr(log, "identity_subject_id", None)
        canonical = f"{identity_kind}:{identity_subject}" if identity_kind and identity_subject else None
        return _field_value_pair(identity_kind, identity_subject, canonical, getattr(log, "identity_display_name", None))

    return _field_value_pair(None, None)


def _searchable_values(log: object) -> tuple[str, ...]:
    status_code = getattr(log, "status_code", None)
    identity_subject = getattr(log, "identity_subject_id", None)
    identity_kind = getattr(log, "identity_kind", None)
    identity_display_name = getattr(log, "identity_display_name", None)
    canonical = f"{identity_kind}:{identity_subject}" if identity_kind and identity_subject else None
    return (
        _stringify(getattr(log, "source_ip", None)),
        _stringify(getattr(log, "provider_id", None)),
        _stringify(getattr(log, "service_type", None)),
        _stringify(getattr(log, "original_model", None)),
        _stringify(getattr(log, "mapped_model", None)),
        _stringify(getattr(log, "path", None)),
        _stringify(getattr(log, "method", None)),
        _stringify(status_code),
        _stringify(identity_kind),
        _stringify(identity_subject),
        _stringify(canonical),
        _stringify(identity_display_name),
    )


def _stringify(value: object | None) -> str:
    if value is None:
        return ""

    text = str(value).strip()
    return "" if text.lower() in {"none", "pending"} else text
