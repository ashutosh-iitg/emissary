# ADR-0010: Use Draft 2020-12 JSON Schema at Tool Boundaries

**Date:** 2026-08-14  
**Status:** accepted  
**Deciders:** Project maintainers

## Context

Tool inputs and outputs cross an untrusted model/extension boundary. Handwritten checks for required fields, nested values, and composition keywords would implement an incomplete schema language.

## Decision

Use the maintained `jsonschema` package and its `Draft202012Validator` for tool contract validation. Validate every call in a batch before any effect executes and validate declared outputs before returning them to the model.

## Alternatives Considered

- **Handwritten validation:** avoids a dependency, but silently supports only a subset and creates inconsistent trust boundaries. Rejected.
- **Pydantic-only tool models:** strong Python ergonomics, but makes one Python modeling framework the public wire contract. Rejected for the core.

## Consequences

### Positive

- Standards-compliant nested validation and reusable JSON contracts.
- Model-visible schemas and execution-time validation use the same vocabulary.

### Negative

- Adds a runtime dependency and its transitive packages.

### Risks

- Schema drafts can evolve; the selected draft is explicit and contract tests pin behavior.

