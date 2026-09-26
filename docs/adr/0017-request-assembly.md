# ADR-0017: Assemble Requests, Not System Prompts

**Date:** 2026-08-15  
**Status:** accepted  
**Supersedes:** [ADR-0016](0016-structured-prompt-assembly.md)  
**Deciders:** Project maintainers

## Context

ADR-0016 proposed a prompt type modelled on DeepSeek Harness's `system-prompt`
package: ordered *system* sections in numeric bands with strict `{{variable}}`
interpolation. Reading the two consumers before implementing disproved its
premise on three counts.

**System prompts carry no variables.** stria's `CRITERIA_SYSTEM`,
`LAW_LAYER_SYSTEM`, and `RELATIONSHIPS_SYSTEM` and doom's `SYSTEM_PROMPT` are
static strings; doom's `SCREENING_PROMPT` is an f-string resolved once at
import. Every runtime variable lives in *user* content, interpolated with
`str.format`. A `{{var}}` feature on system sections would have shipped with no
consumer.

**Structure already lives in the block layer.** Both repos independently
converged on the same two-block shape — a cached document followed by a
per-call instruction:

```python
blocks = [{"text": document, "cache": True}, {"text": instruction, "cache": False}]
```

That shape is carried by `Block = dict[str, Any]`, an untyped bag whose
docstring in `llm/wire/types.py` is a broken stub. Sectioning system text while
leaving this untyped would have added structure where there is none and left it
absent where it matters.

**A live bug shows the cost.** stria's `law_layer.py` and `relationships.py`
call `.format(citation=citation, document="")`, rendering *"The complete text
you may use follows"* immediately above an empty `<law_text></law_text>`, while
the real document travels in a separate cached block. The prompt misdescribes
its own contents, and `prompt_hash` records a template whose rendered form never
held the document. Nothing in the current types makes this expressible as an
error.

One further constraint: the wires disagree about whether a block boundary
survives. Anthropic preserves blocks and marks cached ones with
`cache_control: {"type": "ephemeral"}`; the OpenAI-compatible wire joins them
with `"\n\n"` and drops the cache flag. Anthropic also passes `system=` as a
bare string, so system-side caching would require a change on one wire only.

## Decision

Add `emissary.llm.prompt` with a `Prompt` value modelling **the whole request
input** — system text plus ordered content blocks — not system sections.

```python
@dataclass(frozen=True)
class Prompt:
    system: str
    blocks: tuple[TextBlock, ...] = ()

    @property
    def fingerprint(self) -> str: ...
```

Four decisions inside that:

- **Blocks reuse the existing `TextBlock(text, cache)`** from `llm/messages.py`
  rather than introducing a parallel type. It already carries exactly this
  shape, `UserMessage` already holds it, and one content-block type across the
  conversational and single-shot paths is the point.
- **`Prompt` replaces `Block = dict[str, Any]`** as the documented way to build
  a request. The dict form keeps working at the call boundary.
- **`fingerprint` hashes exactly what is sent** — system text plus every block's
  text and cache flag — so a prompt change is a visible, comparable
  experimental variable in events and evaluation records.
- **No interpolation of any kind.** Callers build block text in Python, where
  they already do. This is the correction ADR-0016 needed most.

Wire behavior is unchanged and stays pinned by existing tests: Anthropic emits
per-block `cache_control` on marked blocks; OpenAI-compatible joins with
`"\n\n"` and drops cache. Making the system prompt cacheable on Anthropic is a
separate decision and is **not** taken here.

Adoption is explicit rather than magical: `call_tool` and `call_choice` accept
either a `prompt=` or the existing `system=`/`blocks=` pair, and reject
receiving both.

`Agent.instructions` **stays a plain `str`.** This ADR originally said it would
accept a `Prompt`; implementation showed that to be wrong. A `Prompt` is a whole
request — system text *and* content blocks — but an agent's content comes from
the task and tool results, which travel as messages. Accepting a `Prompt` there
would ship a type half of which is meaningless in that position, and the
alternative (accept it but reject any prompt carrying blocks) is the same flaw
with a runtime check bolted on. Callers wanting an audit identity for agent
instructions can take `Prompt(instructions).fingerprint` themselves.

## Alternatives Considered

- **ADR-0016 as drafted** (system sections, order bands, `{{var}}`): builds
  machinery for a need no consumer has, while leaving the untyped block layer —
  where the real structure and the real bug are — untouched. Superseded.
- **A template engine (Jinja et al.):** stringly-typed logic at a trust
  boundary, a new dependency, and injection surface, to serve interpolation
  that consumers do in Python today. Rejected.
- **DSH's registry with contribution/waterfall hooks:** correct for a plugin
  ecosystem; emissary has two consumers and explicit composition. A value the
  caller constructs beats a mutable registry. Rejected.
- **Making `Prompt` mandatory (breaking `system=`/`blocks=`):** cleaner surface,
  but forces a synchronized change across two path-dependency consumers for no
  behavioral gain. Rejected.

## Consequences

### Positive

- One typed shape for building a request across stria, doom, and the harness,
  with the cached-document/instruction distinction expressed rather than
  conventional.
- The stria placeholder bug becomes inexpressible: a document is a block, not a
  template hole that can be filled with `""`.
- Fingerprints make evaluation and trajectories honest about which prompt ran.
- `llm/wire/types.py`'s untyped `Block` and its broken docstring go away.

### Negative

- Two accepted call shapes (`prompt=` versus `system=`/`blocks=`) until
  consumers migrate, and a guard rejecting both.
- Migration touches two other repositories.

### Risks

- **stria's audit trail.** `prompt_hash(SYSTEM, USER)` is stored on every draft.
  If `Prompt.fingerprint` replaces it, previously stored hashes no longer
  correspond to any computable value. Decide deliberately during migration
  whether to keep `prompt_hash` alongside, backfill, or accept the
  discontinuity — do not let it change as a side effect.
- Scope creep back toward templating. The boundary: `Prompt` holds text and a
  cache flag. Any conditional or loop belongs in the Python that builds it.
