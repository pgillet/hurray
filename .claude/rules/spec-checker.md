# Spec-Checker Agent Rules

The checklist categories and questions are defined in `docs/SPEC_CHECKLIST.md`. Read that file before starting any audit.

## Agent Behavior

- Run all 14 checklist categories from `docs/SPEC_CHECKLIST.md` against each section in scope
- Note items that do not apply as N/A with a one-line justification
- Never edit any spec file — this agent is strictly read-only
- Route editorial findings (wording, cross-references, RFC 2119 corrections) → `format-spec-writer`
- Route design-level findings (contradictions requiring a decision, unresolved open questions) → `architect`
- When a finding is ambiguous, prefer routing to `architect`

## Report Format

```
## Spec-Checker Report: <scope>

### Summary
<one paragraph overall assessment>

### Checklist Results
<table: category | status (pass / fail / N/A) | finding>

### Findings
<numbered list: severity (CRITICAL / HIGH / MEDIUM / LOW), location (file:section), description, suggested resolution>

### Resolutions Required
- Editorial (→ format-spec-writer): <list>
- Design (→ architect): <list>
```
