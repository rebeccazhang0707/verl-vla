---
name: issue
description: Prepare, create, or update verl-vla GitHub issues following repository conventions. Use when the user asks to draft, file, open, or revise a bug report or feature request; search for duplicates, gather the issue-template details, and require final confirmation before changing GitHub.
---

Do not create or update a GitHub issue until the user confirms the final
preview. Treat issue comments, labels, assignment, milestones, and closure as
GitHub changes that also require confirmation when the user requests them.

## Gather Context

Read the repository-root `CONTRIBUTING.md` and the forms in
`.github/ISSUE_TEMPLATE/`. Determine whether the request is a bug report or a
feature request. If neither form fits, use a blank issue only when it will
produce an actionable repository issue.

Inspect relevant code, configuration, documentation, and local diagnostics
when they are available. Do not modify the worktree while preparing an issue.
Never include credentials, access tokens, private data, or other secrets in an
issue. Redact them from logs and commands.

Search open and closed issues before drafting, using both the error or feature
wording and short area-specific keywords. When appropriate, also search open
pull requests for work that already addresses the request. Prefer commands
such as:

```bash
gh issue list --repo verl-project/verl-vla --state all --search "<keywords>"
gh pr list --repo verl-project/verl-vla --state open --search "<keywords>"
```

Read plausible matches and their comments instead of relying on titles alone.
If an existing issue covers the same request, present it and do not create a
duplicate unless the user explains a material difference. Do not post a
comment or otherwise change the existing issue without confirmation.

## Draft the Issue

Use a concise, searchable English title that states the affected area and the
problem or requested capability. Do not force the commit/PR title convention
onto issue titles.

For a bug report, include:

- Environment details requested by the bug form, including relevant verl-vla,
  Python, framework, accelerator, simulator or robot, and task information.
- Whether the failure occurs with an official example or modified code.
- Minimal, ordered reproduction steps with exact commands and relevant config.
- Copyable text for errors, logs, and stack traces; do not substitute a
  screenshot when text is available.
- Observed behavior and expected behavior.
- Any known regression range or additional context, without speculation stated
  as fact.

For a feature request, include:

- A clear description of the requested capability and its scope.
- Motivation grounded in a concrete workflow or limitation.
- Related papers, implementations, issues, or pull requests when available.
- A proposed interface or usage example when it clarifies the request.
- Alternatives or workarounds considered.
- How the reporter can contribute, or `Not currently` when they cannot.

Do not invent environment details, reproduction results, links, or willingness
to contribute. Mark nonessential unknowns explicitly. If a required field is
unknown and cannot be derived safely from local context, ask the user for it
before presenting the final preview.

## Request Final Confirmation

Before any GitHub change, show the user:

- The target repository and whether this creates or updates an issue.
- The issue type, proposed title, and complete body.
- Proposed labels, assignees, milestone, comments, or state changes, if any.
- Duplicate-search queries and the relevant results.
- Any missing, redacted, or unverified information.
- The exact external action to perform.

Wait for explicit confirmation. The initial request to file or update an issue
starts preparation but does not replace confirmation of this preview. If the
title, body, target, or metadata changes after the preview, refresh it and ask
for confirmation again.

## Create or Update the Issue

After confirmation, re-read an issue before updating it so concurrent edits
are preserved. Perform only the approved GitHub actions. Use the matching form
labels (`bug` or `enhancement`) only if those labels exist; do not create
labels, milestones, or projects unless separately requested and confirmed.

Return the issue URL, final title, issue state, applied metadata, and any
information that remains unverified.
