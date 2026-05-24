# Development Rules

## Conversational Style

- Keep answers short and concise.
- No emojis in commits, issues, PR comments, or code.
- Technical prose only; be direct.
- When the user asks a question, answer it first before making edits or running implementation commands.
- When responding to feedback, explicitly say whether you agree or disagree before saying what changed.

## Code Quality

- Read files in full before wide-ranging changes, before editing files you have not fully inspected, and when asked to investigate or audit.
- No `any` unless absolutely necessary.
- Inline single-line helpers that have only one call site.
- Check `node_modules` for external API types; do not guess.
- No inline imports (`await import()`, `import("pkg").Type`, dynamic type imports). Use top-level imports.
- Use erasable TypeScript syntax: no parameter properties, `enum`, `namespace`/`module`, `import =`, `export =`, or constructs that require nontrivial emit.
- Always ask before removing functionality or code that appears intentional.
- Do not preserve backward compatibility unless requested.

## Commands

- After code changes, run `npm run check` and fix all errors, warnings, and infos.
- Never run `npm run dev` or otherwise start the dev server unless the user explicitly asks; assume the user may already have it running in another terminal.
- Do not run production builds or broad test suites unless requested.
- For ad-hoc scripts, write them to `/tmp`, run them, then remove them. Do not embed multi-line scripts in `bash` commands.
- Never commit unless the user asks.

## Dependency and Install Security

- Treat npm dependency and lockfile changes as reviewed code.
- Direct external dependencies must be pinned to exact versions.
- Hydrate/update locally with `npm install --ignore-scripts`; clean/CI-style with `npm ci --ignore-scripts`.
- Do not run lifecycle scripts unless requested.

## Git

- Only commit files changed in this session.
- Stage explicit paths; never use `git add -A` or `git add .`.
- Never run `git reset --hard`, `git checkout .`, `git clean -fd`, `git stash`, `git commit --no-verify`, or force push.
