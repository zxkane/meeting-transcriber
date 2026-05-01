# Audio Transcriber — FunASR

This project is a Claude Code plugin providing Chinese audio transcription
using FunASR with speaker diarization and LLM cleanup.

## Project Structure

- `plugins/audio-transcriber/` — The plugin source code
- `.agents/skills/` — Autonomous dev team skills (installed via skills.sh)
- `.claude/skills/` — Symlinks for Claude Code skill discovery
- `.claude-plugin/marketplace.json` — skills.sh marketplace registration
- `output/` — Generated transcription outputs (gitignored)

## Development Workflow

- Use `/autonomous-dev` skill for code changes (TDD, worktree isolation, CI verification)
- Use `/autonomous-review` skill for PR code review before merging
- Use `/create-issue` skill for structured GitHub issue creation
- Use `document-skills:skill-creator` skill to verify/review SKILL.md changes
- All skill source lives under `plugins/audio-transcriber/skills/audio-transcribe/`
- Scripts are in `scripts/`, references in `references/`
- The main entry point is `SKILL.md`
- Run tests: `cd plugins/audio-transcriber/skills/audio-transcribe/scripts && python3 -m pytest test_speaker_verification.py -v`

## Conventions

- English for code, comments, commit messages, and documentation
- Chinese audio content is the primary use case but the pipeline is language-agnostic
- Keep SKILL.md lean (<500 lines); move details to references/
- Scripts run directly from plugin directory via `$SCRIPTS` — never copy to CWD

## Placeholder Names in Tests, Comments, and Docs

Never use real person names (podcast hosts, guests, colleagues, public figures)
in test fixtures, docstrings, inline comments, or example strings. Always use
generic placeholder names instead.

- **Chinese**: 张三, 李四, 王五, 赵六, 张飞, 关羽, 刘备, 李雷, 韩梅梅
- **English**: Alice, Bob, Carol, Dave, Eve, Host, Guest
- **Roles**: use generic labels like `Speaker 1`, `Host`, `Guest` when a name
  is not needed

This applies to:
- Test data (`test_*.py`)
- Docstring examples (e.g., `"主播：张三"`)
- Inline comments illustrating input/output
- Sample files under `references/` or `docs/`

When a real name appears in a user-provided show notes or audio file during
development, strip it before checking anything into the repo.
