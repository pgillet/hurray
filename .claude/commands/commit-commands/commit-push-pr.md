---
allowed-tools: Bash(git fetch:*), Bash(git rebase:*), Bash(git checkout:*), Bash(git add:*), Bash(git status:*), Bash(git push:*), Bash(git commit:*), Bash(git log:*), Bash(git diff:*), Bash(git branch:*)
description: Commit, push, and open a PR (rebases from main first)
---

## Context

- Current git status: !`git status`
- Current git diff (staged and unstaged changes): !`git diff HEAD`
- Current branch: !`git branch --show-current`

## Your task

Based on the above changes:

1. Create a new branch if on main
2. Run `git fetch origin && git rebase origin/main` to ensure the branch is up to date before committing
3. Create a single commit with an appropriate message
4. Push the branch to origin with `-u` flag
5. Create a pull request using the `mcp__plugin_github_github__create_pull_request` tool (owner: pgillet, repo: hurray, base: main)
6. You have the capability to call multiple tools in a single response. You MUST do all of the above in a single message. Do not use any other tools or do anything else. Do not send any other text or messages besides these tool calls.
