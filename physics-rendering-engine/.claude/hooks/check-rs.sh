#!/bin/bash
# PostToolUse hook: run clippy after .rs file edits
f=$(jq -r '.tool_response.filePath // .tool_input.file_path')
if echo "$f" | grep -qE '\.rs$'; then
  cargo clippy --all-targets -- -D warnings 2>&1 | head -50
fi
