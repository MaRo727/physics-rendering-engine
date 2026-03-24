#!/bin/bash
# PostToolUse hook: prompt agent-updater after a successful git commit
cmd=$(jq -r '.tool_input.command // empty')
if echo "$cmd" | grep -qE 'git commit'; then
  echo "POST-COMMIT: A commit was just made. Run the agent-updater agent to check if any agent definitions need updating based on this commit."
fi
