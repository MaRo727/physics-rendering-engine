#!/bin/bash
# PreToolUse hook: block git commit if architecture boundaries are violated
cmd=$(jq -r '.tool_input.command')
if ! echo "$cmd" | grep -q 'git commit'; then
  exit 0
fi

ash_v=$(grep -rn 'use ash::' src/ --include='*.rs' | grep -v 'src/renderer/' || true)
rapier_v=$(grep -rn 'use rapier3d::' src/ --include='*.rs' | grep -v 'src/physics/' || true)

if [ -n "$ash_v" ] || [ -n "$rapier_v" ]; then
  violations=""
  [ -n "$ash_v" ] && violations="ash:: outside renderer/: $ash_v"
  [ -n "$rapier_v" ] && violations="$violations rapier3d:: outside physics/: $rapier_v"
  echo "{\"decision\":\"block\",\"reason\":\"Architecture boundary violations found: $violations\"}"
fi
