#!/bin/bash
# PostToolUse hook: validate shader after edits
f=$(jq -r '.tool_response.filePath // .tool_input.file_path')
if echo "$f" | grep -qE '\.(rgen|rchit|rmiss|vert|frag)$'; then
  glslangValidator --target-env vulkan1.2 -V "$f" 2>&1
fi
