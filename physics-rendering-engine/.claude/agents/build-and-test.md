# Build & Test

Compile the project and run clippy to catch warnings, errors, and lint issues.

## Steps

1. Run `cargo clippy --all-targets -- -D warnings` from the project root
2. If clippy fails, analyze each error/warning and fix it
3. After fixes, run `cargo build` to confirm the project compiles cleanly
4. Report a summary: number of issues found, what was fixed, and final build status

## Rules

- Never suppress warnings with `#[allow(...)]` unless there's a genuine reason
- If a warning is about unused code that is clearly intentional (e.g., future milestone code), mention it but don't delete it
- Fix clippy lints in-place — prefer the idiomatic Rust form clippy suggests
