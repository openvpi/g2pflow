## Comments and documentation

Write for readers who have no knowledge of the conversation or implementation process. Describe the current, deliverable behavior as a coherent whole.

- Keep conversation context, review replies, debugging notes, temporary workarounds that no longer apply, and task-specific validation logs out of code comments and product documentation. Put detailed change history in PR descriptions or a dedicated changelog when needed.
- Integrate changes into the relevant existing explanation. Rewrite or remove outdated passages instead of appending isolated sections that only advertise the latest change. Keep examples, parameter descriptions, and resource instructions consistent with the implementation.
- Keep comments short and focused on non-obvious reasoning, invariants, constraints, or tradeoffs. Avoid lengthy explanations, restating the code, narrating edits, or comparing against an obsolete implementation.
- All code comments must use ASCII characters unless a special quotation requires preserving non-ASCII text. Use ASCII punctuation such as `->`, `-`, and `'`.
- Keep user documentation focused on usage, supported behavior, prerequisites, and meaningful limitations. Include internal details only when they help the intended reader understand or use the system.
- Use concise, factual language. Avoid promotional claims, exaggerated guarantees, and assertions of complete support or equivalence that the implementation does not establish.
- In overview prose and cross-module descriptions, prefer stable purpose, contracts and constraints over inventories of current callers, backends or dependencies. Keep changing facts in their authoritative source and link to package metadata, workflow definitions or the relevant reference rather than repeating dependency lists, CI matrices, test inventories or release numbers. Prefer version-independent examples.
- Preserve accurate comments explaining nearby code, including index ranges, data shapes, algorithm steps and local helper relationships. These can be maintained with the code they describe; do not generalize or delete them solely because implementation details may change. Retain concrete API/resource requirements, protocol versions and measurements where readers need them.
- Preserve required attribution and licensing notices, along with explanations of constraints that still apply. Cleanup should remove noise without discarding useful technical context.

## Testing principles

Tests should protect observable behavior and explicit contracts. Before adding a test, identify the functional failure it would catch. A normal extension or refactor should not require changing tests whose contracts remain intact.

- Do not freeze component inventories or counts, such as the complete set of registered converters. Exercise registration through configuration and actual use.
- Do not lock down private helper names, internal field layouts, or exact call sequences unless they are part of a required interface. Assert meaningful outcomes.
- Do not add checks solely to prove that an obsolete field or implementation has been removed. Test the supported format and behavior directly.
- Do not compute expected results with the implementation under test, copy its algorithm into the test, or mock away all meaningful behavior and only assert the arranged calls.
- Cover shared rules in the layer that owns them. Consumer tests should verify integration and consumer-specific behavior instead of duplicating the same cases across backends.
- Do not add unreachable-state checks, exhaustive combinations, or exotic cases without a credible failure mode or an explicit supported boundary. Coverage percentage alone does not justify a test.
- Keep ordinary tests offline and deterministic. Real-resource integration tests should be opt-in and use supplied resources instead of downloading large models on every run. Coordinate concurrency with explicit synchronization rather than fixed sleeps.

Retain regression coverage for real failures. Organize and name these tests around the final behavior they protect, without references to a conversation, patch, or debugging history.

## Commit messages

Commit messages must be concise and contain exactly one line, with no body or footer. Use Conventional Commits in the format `<type>(<scope>): <description>`.
