# CLI

The `spacellm` command-line interface, typer-based, Rich output.

## Installation

The CLI installs automatically with the package:

```bash
pip install spacellm
spacellm --help
```

## Commands

### `version`

```bash
$ spacellm version
0.1.0.dev0
```

### `runs`

List recent runs from a SQLite run database.

```bash
spacellm runs                   # default DB path: runs.db
spacellm runs --db custom.db
spacellm runs --limit 100
spacellm runs -n 5
```

If the database does not exist, the CLI emits a warning and exits cleanly with code 0.

### `show`

Show details for one run. The argument can be the full 32-character run id or any unique prefix (8 characters is conventional).

```bash
spacellm show abc12345
spacellm show abc12345 --db custom.db
```

Exit codes:

* `0`, run found, details printed.
* `1`, DB missing or no run matches the id / prefix.

### `profile`

Placeholder in v0.1, the static and sensitivity profilers are accessible programmatically via `spacellm.profiling`. The CLI subcommand lands in v0.2.

```bash
spacellm profile  # → friendly message pointing at the Python API
```

## Future commands (v0.5+)

* `spacellm bench`, run SpaceBench against a hardened model.
* `spacellm dose`, generate physics-lite dose profiles.
* `spacellm compare`, diff two runs side by side.
