# Show the available development commands.
default:
    @just --list

# Run Ruff over source and tests.
lint:
    uv run ruff check src tests

# Run the core tests in the current development environment.
test-core:
    uv run pytest -q tests/core

# Run the GUI tests with GUI dependencies installed.
test-gui:
    uv run --extra gui pytest -q tests/gui

# Run the complete test suite.
test:
    uv run --extra gui pytest -q

# Test the installed package without GUI dependencies.
test-core-isolated python="3.11":
    uv run --isolated --no-project --python {{python}} --with . --with pytest pytest -q tests/core

# Test the core package against the supported Python endpoints.
test-core-matrix:
    just test-core-isolated 3.11
    just test-core-isolated 3.13

# Build the wheel and source distribution.
build:
    uv build

# Run the usual local checks before committing.
check:
    just lint
    just test

# Run the comprehensive local checks before releasing.
release-check:
    just lint
    just test-core-matrix
    just test
    just build
