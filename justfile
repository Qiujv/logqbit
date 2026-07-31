# Use PowerShell explicitly instead of relying on whichever Unix-like shell is
# discoverable on PATH when Just runs on Windows.
set windows-shell := ["powershell.exe", "-NoLogo", "-NoProfile", "-Command"]

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
    uv run --isolated --no-project --python {{python}} --with-editable . --with pytest pytest -q tests/core

# Test the core package against the supported Python endpoints.
test-core-matrix: (test-core-isolated "3.11") (test-core-isolated "3.13")

# Build the wheel and source distribution.
build:
    uv build

# Build the documentation and treat warnings as errors.
docs:
    uv run mkdocs build --strict

# Run the usual local checks before committing.
check: lint test docs

# Run the comprehensive local checks before releasing.
release-check: lint test-core-matrix test docs build
