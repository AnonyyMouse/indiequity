# Contributing to indiequity

Thank you for your interest in improving `indiequity`! We welcome contributions from financial analysts, economics students, and software engineers.

## Development Workflow
1. Fork the repository and clone it locally.
2. Set up a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```
3. Create a descriptive feature branch:
   ```bash
   git checkout -b feature/my-new-ratio
   ```
4. Run tests before submitting changes:
   ```bash
   pytest
   ```
5. Commit and open a Pull Request against `main`.

## Code Guidelines
- Add type hints and docstrings for all new functions.
- Keep tests offline by adding mock statement fixtures rather than relying on live API calls.
