# Contributing

Thank you for your interest in contributing to the FBWM-FTOPSIS-PPO framework.

## Getting Started

1. Fork the repository
2. Create a feature branch from `develop`:
   ```bash
   git checkout develop
   git checkout -b feature/your-feature-name
   ```
3. Install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

## Development Workflow

This project follows the **Git Flow** branching model:

- `main` — stable release branch (protected)
- `develop` — integration branch for ongoing work
- `feature/*` — feature branches created from `develop`

## Code Standards

- Follow [PEP 8](https://peps.python.org/pep-0008/) style guidelines
- Use type hints where practical
- Write descriptive docstrings for all public functions and classes
- Keep functions focused and modular

## Testing

Run the test suite before submitting:

```bash
python -m pytest test/ -v
```

## Commit Messages

Use clear, descriptive commit messages:

```
<type>: <short summary>

<optional body with details>
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `style`

## Pull Requests

1. Ensure all tests pass
2. Update documentation if needed
3. Target the `develop` branch (not `main`)
4. Fill in the pull request template

## Reporting Issues

Use the GitHub issue templates for:
- **Bug reports** — include Python version, OS, and steps to reproduce
- **Feature requests** — describe the motivation and proposed solution
