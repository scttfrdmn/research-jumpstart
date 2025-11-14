# Linting and Static Analysis

This directory contains linting configurations and tools for the Research Jumpstart project.

## Quick Start

### Install Linting Tools

```bash
# Python linting tools
pip install -r .linting/requirements-lint.txt

# Markdown linting (requires Node.js)
npm install -g markdownlint-cli
```

### Run All Linters

```bash
./scripts/lint.sh
```

### Auto-fix Issues

```bash
./scripts/lint-fix.sh
```

## Linting Tools

### Python

- **Ruff**: Fast all-in-one linter (replaces flake8, pylint, isort)
  - Config: `.linting/ruff.toml`
  - Checks: code style, imports, complexity, bugbear patterns
  - Format: automatic code formatting

- **mypy**: Static type checker
  - Config: `.linting/pyproject.toml`
  - Relaxed for research code (allows missing imports)

- **Bandit**: Security vulnerability scanner
  - Checks for common security issues in Python code

### CloudFormation

- **cfn-lint**: AWS CloudFormation linter
  - Config: `.linting/.cfnlintrc.yaml`
  - Validates templates against AWS specifications
  - Checks best practices and common errors

### YAML

- **yamllint**: YAML linter
  - Config: `.linting/.yamllint`
  - Checks syntax, formatting, and structure

### Markdown

- **markdownlint**: Markdown linter
  - Config: `.linting/.markdownlint.json`
  - Checks heading structure, consistency, formatting

## CI/CD Integration

GitHub Actions workflow: `.github/workflows/lint.yml`

Runs on:
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop`

Separate jobs for:
- Python linting
- CloudFormation linting
- YAML linting
- Markdown linting

## Configuration Files

| File | Purpose |
|------|---------|
| `ruff.toml` | Ruff linter and formatter config |
| `pyproject.toml` | Python project config (black, mypy, isort) |
| `.cfnlintrc.yaml` | CloudFormation linter config |
| `.yamllint` | YAML linter config |
| `.markdownlint.json` | Markdown linter config |
| `requirements-lint.txt` | Python linting dependencies |

## Common Issues

### Python

**Issue**: Import order
**Fix**: `ruff check --fix` will automatically sort imports

**Issue**: Line too long
**Fix**: `ruff format` will automatically wrap lines

### CloudFormation

**Issue**: Invalid property for resource
**Fix**: Check AWS documentation for correct property names

**Issue**: Missing required properties
**Fix**: Add required properties to resource definitions

### Markdown

**Issue**: Inconsistent heading levels
**Fix**: Ensure headings increment by one level (H1 → H2 → H3)

**Issue**: Multiple top-level headings
**Fix**: Use only one H1 per document

## Ignoring Rules

### Python (Ruff)

```python
# Ignore specific rule for one line
result = expensive_function()  # noqa: F841

# Ignore all rules for one line
result = expensive_function()  # noqa

# Ignore specific rule for entire file
# ruff: noqa: E501
```

### CloudFormation

Add to `.linting/.cfnlintrc.yaml`:
```yaml
ignore_checks:
  - E3002  # Ignore specific rule
```

### Markdown

Add to `.linting/.markdownlint.json`:
```json
{
  "MD013": false  // Disable line length rule
}
```

## Best Practices

1. **Run linters before committing**
   ```bash
   ./scripts/lint-fix.sh && ./scripts/lint.sh
   ```

2. **Fix auto-fixable issues first**
   - Ruff can fix most Python issues automatically
   - Markdownlint can fix many Markdown issues

3. **Review manual fixes**
   - Some issues require understanding context
   - Don't blindly accept all suggestions

4. **Update configs as needed**
   - Add domain-specific exceptions
   - Balance strictness with productivity

## Resources

- [Ruff documentation](https://docs.astral.sh/ruff/)
- [cfn-lint rules](https://github.com/aws-cloudformation/cfn-lint)
- [yamllint documentation](https://yamllint.readthedocs.io/)
- [markdownlint rules](https://github.com/DavidAnson/markdownlint/blob/main/doc/Rules.md)
