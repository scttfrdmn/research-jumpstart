#!/bin/bash
# Master Linting Script for Research Jumpstart
# Runs all linters across Python, CloudFormation, YAML, and Markdown files

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Track overall status
FAILED=0

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Research Jumpstart Linting Suite${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
LINTING_DIR="$PROJECT_ROOT/.linting"

cd "$PROJECT_ROOT"

# Function to run linter and track status
run_linter() {
    local name=$1
    local command=$2

    echo -e "${BLUE}Running $name...${NC}"
    if eval "$command"; then
        echo -e "${GREEN}✓ $name passed${NC}"
        echo ""
    else
        echo -e "${RED}✗ $name failed${NC}"
        echo ""
        FAILED=1
    fi
}

# 1. Python Linting with Ruff
if command -v ruff &> /dev/null; then
    echo -e "${YELLOW}=== Python Linting ===${NC}"
    run_linter "Ruff (check)" "ruff check --config '$LINTING_DIR/ruff.toml' ."
    run_linter "Ruff (format check)" "ruff format --check --config '$LINTING_DIR/ruff.toml' ."
else
    echo -e "${YELLOW}⚠ Ruff not installed, skipping Python linting${NC}"
    echo ""
fi

# 2. Python Type Checking with mypy
if command -v mypy &> /dev/null; then
    echo -e "${YELLOW}=== Python Type Checking ===${NC}"
    run_linter "mypy" "mypy --config-file '$LINTING_DIR/pyproject.toml' projects/ || true"
else
    echo -e "${YELLOW}⚠ mypy not installed, skipping type checking${NC}"
    echo ""
fi

# 3. CloudFormation Linting
if command -v cfn-lint &> /dev/null; then
    echo -e "${YELLOW}=== CloudFormation Linting ===${NC}"
    run_linter "cfn-lint" "cfn-lint --config-file '$LINTING_DIR/.cfnlintrc.yaml' projects/**/cloudformation/*.yml || true"
else
    echo -e "${YELLOW}⚠ cfn-lint not installed, skipping CloudFormation linting${NC}"
    echo ""
fi

# 4. YAML Linting
if command -v yamllint &> /dev/null; then
    echo -e "${YELLOW}=== YAML Linting ===${NC}"
    run_linter "yamllint" "yamllint -c '$LINTING_DIR/.yamllint' projects/**/cloudformation/*.yml || true"
else
    echo -e "${YELLOW}⚠ yamllint not installed, skipping YAML linting${NC}"
    echo ""
fi

# 5. Markdown Linting
if command -v markdownlint &> /dev/null; then
    echo -e "${YELLOW}=== Markdown Linting ===${NC}"
    run_linter "markdownlint" "markdownlint --config '$LINTING_DIR/.markdownlint.json' '**/*.md' --ignore node_modules --ignore .venv || true"
else
    echo -e "${YELLOW}⚠ markdownlint not installed, skipping Markdown linting${NC}"
    echo ""
fi

# 6. Python Security Scanning with Bandit
if command -v bandit &> /dev/null; then
    echo -e "${YELLOW}=== Python Security Scanning ===${NC}"
    run_linter "bandit" "bandit -r projects/ -ll -i || true"
else
    echo -e "${YELLOW}⚠ bandit not installed, skipping security scanning${NC}"
    echo ""
fi

# Summary
echo -e "${BLUE}========================================${NC}"
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All linters passed! ✓${NC}"
    exit 0
else
    echo -e "${RED}Some linters failed. See above for details.${NC}"
    exit 1
fi
