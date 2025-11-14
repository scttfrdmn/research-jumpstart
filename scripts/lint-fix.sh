#!/bin/bash
# Auto-fix linting issues where possible
# This script attempts to automatically fix linting issues

set -e

# Colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Auto-fixing Linting Issues${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
LINTING_DIR="$PROJECT_ROOT/.linting"

cd "$PROJECT_ROOT"

# 1. Fix Python with Ruff
if command -v ruff &> /dev/null; then
    echo -e "${YELLOW}Fixing Python issues with Ruff...${NC}"
    ruff check --fix --config "$LINTING_DIR/ruff.toml" .
    ruff format --config "$LINTING_DIR/ruff.toml" .
    echo -e "${GREEN}✓ Python files formatted${NC}"
    echo ""
else
    echo -e "${YELLOW}⚠ Ruff not installed${NC}"
fi

# 2. Fix Markdown
if command -v markdownlint &> /dev/null; then
    echo -e "${YELLOW}Fixing Markdown issues...${NC}"
    markdownlint --fix --config "$LINTING_DIR/.markdownlint.json" '**/*.md' --ignore node_modules --ignore .venv || true
    echo -e "${GREEN}✓ Markdown files fixed${NC}"
    echo ""
else
    echo -e "${YELLOW}⚠ markdownlint not installed${NC}"
fi

echo -e "${GREEN}Auto-fix complete!${NC}"
echo -e "${YELLOW}Note: Some issues may require manual fixes.${NC}"
echo -e "${YELLOW}Run './scripts/lint.sh' to check remaining issues.${NC}"
