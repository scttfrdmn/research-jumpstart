# Project Assets

This directory contains visual assets and supplementary materials for the Climate Model Ensemble Analysis project.

## Contents

### architecture-diagram.svg
System architecture diagram showing:
- Studio Lab version (simple architecture)
- Unified Studio version (production architecture with S3, EMR, Bedrock)
- Data flow and component relationships

**Status**: 🚧 To be created
**Tools**: draw.io, Lucidchart, or similar
**Format**: SVG (scalable) with PNG export for documentation

---

### sample-outputs/
Example outputs from running the analysis:
- `ensemble-timeseries.png` - Multi-model temperature projection with uncertainty
- `model-agreement.png` - Box plots showing model spread by decade
- `regional-map.png` - Map showing analysis region
- `summary-statistics.csv` - Example numerical results

**Status**: 🚧 Generated when notebook is executed

---

### cost-calculator.xlsx
Interactive spreadsheet for estimating AWS costs based on:
- Number of models
- Number of variables
- Time range
- Region size
- Frequency of analysis

**Status**: 🚧 To be created
**Template**: See `/tools/cost-calculator-template.xlsx`

---

## Creating the Architecture Diagram

### Recommended Tool: draw.io (free)

1. Visit https://app.diagrams.net/
2. Start with blank diagram
3. Use these components:

**Studio Lab Architecture**:
```
- Rectangle: "SageMaker Studio Lab (Free Tier)"
- Inside: Smaller rectangles for:
  - Jupyter Environment
  - Analysis Workflow (numbered steps)
  - Generated data
- Arrows showing data flow
```

**Unified Studio Architecture**:
```
- Multiple connected rectangles:
  - JupyterLab Environment
  - S3 (CMIP6 data)
  - Processing Layer (xarray/dask)
  - Optional EMR cluster
  - Bedrock (AI)
  - S3 (results)
- Arrows with labels showing:
  - Data access (no egress charges)
  - Processing flow
  - API calls
```

### Style Guidelines

- **Colors**:
  - AWS services: Orange (#FF9900)
  - Compute: Blue (#0066CC)
  - Storage: Green (#00CC66)
  - AI services: Purple (#9933CC)

- **Labels**:
  - Clear service names
  - Cost indicators (e.g., "$0 egress")
  - Performance notes (e.g., "parallel processing")

- **Export**:
  - Primary: SVG (for scaling)
  - Secondary: PNG at 300 DPI (for documentation)

### Text-Based Alternative

If you prefer text-based diagrams, see `architecture-diagram.txt` in this directory for ASCII art version that can be rendered in documentation.

---

## Sample Output Guidelines

When adding sample outputs:

1. **Use realistic data**: Run the notebook to generate actual outputs
2. **Include captions**: Each image should have descriptive filename
3. **Optimize file size**:
   - PNG for plots (< 500KB each)
   - Use `plt.savefig(..., dpi=150, bbox_inches='tight')`
4. **Document parameters**: Include metadata about what was analyzed

---

## Cost Calculator Structure

Excel/Google Sheets columns:
- Number of models (input)
- Variables (input)
- Time range (input)
- Region size (input)
- Analyses per month (input)
- → Compute hours (calculated)
- → Storage GB (calculated)
- → Bedrock tokens (calculated)
- → Total cost (calculated)

Include multiple scenarios:
- Light usage (3 models, 1 scenario)
- Medium usage (10 models, 2 scenarios)
- Heavy usage (20+ models, multiple scenarios)

---

## Contributing Assets

If you create improved versions of these assets:

1. Follow naming conventions
2. Include source files (e.g., .drawio for diagrams)
3. Provide both web-optimized and high-res versions
4. Submit PR with description of improvements

See main [CONTRIBUTING.md](../../../../CONTRIBUTING.md) for guidelines.

---

*Last updated: 2025-01-09*
