# Changelog

All notable changes to the Learning Analytics project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-11-13

### Added
- Initial release of Learning Analytics project
- **Tier 0**: Single-course dropout prediction (60-90 min, Colab/Studio Lab)
  - LSTM-based dropout prediction model
  - Simulated MOOC interaction data
  - Feature engineering pipeline
  - Intervention recommendation system
  - Complete documentation

- **Tier 1**: Multi-institution learning outcomes ensemble (4-8 hours, Studio Lab only)
  - Cross-institutional data harmonization
  - Ensemble models across 5 institutions
  - Learning pathway prediction
  - Intervention timing optimization
  - Persistent storage for datasets and models
  - Complete notebook workflow

- **Documentation**:
  - Comprehensive main README with problem statement
  - Tier 0 and Tier 1 specific READMEs
  - Architecture diagrams and workflow descriptions
  - Cost estimation and optimization guidelines
  - Transition pathway from free to production
  - Data privacy and FERPA compliance notes

- **Project Structure**:
  - Organized tier-based structure
  - Separate directories for notebooks, source code, data, and models
  - CloudFormation templates for production deployment
  - Workshop materials structure

### Design Decisions
- Used LSTM for sequence modeling (captures temporal patterns in learning)
- Simulated data for Tier 0 (enables immediate start without data access)
- Studio Lab requirement for Tier 1 (essential for persistent storage)
- Emphasis on actionable insights (intervention recommendations)
- Cross-institutional focus (enables generalizability research)

### Known Limitations
- Tier 0 uses simulated data (not real MOOC logs)
- Tier 1 requires Studio Lab approval (can take days)
- No real-time prediction API in free tiers
- Limited to 15GB storage in Studio Lab

## [Unreleased]

### Planned Features
- v1.1.0: Workshop materials (slides, exercises, solutions)
- v1.2.0: Fairness and bias analysis toolkit
- v1.3.0: Explainable AI integration (SHAP, LIME)
- v2.0.0: Real MOOC dataset integration
- v2.1.0: Multi-semester longitudinal analysis
- v2.2.0: Causal inference extensions

### Under Consideration
- Integration with popular LMS platforms (Canvas, Blackboard)
- Real-time prediction dashboard
- A/B testing framework for interventions
- Federated learning for privacy-preserving multi-institution analysis
- Mobile app for student engagement tracking

---

## Version Naming

- **Major version** (X.0.0): Breaking changes, new tiers, major architecture changes
- **Minor version** (1.X.0): New features, datasets, analysis methods
- **Patch version** (1.0.X): Bug fixes, documentation updates, minor improvements

---

*Last updated: 2025-11-13*
