# Contributing to NBA Injury Prediction

Thank you for your interest in contributing! This document provides guidelines for contributions.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Python version and OS
- Error messages/stack traces

### Suggesting Enhancements

For feature requests:
- Describe the enhancement
- Explain the use case
- Provide examples if applicable

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (if available)
5. Commit with clear messages
6. Push to your fork
7. Open a Pull Request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/nba-injury-prediction.git
cd nba-injury-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (if needed)
pip install pytest black flake8
```

## Code Style

- Follow PEP 8 style guide
- Use meaningful variable names
- Add docstrings to functions/classes
- Keep functions focused and modular
- Comment complex logic

### Running Linters

```bash
# Format code
black src/ scripts/ app.py

# Check style
flake8 src/ scripts/ app.py --max-line-length=100
```

## Testing

Currently, the project uses manual testing with data files. If adding tests:

```bash
# Run tests (when available)
pytest tests/
```

## Documentation

- Update README.md for significant changes
- Add docstrings to new functions
- Update config.yaml comments if changing parameters
- Document new features in QUICKSTART.md

## Project Structure

When adding new features, maintain the structure:

```
src/          # Core modules
scripts/      # Executable scripts
config/       # Configuration files
notebooks/    # Analysis notebooks
app.py        # Web interface
```

## Commit Messages

Use clear commit messages:
- `feat: Add SHAP waterfall plots`
- `fix: Handle missing salary data`
- `docs: Update installation instructions`
- `refactor: Simplify feature engineering`

## Areas for Contribution

### High Priority
- [ ] Add unit tests for data_loader
- [ ] Add unit tests for preprocessing
- [ ] Implement data validation checks
- [ ] Add more SHAP visualizations
- [ ] Optimize hyperparameters for latest data

### Medium Priority
- [ ] Add CI/CD pipeline
- [ ] Create Docker container
- [ ] Add more feature engineering
- [ ] Improve Streamlit UI
- [ ] Add model versioning

### Low Priority
- [ ] Add alternative models (Random Forest, Neural Networks)
- [ ] Create API endpoint
- [ ] Add player comparison tool
- [ ] Historical trend analysis

## Questions?

Open an issue or reach out to maintainers.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
