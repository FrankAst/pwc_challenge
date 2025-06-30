# Dependency Management Guide

This project uses pip-tools for dependency locking to ensure reproducible environments.

## Files Structure
- `requirements.in` - Core dependencies (unpinned)
- `requirements.txt` - Locked core dependencies (auto-generated)
- `requirements-dev.in` - Development dependencies (unpinned)
- `requirements-dev.txt` - Locked development dependencies (auto-generated)

## Workflow

### 1. Adding New Dependencies
To add a new dependency:
1. Add it to `requirements.in` (for core deps) or `requirements-dev.in` (for dev deps)
2. Run the compilation command to update the locked file
3. Install the new dependencies

### 2. Updating Dependencies
```bash
# Update all dependencies to latest compatible versions
pip-compile --upgrade requirements.in
pip-compile --upgrade requirements-dev.in

# Update specific package
pip-compile --upgrade-package pandas requirements.in

# Sync your environment with the locked requirements
pip-sync requirements.txt requirements-dev.txt
```

### 3. Installing Dependencies
```bash
# Production environment
pip install -r requirements.txt

# Development environment  
pip install -r requirements.txt -r requirements-dev.txt

# Or use pip-sync for exact environment matching
pip-sync requirements.txt requirements-dev.txt
```

### 4. Regular Maintenance
```bash
# Recompile when requirements.in changes
pip-compile requirements.in
pip-compile requirements-dev.in

# Check for outdated packages
pip list --outdated
```

## Commands Cheat Sheet
```bash
# Install pip-tools
pip install pip-tools

# Compile requirements
pip-compile requirements.in
pip-compile requirements-dev.in

# Upgrade all packages
pip-compile --upgrade requirements.in

# Sync environment (removes unused packages)
pip-sync requirements.txt requirements-dev.txt

# Generate both files at once
pip-compile requirements.in && pip-compile requirements-dev.in
```

## Benefits
- **Reproducibility**: Exact versions locked for all dependencies
- **Security**: Easy to track vulnerable package versions
- **Collaboration**: Team uses identical dependency versions
- **CI/CD**: Consistent builds across environments
- **Flexibility**: Easy updates while maintaining control