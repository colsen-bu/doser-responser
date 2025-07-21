# Doser Responser - AI Coding Agent Instructions

## Project Overview
Doser Responser is a Dash-based web application for analyzing dose-response relationships in scientific experiments. The application allows users to upload CSV data, visualize 96-well plate layouts, exclude/include data points interactively, and fit multiple mathematical models to dose-response curves.

## Architecture

### Core Structure
- **Entry Point**: `wsgi.py` (production) and `dash_app/app.py` (development)
- **Main Application**: Single-page app in `dash_app/index.py` (~1500 lines)
- **Theme**: Uses `dash-bootstrap-components` with VAPOR theme consistently

### Key Design Patterns

#### Data Flow Architecture
The app uses Dash's client-side data stores for state management:
```python
dcc.Store(id='shared-data', storage_type='memory'),          # Raw CSV data
dcc.Store(id='excluded-wells', storage_type='memory'),       # User exclusions
dcc.Store(id='curve-fit-data', storage_type='memory'),       # Fitted model results
dcc.Store(id='calculation-state', storage_type='memory'),    # UI state
```

#### Interactive Well Selection Pattern
Wells use pattern-matching callbacks with structured IDs:
```python
Input({'type': 'well-cell', 'well': ALL, 'experiment': ALL}, 'n_clicks')
```
Each well cell has ID format: `{'type': 'well-cell', 'well': 'A1', 'experiment': 'exp1'}`

#### Mathematical Model Architecture
Four dose-response models are implemented as pure functions:
- `hill_equation()` - 4-parameter logistic (most common)
- `three_param_logistic()` - 3-parameter (bottom=0)
- `five_param_logistic()` - 5-parameter with asymmetry
- `exponential_model()` - Simple exponential

Models are fitted using `scipy.optimize.curve_fit` with careful parameter bounds and initial guesses.

## Critical Workflows

### Development Setup
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
python dash_app/app.py     # Dev server on http://localhost:8050
```

### Docker Development
```bash
docker build -t doser-responser .
docker run -p 8050:8050 doser-responser
```

### Production Deployment
- **CI/CD**: Woodpecker CI/CD with Docker containerization
- **Runtime**: Uses `gunicorn` with `wsgi.py` as entry point
- **Reverse Proxy**: Nginx Proxy Manager handles routing and SSL
- **Container**: Docker with socket access for container management
- Import order is critical: `wsgi.py` must import `dash_app.index` to register callbacks

### Deployment Pipeline Workflow
1. **Push to main**: Triggers Woodpecker CI/CD pipeline
2. **Build & Test**: Validates imports and dependencies
3. **Docker Build**: Creates container with multi-stage optimization
4. **Deploy**: Pulls latest image and restarts container on production server
5. **Health Check**: Validates application startup via `/_dash-layout` endpoint

### Required Woodpecker Secrets
- `docker_registry`, `docker_username`, `docker_password` - Container registry access
- `webhook_url` - Optional deployment notifications

### Data Format Requirements
CSV must contain these exact columns:
- `Well` (e.g., "A1", "B12")
- `Dose_uM` (numeric, supports 0 for control)
- `Response_Metric` (numeric measurement)
- `Treatment` (string identifier)
- `Experiment_ID` (string, for multi-experiment files)

## Project-Specific Conventions

### Callback Organization
- All callbacks in `index.py` use `@callback` decorator (not `@app.callback`)
- Complex callbacks handle multiple triggers using `dash.callback_context`
- State management flows: Upload → Store → Visualization → User Interaction → Recalculation

### Zero-Dose Handling
Special log-scale visualization pattern for dose=0:
```python
zero_position = min_nonzero / 10  # Place zero one tick below minimum
```
This allows log-scale plotting while preserving zero-dose controls.

### Parameter Slider Pattern
Dynamic sliders created with structured IDs:
```python
id={'type': 'param-slider', 'param': param_name}
```
Values automatically update curve fitting when changed.

### Error Handling
- File upload: Graceful CSV parsing with column validation
- Curve fitting: Returns success/failure status with error messages
- Model comparison: AIC/BIC metrics for automatic best-model selection

## Critical Files
- `dash_app/index.py` - Single monolithic file containing all UI and logic
- `dash_app/app.py` - Minimal Dash app configuration with VAPOR theme
- `wsgi.py` - Production entry point that imports index.py
- `requirements.txt` - Pinned versions, especially numpy<2.0.0 compatibility
- `Dockerfile` - Multi-stage build for production deployment
- `.woodpecker.yml` - CI/CD pipeline configuration
- `docker-compose.yml` - Local development with Docker

## Common Gotchas
- Import `dash_app.index` in `wsgi.py` is required to register callbacks
- Import `index` in `app.py` when running development server to register layout
- Parameter sliders need manual parameter extraction from callback context
- Pattern-matching callbacks require careful JSON parsing from `prop_id`
- Zero doses need special handling in log-scale visualizations
- Model fitting can fail silently - always check `success` flag in results
- Docker health checks use `/_dash-layout` endpoint, not root path
- Woodpecker deployment requires proper secret configuration for registry access
