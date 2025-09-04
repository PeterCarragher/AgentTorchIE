# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AgentTorch is an open-source platform for building Large Population Models (LPMs) - differentiable agent-based simulations that can model millions of interacting entities. It's designed as "PyTorch for large-scale agent-based simulations" with focus on scalability, differentiability, composition with neural networks/LLMs, and generalization across domains.

## Development Commands

### Installation and Setup
```bash
# Install in development mode
pip install -e .

# Install development dependencies
pip install -r development.txt

# Set up pre-commit hooks
pre-commit install --config pre-commit.yaml
```

### Testing
```bash
# Run all tests
pytest -vvv tests/

# Run specific test file
pytest -vvv tests/test_behavior.py

# Run tests with coverage
pytest --cov=agent_torch tests/
```

### Code Formatting and Quality
```bash
# Format code with black
black agent_torch/ tests/

# Run pre-commit hooks manually
pre-commit run --all-files
```

### Documentation
```bash
# Serve documentation locally
mkdocs serve

# Build documentation
mkdocs build
```

### Build and Release
```bash
# Build package
python -m build

# Upload to PyPI (maintainers only)
./upload_to_pypi.sh
```

## Architecture Overview

AgentTorch follows a modular architecture with clear separation of concerns:

### Core Components

1. **Runner** (`agent_torch/core/runner.py`): Main execution engine that manages simulation episodes and steps
2. **Controller** (`agent_torch/core/controller.py`): Handles substep execution (observe-act-transition loop)
3. **Executor** (`agent_torch/core/executor.py`): High-level interface for running simulations with data loaders
4. **Initializer** (`agent_torch/core/initializer.py`): Creates simulation state from configuration and registry
5. **Registry** (`agent_torch/core/registry.py`): Stores references to substep implementations and helper functions

### Key Abstractions

- **Substeps**: Basic simulation units with three phases:
  - `SubstepObservation`: Observes current state
  - `SubstepAction`/`SubstepPolicy`: Decides actions based on observations  
  - `SubstepTransition`: Updates state based on actions

- **Agents**: Simulation entities defined in YAML configuration
- **Environment**: Contains agent states, networks, and global variables
- **State**: Complete simulation state including agents, environment, and metadata

### Configuration System

Simulations are configured via YAML files that define:
- Agent types and properties
- Substep definitions and execution order
- Environment variables and networks
- Simulation metadata (episodes, steps, etc.)

### LLM Integration

AgentTorch supports LLM-powered agent behavior through:
- **Archetypes** (`agent_torch/core/llm/archetype.py`): LLM-based agent personas
- **Behavior** (`agent_torch/core/llm/behavior.py`): Behavioral sampling from LLMs
- **Backend** (`agent_torch/core/llm/backend.py`): LLM provider abstractions

## Project Structure

```
agent_torch/
├── core/                    # Core simulation engine
│   ├── analyzer/           # Simulation analysis tools
│   ├── distributions/      # Probability distributions for agents
│   ├── helpers/           # Utility functions for state management
│   └── llm/               # LLM integration components
├── config/                # Configuration management
├── data/                  # Data loading utilities
├── examples/              # Example model implementations
├── models/               # Complete model implementations
│   ├── covid/           # Disease spread simulation
│   ├── macro_economics/ # Economic modeling
│   └── predator_prey/   # Ecosystem simulation
├── populations/         # Population data for different regions
└── models/              # Domain-specific models

docs/                    # Documentation and tutorials
tests/                  # Unit tests
```

## Working with Models

### Creating New Models

1. Define substeps by inheriting from `SubstepObservation`, `SubstepAction`, and `SubstepTransition`
2. Create YAML configuration defining agents, environment, and substeps
3. Register substeps in a registry dictionary
4. Use `Executor` or `Runner` to execute simulations

### Example Model Structure
```
models/your_model/
├── substeps/           # Substep implementations
├── yamls/             # Configuration files
├── data/              # Model-specific data
├── simulator.py       # Main simulation logic
└── __init__.py        # Registry definition
```

## Distributed Execution

AgentTorch supports distributed simulation execution:
- Use `distributed_runner.py` for multi-GPU/multi-node execution
- Configure via `run_movement_sim_distributed.py` examples
- Leverage Dask for distributed data processing

## Key Dependencies

- **PyTorch**: Core tensor operations and GPU acceleration
- **OmegaConf**: Configuration management
- **NetworkX**: Graph operations for agent networks
- **LangChain**: LLM integration
- **DSPy**: Structured LLM programming
- **Dask**: Distributed computing

## Development Guidelines

1. Follow the existing code patterns when creating new substeps or models
2. Use type hints and docstrings for new functions
3. Add tests for new functionality in the `tests/` directory
4. Update documentation when adding new features
5. Use the existing population data formats in `populations/` directory
6. Follow the YAML configuration patterns from existing models

## Common Development Tasks

- **Adding new substeps**: Create classes inheriting from base substep classes
- **Creating populations**: Use the census data processing utilities in `agent_torch/data/census/`
- **LLM integration**: Use the archetype and behavior system for LLM-powered agents  
- **Vectorized operations**: Leverage PyTorch's vectorization for performance
- **Calibration**: Use the built-in parameter calibration system for model tuning