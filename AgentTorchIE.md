# AgentTorch Integration with SINN Opinion Dynamics

## Overview

This document outlines the integration plan for combining the Sociologically-Informed Neural Networks (SINN) opinion dynamics model with the AgentTorch FLAME framework. The goal is to replace epidemiological models with end-to-end differentiable opinion dynamics models capable of scaling to millions of agents.

## Background

### FLAME Framework (AgentTorch)
- **Domain-Specific Language**: Composable, differentiable substeps for large-scale simulations
- **Architecture**: Observation-Action-Transition pattern for agent behaviors
- **Scalability**: Tensorized execution supporting millions of agents via archetypes
- **Differentiability**: End-to-end gradient flow enabling gradient-based calibration
- **Deployment**: Decentralized protocols for privacy-preserving real-world integration

### SINN Opinion Dynamics Model
- **Neural Network**: MLPNet for learning complex opinion evolution patterns
- **Opinion Dynamics**: Multiple models (DeGroot, SBCM, BCM, FJ) for social influence
- **Differentiable ODEs**: Automatic differentiation for parameter learning
- **Multi-Loss Training**: Data loss + ODE constraints + regularization terms
- **Archetype Support**: User profile embeddings and attention mechanisms

## Integration Architecture

### 1. Core Substep Integration

#### Opinion Observation Substep
```python
class OpinionObservation(SubstepObservation):
    """
    Observes current agent opinion states, social network neighbors, 
    and temporal context for opinion dynamics modeling.
    """
    def forward(self, state):
        # Extract agent opinion values
        # Get social network neighborhood 
        # Collect temporal features
        # Return observation dict for SINN model
```

#### Opinion Action Substep  
```python
class OpinionAction(SubstepAction):
    """
    Uses SINN neural network to predict opinion updates based on
    social influence and individual characteristics.
    """
    def __init__(self, sinn_model, opinion_dynamics_type):
        self.sinn_net = sinn_model  # MLPNet from SINN
        self.dynamics_type = opinion_dynamics_type  # DeGroot, SBCM, etc.
    
    def forward(self, state, observation):
        # Process through SINN neural network
        # Apply selected opinion dynamics model
        # Return opinion update predictions
```

#### Opinion Transition Substep
```python
class OpinionTransition(SubstepTransition):
    """
    Updates agent opinion states based on SINN model predictions
    while maintaining social network structure.
    """
    def forward(self, state, action):
        # Update agent opinion values
        # Preserve social network topology
        # Track opinion evolution over time
```

### 2. Neural Network Component Mapping

| SINN Component | FLAME Integration | Purpose |
|----------------|-------------------|---------|
| `MLPNet` | `OpinionAction.sinn_net` | Core neural network for opinion prediction |
| `modules.py` archetype handling | AgentTorch archetype system | Scale to millions via representative agents |
| Opinion dynamics models | Environment substeps | Social influence mechanisms |
| Loss functions | FLAME calibration system | Gradient-based parameter learning |
| User profiles | Agent state properties | Individual characteristics |

### 3. Opinion Dynamics Models as Environment Substeps

#### DeGroot Model Integration
```python
class DeGrootDynamics(SubstepTransition):
    """DeGroot consensus model as FLAME substep"""
    def __init__(self, influence_matrix):
        self.M = influence_matrix  # Learnable influence weights
        
    def forward(self, state, action):
        # Implement: x_i(t+1) = Σ M_ij * x_j(t)
```

#### SBCM (Stochastic Bounded Confidence) Integration  
```python
class SBCMDynamics(SubstepTransition):
    """Stochastic Bounded Confidence Model as FLAME substep"""
    def __init__(self, confidence_threshold):
        self.rho = confidence_threshold  # Learnable parameter
        
    def forward(self, state, action):
        # Implement distance-based interaction probability
        # Apply differentiable sampling for partner selection
```

### 4. Data Integration Strategy

#### Agent State Extension
```python
# Extend AgentTorch agent states
agent_state = {
    'age': tensor,           # Existing demographic
    'location': tensor,      # Existing spatial
    'opinion': tensor,       # NEW: Current opinion value  
    'opinion_history': tensor, # NEW: Temporal opinion track
    'profile_embedding': tensor, # NEW: SINN user profiles
    'social_susceptibility': tensor, # NEW: Influence parameters
}
```

#### Network Representation
- Leverage AgentTorch's existing network infrastructure
- Add opinion-specific edge weights for social influence
- Maintain scalability through sparse tensor operations

#### Temporal Integration
- Map SINN's continuous-time ODEs to FLAME's discrete time steps
- Preserve differentiability through reparameterization
- Support both synchronous and asynchronous opinion updates

### 5. Training and Calibration Integration

#### Multi-Loss Training
```python
class SINNFLAMELoss:
    def __init__(self, alpha, beta):
        self.alpha = alpha  # ODE constraint weight
        self.beta = beta   # Regularization weight
        
    def compute_loss(self, predictions, ground_truth, ode_constraints, regularization):
        data_loss = F.cross_entropy(predictions, ground_truth)
        ode_loss = self.alpha * ode_constraints.mean()
        reg_loss = self.beta * regularization.mean()
        return data_loss + ode_loss + reg_loss
```

#### Gradient-Based Calibration
- Use FLAME's automatic differentiation for end-to-end learning
- Integrate SINN's ODE constraint gradients with FLAME's simulation gradients  
- Support heterogeneous data streams (social media, surveys, behavioral data)

### 6. Archetype System Enhancement

#### Opinion-Aware Archetypes
```python
class OpinionArchetype:
    """Enhanced archetype system for opinion dynamics"""
    def __init__(self, demographic_features, opinion_features, social_features):
        self.demographics = demographic_features  # Age, gender, location
        self.opinions = opinion_features         # Political views, attitudes  
        self.social = social_features           # Network position, influence
        
    def sample_behavior(self, context):
        # Use SINN neural network for archetype-level opinion prediction
        # Sample individual behaviors from archetype distributions
```

## Implementation Roadmap

### Phase 1: Core Infrastructure (Weeks 1-2)
- [ ] Create opinion dynamics substeps (Observation, Action, Transition)
- [ ] Integrate SINN MLPNet into AgentTorch action framework
- [ ] Extend agent state representation for opinion tracking
- [ ] Basic DeGroot model integration as proof of concept

### Phase 2: Advanced Dynamics (Weeks 3-4)
- [ ] Implement SBCM, BCM, and FJ models as FLAME substeps
- [ ] Add differentiable ODE constraint computation
- [ ] Integrate multi-loss training with FLAME calibration
- [ ] Test scalability with archetype-based sampling

### Phase 3: Real-World Integration (Weeks 5-6)
- [ ] Add support for heterogeneous opinion data sources
- [ ] Implement privacy-preserving protocols for opinion tracking
- [ ] Create validation framework against real opinion dynamics datasets
- [ ] Performance optimization for million-agent simulations

### Phase 4: Advanced Features (Weeks 7-8)
- [ ] Multi-modal opinion dynamics (text, behavior, networks)
- [ ] Temporal opinion evolution with memory effects
- [ ] Intervention modeling (media campaigns, social policies)
- [ ] Distributed deployment for real-world opinion monitoring

## Technical Challenges and Solutions

### Challenge 1: Scale vs. Expressiveness
**Problem**: SINN's individual-level modeling vs. FLAME's archetype-based scaling

**Solution**: 
- Use archetype-level SINN models with individual sampling
- Implement hierarchical opinion modeling (individual → archetype → population)
- Leverage FLAME's tensorized operations for efficient neural network inference

### Challenge 2: Discrete vs. Continuous Dynamics
**Problem**: SINN's continuous ODEs vs. FLAME's discrete time steps

**Solution**:
- Reparameterize discrete opinion choices for differentiability
- Use Euler integration for ODE approximation in discrete steps
- Implement adaptive time-stepping for numerical stability

### Challenge 3: Data Integration
**Problem**: Combining diverse opinion data sources with varying quality and privacy constraints

**Solution**:
- Use FLAME's federated calibration for distributed opinion data
- Implement uncertainty-aware opinion modeling
- Add privacy-preserving aggregation for sensitive opinion tracking

## Expected Outcomes

### Research Contributions
1. **Scalable Opinion Dynamics**: First framework capable of neural opinion modeling at population scale
2. **End-to-End Differentiability**: Gradient-based learning for complex social influence patterns
3. **Real-World Deployment**: Privacy-preserving opinion dynamics monitoring and intervention

### Applications
1. **Political Science**: Large-scale election prediction and voter behavior analysis
2. **Social Policy**: Understanding public opinion evolution and policy intervention effects
3. **Market Research**: Consumer sentiment tracking and influence campaign optimization
4. **Public Health**: Health behavior change and misinformation propagation modeling

## References

### Core Papers
- **FLAME**: Chopra et al. "Large Population Models: Technical Contributions and Open Problems" (2024)
- **SINN**: Okawa & Iwata "Predicting Opinion Dynamics via Sociologically-Informed Neural Networks" KDD 2022
- **AgentTorch**: Chopra et al. "A framework for learning in agent-based models" AAMAS 2024

### Opinion Dynamics Submodule
- Repository: `opinion_dynamics/` (https://github.com/mayaokawa/opinion_dynamics.git)
- Main implementation: `opinion_dynamics/src/sinn.py`
- Neural network: `opinion_dynamics/modules.py`

---

*Document Version: 1.0*  
*Last Updated: 2025-01-04*  
*Contributors: Claude Code Integration Analysis*