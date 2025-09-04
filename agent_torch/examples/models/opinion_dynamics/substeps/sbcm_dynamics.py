import torch
import torch.nn as nn
import torch.nn.functional as F
from agent_torch.core.substep import SubstepTransition


def gumbel_softmax(logits, temperature=0.2):
    """
    Differentiable discrete sampling using Gumbel softmax trick.
    Used for differentiable partner selection in SBCM.
    
    Reference: SINN codebase opinion_dynamics/src/sinn.py lines 50-55
    Original implementation exactly reproduced here for consistency.
    """
    eps = 1e-20
    u = torch.rand(logits.shape, device=logits.device)
    gumbel_noise = -torch.log(-torch.log(u + eps) + eps)
    y = logits + gumbel_noise
    return F.softmax(y / temperature, dim=-1)


class SBCMDynamics(SubstepTransition):
    """
    Stochastic Bounded Confidence Model (SBCM) as a FLAME substep.
    
    Implements the differentiable SBCM from SINN paper:
    - Agents select interaction partners based on opinion distance
    - Closer opinions have higher interaction probability  
    - Uses Gumbel softmax for differentiable partner selection
    - Updates opinions based on selected interactions
    
    Key equation: p_uv = (|x_u - x_v| + ε)^ρ
    where ρ controls the bounded confidence effect
    
    Reference: SINN codebase opinion_dynamics/src/sinn.py lines 171-185
    - Line 172: distance = torch.abs(x_u - vector_x)
    - Line 175: p_uv = (distance + 1e-12).pow(self.rho) 
    - Line 178: tilde_z_ut = self.sampling(p_uv)
    - Line 181-182: rhs_ode = tilde_z_ut * (x_u - vector_x); rhs_ode.sum(-1)
    """
    
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)
        
        # SBCM-specific parameters
        # Reference: SINN codebase opinion_dynamics/src/sinn.py line 95
        # Line 95: self.rho = nn.Parameter(torch.ones(1)) ## Exponent parameter ρ
        if "rho" not in self.learnable_args:
            # Default rho parameter for bounded confidence (SINN line 95)
            self.rho = nn.Parameter(torch.tensor(1.0))
        else:
            self.rho = self.learnable_args["rho"]
            
        # Gumbel softmax temperature for differentiable sampling
        self.temperature = self.fixed_args.get("temperature", 0.1)
        
        # Small epsilon to avoid numerical issues  
        # Reference: SINN codebase opinion_dynamics/src/sinn.py line 175
        # Line 175: p_uv = (distance + 1e-12).pow(self.rho)
        self.eps = self.fixed_args.get("eps", 1e-12)
        
    def compute_interaction_probabilities(self, agent_opinions, neighbor_opinions):
        """
        Compute SBCM interaction probabilities based on opinion distances.
        
        Args:
            agent_opinions: [batch_size, 1] - opinions of focal agents
            neighbor_opinions: [batch_size, num_neighbors] - opinions of potential partners
            
        Returns:
            probabilities: [batch_size, num_neighbors] - interaction probabilities
            
        Reference: SINN codebase opinion_dynamics/src/sinn.py lines 172-175
        - Line 172: distance = torch.abs(x_u - vector_x)
        - Line 175: p_uv = (distance + 1e-12).pow(self.rho)
        """
        # Compute opinion distances: |x_u - x_v| (SINN line 172)
        distances = torch.abs(agent_opinions - neighbor_opinions)
        
        # SBCM probability: p_uv = (distance + ε)^ρ (SINN line 175)
        # Note: In SBCM, closer opinions (smaller distances) should have higher interaction probability
        # So we use negative exponent or invert the relationship
        probabilities = (distances + self.eps).pow(-torch.abs(self.rho))
        
        return probabilities
        
    def differentiable_partner_selection(self, interaction_probs):
        """
        Use Gumbel softmax for differentiable partner selection.
        
        Args:
            interaction_probs: [batch_size, num_neighbors] - interaction probabilities
            
        Returns:
            selection_weights: [batch_size, num_neighbors] - differentiable selection weights
            
        Reference: SINN codebase opinion_dynamics/src/sinn.py lines 107-110, 178
        - Line 108: vec = F.softmax(vec, dim=1)
        - Line 109: logits = gumbel_softmax(vec, 0.1)
        - Line 178: tilde_z_ut = self.sampling(p_uv)
        """
        # Convert to logits for Gumbel softmax
        logits = torch.log(interaction_probs + self.eps)
        
        # Apply Gumbel softmax for differentiable discrete selection (SINN line 178)
        selection_weights = gumbel_softmax(logits, self.temperature)
        
        return selection_weights
        
    def compute_opinion_updates(self, agent_opinions, neighbor_opinions, selection_weights):
        """
        Compute opinion updates based on selected interaction partners.
        
        Args:
            agent_opinions: [batch_size, 1] - current opinions of focal agents
            neighbor_opinions: [batch_size, num_neighbors] - opinions of neighbors  
            selection_weights: [batch_size, num_neighbors] - partner selection weights
            
        Returns:
            opinion_updates: [batch_size, 1] - opinion change for each agent
            
        Reference: SINN codebase opinion_dynamics/src/sinn.py lines 181-182
        - Line 181: rhs_ode = tilde_z_ut * (x_u - vector_x)
        - Line 182: rhs_ode = rhs_ode.sum(-1)
        """
        # Compute opinion differences with selected partners (SINN line 181)
        opinion_diffs = neighbor_opinions - agent_opinions
        
        # Weight by selection probabilities and sum (SINN lines 181-182)
        weighted_influence = (selection_weights * opinion_diffs).sum(dim=-1, keepdim=True)
        
        return weighted_influence
        
    def forward(self, state, action):
        """
        Apply SBCM dynamics to update agent opinions.
        
        Args:
            state: Current simulation state
            action: Action outputs (may contain predicted opinions)
            
        Returns:
            Updated state with new opinion values
        """
        # Extract current opinions from state
        agent_opinions = state["agents"]["citizens"]["opinion"]  # [num_agents, 1]
        
        # Get social network structure
        # Assuming network adjacency or neighbor lists are available in state
        if "network" in state:
            network_data = state["network"]
            neighbor_indices = network_data.get("neighbor_indices", None)
            neighbor_opinions = network_data.get("neighbor_opinions", None)
        else:
            # Fallback: use all other agents as potential neighbors
            neighbor_opinions = agent_opinions.expand(-1, agent_opinions.size(0) - 1)
            neighbor_indices = None
            
        # Compute interaction probabilities based on SBCM
        interaction_probs = self.compute_interaction_probabilities(
            agent_opinions, neighbor_opinions
        )
        
        # Differentiable partner selection using Gumbel softmax
        selection_weights = self.differentiable_partner_selection(interaction_probs)
        
        # Compute opinion updates
        opinion_updates = self.compute_opinion_updates(
            agent_opinions, neighbor_opinions, selection_weights
        )
        
        # Apply opinion updates with learning rate/step size
        step_size = self.fixed_args.get("step_size", 0.1)
        updated_opinions = agent_opinions + step_size * opinion_updates
        
        # Optionally clip opinions to valid range (e.g., [-1, 1] or [0, 1])
        if self.fixed_args.get("clip_opinions", True):
            opinion_range = self.fixed_args.get("opinion_range", [-1.0, 1.0])
            updated_opinions = torch.clamp(updated_opinions, opinion_range[0], opinion_range[1])
        
        # Update state with new opinions
        new_state = state.copy()
        new_state["agents"]["citizens"]["opinion"] = updated_opinions
        
        # Store additional outputs for analysis/debugging
        if self.config.get("store_interaction_data", False):
            new_state["sbcm_interaction_probs"] = interaction_probs
            new_state["sbcm_selection_weights"] = selection_weights
            new_state["sbcm_opinion_updates"] = opinion_updates
            
        return new_state
        
    def get_ode_constraints(self, state, predicted_opinions, neighbor_opinions, selection_weights):
        """
        Compute ODE constraint loss for gradient-based learning.
        This implements the physics-informed loss from SINN.
        
        Args:
            state: Current simulation state
            predicted_opinions: Neural network predicted opinions
            neighbor_opinions: Opinions of interaction partners
            selection_weights: Differentiable partner selection weights
            
        Returns:
            ode_constraint_loss: Scalar loss term
        """
        # Get current opinions
        current_opinions = state["agents"]["citizens"]["opinion"]
        
        # Compute expected opinion change based on SBCM dynamics
        opinion_diffs = neighbor_opinions - current_opinions
        expected_change = (selection_weights * opinion_diffs).sum(dim=-1, keepdim=True)
        
        # Compute actual opinion change from neural network
        actual_change = predicted_opinions - current_opinions
        
        # ODE constraint: predicted change should match SBCM dynamics
        ode_constraint = F.mse_loss(actual_change, expected_change)
        
        return ode_constraint
        
    def get_regularization_loss(self):
        """
        Compute regularization loss on SBCM parameters.
        
        Returns:
            regularization_loss: Scalar loss term
            
        Reference: SINN codebase opinion_dynamics/src/sinn.py line 185
        - Line 185: regularizer = self.beta * torch.zeros(1)
        Note: SBCM in SINN uses no regularization, but we add L2 for stability
        """
        # L2 regularization on rho parameter
        reg_loss = torch.norm(self.rho, p=2)
        
        return reg_loss


class SBCMOpinionDynamicsConfig:
    """
    Configuration helper for SBCM opinion dynamics substep.
    """
    
    @staticmethod
    def get_default_config():
        """
        Get default configuration for SBCM substep.
        """
        return {
            "input_variables": ["agents.citizens.opinion", "network"],
            "output_variables": ["agents.citizens.opinion"],
            "arguments": {
                "learnable": {
                    "rho": torch.tensor(1.0)  # Bounded confidence parameter
                },
                "fixed": {
                    "temperature": 0.1,        # Gumbel softmax temperature
                    "eps": 1e-12,             # Numerical stability epsilon
                    "step_size": 0.1,         # Opinion update step size
                    "clip_opinions": True,     # Whether to clip opinion values
                    "opinion_range": [-1.0, 1.0]  # Valid opinion range
                }
            }
        }