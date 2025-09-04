"""
Unit tests for SBCM dynamics implementation.

Tests individual components of the SBCM substep in isolation.
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np
from unittest.mock import Mock, patch

# Import the SBCM implementation
from agent_torch.examples.models.opinion_dynamics.substeps.sbcm_dynamics import (
    SBCMDynamics, 
    SBCMOpinionDynamicsConfig,
    gumbel_softmax
)


class TestGumbelSoftmax:
    """Test suite for Gumbel softmax function."""
    
    def test_gumbel_softmax_basic_functionality(self):
        """Test that Gumbel softmax produces valid probability distributions."""
        # Setup
        logits = torch.tensor([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]])
        temperature = 0.5
        
        # Execute
        result = gumbel_softmax(logits, temperature)
        
        # Verify
        assert result.shape == logits.shape, "Output shape should match input"
        assert torch.allclose(result.sum(dim=1), torch.ones(2), atol=1e-5), "Probabilities should sum to 1"
        assert torch.all(result >= 0), "All probabilities should be non-negative"
        assert torch.all(result <= 1), "All probabilities should be <= 1"
        
    def test_gumbel_softmax_temperature_effect(self):
        """Test that temperature affects the sharpness of distribution."""
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        
        # High temperature (soft distribution)
        soft_result = gumbel_softmax(logits, temperature=10.0)
        
        # Low temperature (sharp distribution)  
        sharp_result = gumbel_softmax(logits, temperature=0.1)
        
        # Sharp distribution should have lower entropy
        soft_entropy = -(soft_result * torch.log(soft_result + 1e-10)).sum()
        sharp_entropy = -(sharp_result * torch.log(sharp_result + 1e-10)).sum()
        
        assert sharp_entropy < soft_entropy, "Lower temperature should produce sharper distribution"
        
    def test_gumbel_softmax_gradient_flow(self):
        """Test that gradients flow through Gumbel softmax."""
        logits = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        
        result = gumbel_softmax(logits, temperature=1.0)
        loss = result.sum()
        loss.backward()
        
        assert logits.grad is not None, "Gradients should flow to input logits"
        assert not torch.allclose(logits.grad, torch.zeros_like(logits.grad)), "Gradients should be non-zero"


class TestSBCMDynamics:
    """Test suite for SBCM dynamics implementation."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for SBCM."""
        return {
            "simulation_metadata": {"calibration": True},
            "store_interaction_data": True
        }
    
    @pytest.fixture
    def sbcm_arguments(self):
        """Create default arguments for SBCM."""
        return {
            "learnable": {"rho": torch.tensor(1.0)},
            "fixed": {
                "temperature": 0.1,
                "eps": 1e-12,
                "step_size": 0.1,
                "clip_opinions": True,
                "opinion_range": [-1.0, 1.0]
            }
        }
    
    @pytest.fixture
    def sbcm_model(self, mock_config, sbcm_arguments):
        """Create SBCM model instance."""
        input_vars = ["agents.citizens.opinion", "network"]
        output_vars = ["agents.citizens.opinion"]
        
        return SBCMDynamics(
            config=mock_config,
            input_variables=input_vars,
            output_variables=output_vars,
            arguments=sbcm_arguments
        )
    
    def test_sbcm_initialization(self, sbcm_model):
        """Test SBCM model initialization."""
        assert hasattr(sbcm_model, 'rho'), "Should have rho parameter"
        assert isinstance(sbcm_model.rho, torch.nn.Parameter), "Rho should be learnable parameter"
        assert sbcm_model.temperature == 0.1, "Temperature should be set correctly"
        assert sbcm_model.eps == 1e-12, "Epsilon should be set correctly"
        
    def test_compute_interaction_probabilities(self, sbcm_model):
        """Test SBCM interaction probability computation."""
        # Setup test data
        agent_opinions = torch.tensor([[0.5], [0.0], [-0.5]])  # [3, 1]
        neighbor_opinions = torch.tensor([
            [0.4, 0.6, 0.2],  # Agent 1's neighbors
            [0.1, -0.1, 0.3], # Agent 2's neighbors  
            [-0.4, -0.6, -0.2] # Agent 3's neighbors
        ])  # [3, 3]
        
        # Execute
        probs = sbcm_model.compute_interaction_probabilities(agent_opinions, neighbor_opinions)
        
        # Verify
        assert probs.shape == (3, 3), "Output shape should match neighbor dimensions"
        assert torch.all(probs > 0), "All probabilities should be positive"
        
        # Test that closer opinions have higher probabilities
        # Agent 1 (opinion=0.5) should have highest prob with neighbor 1 (opinion=0.4)
        agent_1_probs = probs[0]
        closest_neighbor_idx = torch.argmin(torch.abs(agent_opinions[0] - neighbor_opinions[0]))
        assert torch.argmax(agent_1_probs) == closest_neighbor_idx, "Closest neighbor should have highest probability"
        
    def test_differentiable_partner_selection(self, sbcm_model):
        """Test differentiable partner selection mechanism."""
        # Setup
        interaction_probs = torch.tensor([
            [0.8, 0.3, 0.1],  # Clear preference for first partner
            [0.2, 0.2, 0.2],  # Equal preferences
            [0.1, 0.5, 0.9]   # Clear preference for third partner
        ])
        
        # Execute
        selection_weights = sbcm_model.differentiable_partner_selection(interaction_probs)
        
        # Verify
        assert selection_weights.shape == interaction_probs.shape, "Shape should be preserved"
        assert torch.allclose(selection_weights.sum(dim=1), torch.ones(3), atol=1e-5), "Weights should sum to 1"
        assert torch.all(selection_weights >= 0), "All weights should be non-negative"
        
        # Test that higher probabilities get higher weights (approximately)
        for i in range(len(interaction_probs)):
            max_prob_idx = torch.argmax(interaction_probs[i])
            max_weight_idx = torch.argmax(selection_weights[i])
            # Allow some stochasticity in Gumbel softmax
            assert max_prob_idx == max_weight_idx or selection_weights[i, max_prob_idx] > 0.2, \
                "Highest probability should generally get highest weight"
        
    def test_compute_opinion_updates(self, sbcm_model):
        """Test opinion update computation."""
        # Setup
        agent_opinions = torch.tensor([[0.0], [0.5], [-0.5]])  # [3, 1]
        neighbor_opinions = torch.tensor([
            [0.2, 0.4, -0.2],  # Positive influence on agent 1
            [0.3, 0.7, 0.1],   # Mixed influence on agent 2
            [-0.3, -0.7, -0.1] # Negative influence on agent 3
        ])  # [3, 3]
        
        # Equal selection weights for simplicity
        selection_weights = torch.ones(3, 3) / 3.0
        
        # Execute
        updates = sbcm_model.compute_opinion_updates(agent_opinions, neighbor_opinions, selection_weights)
        
        # Verify
        assert updates.shape == (3, 1), "Output shape should match agent opinions"
        
        # Test update direction makes sense
        # Agent 1 (opinion=0.0) with positive neighbors should have positive update
        expected_update_1 = (neighbor_opinions[0] - agent_opinions[0]).mean()
        assert torch.allclose(updates[0], expected_update_1, atol=1e-6), "Update should match expected influence"
        
    def test_forward_pass_basic(self, sbcm_model):
        """Test basic forward pass functionality."""
        # Setup mock state
        state = {
            "agents": {
                "citizens": {
                    "opinion": torch.tensor([[0.1], [0.3], [-0.2], [0.0]])  # 4 agents
                }
            },
            "network": {
                "neighbor_opinions": torch.tensor([
                    [0.2, 0.4, 0.0],   # Agent 1's neighbors  
                    [0.1, 0.5, 0.2],   # Agent 2's neighbors
                    [-0.1, -0.3, 0.1], # Agent 3's neighbors
                    [-0.1, 0.1, 0.2]   # Agent 4's neighbors
                ])
            }
        }
        
        action = {}  # Empty action for this test
        
        # Execute
        new_state = sbcm_model.forward(state, action)
        
        # Verify
        assert "agents" in new_state, "New state should contain agents"
        assert "citizens" in new_state["agents"], "Should contain citizens"
        assert "opinion" in new_state["agents"]["citizens"], "Should contain opinions"
        
        new_opinions = new_state["agents"]["citizens"]["opinion"]
        old_opinions = state["agents"]["citizens"]["opinion"]
        
        assert new_opinions.shape == old_opinions.shape, "Opinion shape should be preserved"
        assert not torch.allclose(new_opinions, old_opinions, atol=1e-6), "Opinions should have changed"
        
        # Test opinion clipping
        assert torch.all(new_opinions >= -1.0), "Opinions should be >= -1"
        assert torch.all(new_opinions <= 1.0), "Opinions should be <= 1"
        
    def test_ode_constraints(self, sbcm_model):
        """Test ODE constraint computation."""
        # Setup
        state = {
            "agents": {
                "citizens": {
                    "opinion": torch.tensor([[0.0], [0.5]])
                }
            }
        }
        
        predicted_opinions = torch.tensor([[0.1], [0.4]])  # Predicted by neural network
        neighbor_opinions = torch.tensor([[0.2, -0.1], [0.3, 0.6]])
        selection_weights = torch.tensor([[0.7, 0.3], [0.4, 0.6]])
        
        # Execute
        ode_loss = sbcm_model.get_ode_constraints(
            state, predicted_opinions, neighbor_opinions, selection_weights
        )
        
        # Verify
        assert isinstance(ode_loss, torch.Tensor), "Should return tensor"
        assert ode_loss.dim() == 0, "Should return scalar loss"
        assert ode_loss >= 0, "Loss should be non-negative"
        
    def test_regularization_loss(self, sbcm_model):
        """Test regularization loss computation."""
        # Execute
        reg_loss = sbcm_model.get_regularization_loss()
        
        # Verify
        assert isinstance(reg_loss, torch.Tensor), "Should return tensor"
        assert reg_loss.dim() == 0, "Should return scalar"
        assert reg_loss >= 0, "Regularization loss should be non-negative"
        
        # Test that it depends on rho parameter
        original_rho = sbcm_model.rho.data.clone()
        sbcm_model.rho.data *= 2  # Double the parameter
        
        new_reg_loss = sbcm_model.get_regularization_loss()
        assert new_reg_loss > reg_loss, "Larger parameters should have larger regularization"
        
        # Restore original value
        sbcm_model.rho.data = original_rho
        
    def test_gradient_flow(self, sbcm_model):
        """Test that gradients flow through the model."""
        # Setup
        state = {
            "agents": {
                "citizens": {
                    "opinion": torch.tensor([[0.1], [0.3]], requires_grad=True)
                }
            },
            "network": {
                "neighbor_opinions": torch.tensor([[0.2, 0.0], [0.1, 0.5]])
            }
        }
        
        action = {}
        
        # Execute forward pass
        new_state = sbcm_model.forward(state, action)
        
        # Compute loss and backpropagate
        loss = new_state["agents"]["citizens"]["opinion"].sum()
        loss.backward()
        
        # Verify gradients
        assert sbcm_model.rho.grad is not None, "Rho parameter should have gradients"
        assert not torch.allclose(sbcm_model.rho.grad, torch.zeros_like(sbcm_model.rho.grad)), \
            "Gradients should be non-zero"


class TestSBCMConfig:
    """Test suite for SBCM configuration helper."""
    
    def test_default_config_structure(self):
        """Test that default config has required structure."""
        config = SBCMOpinionDynamicsConfig.get_default_config()
        
        # Verify required keys
        assert "input_variables" in config, "Should have input_variables"
        assert "output_variables" in config, "Should have output_variables"
        assert "arguments" in config, "Should have arguments"
        
        # Verify arguments structure
        args = config["arguments"]
        assert "learnable" in args, "Should have learnable parameters"
        assert "fixed" in args, "Should have fixed parameters"
        
        # Verify specific parameters
        assert "rho" in args["learnable"], "Should have rho parameter"
        assert "temperature" in args["fixed"], "Should have temperature parameter"
        assert "step_size" in args["fixed"], "Should have step_size parameter"
        
    def test_default_config_values(self):
        """Test that default config values are reasonable."""
        config = SBCMOpinionDynamicsConfig.get_default_config()
        args = config["arguments"]
        
        # Test learnable parameters
        assert args["learnable"]["rho"].item() == 1.0, "Default rho should be 1.0"
        
        # Test fixed parameters
        assert 0 < args["fixed"]["temperature"] < 1, "Temperature should be in (0,1)"
        assert args["fixed"]["eps"] > 0, "Epsilon should be positive"
        assert 0 < args["fixed"]["step_size"] <= 1, "Step size should be in (0,1]"
        assert args["fixed"]["clip_opinions"] == True, "Should clip opinions by default"
        
        # Test opinion range
        opinion_range = args["fixed"]["opinion_range"]
        assert len(opinion_range) == 2, "Opinion range should have min and max"
        assert opinion_range[0] < opinion_range[1], "Min should be less than max"


# Integration test helpers
class TestSBCMIntegrationHelpers:
    """Helper functions for integration testing."""
    
    @staticmethod
    def create_test_population(n_agents=10, opinion_range=(-1, 1)):
        """Create test population with random opinions."""
        return torch.uniform(torch.zeros(n_agents, 1), opinion_range[0], opinion_range[1])
    
    @staticmethod
    def create_test_network(n_agents=10, n_neighbors=3):
        """Create test social network structure."""
        neighbor_opinions = torch.randn(n_agents, n_neighbors)
        neighbor_indices = torch.randint(0, n_agents, (n_agents, n_neighbors))
        
        return {
            "neighbor_opinions": neighbor_opinions,
            "neighbor_indices": neighbor_indices
        }
    
    @staticmethod
    def assert_valid_opinions(opinions, opinion_range=(-1, 1)):
        """Assert that opinions are within valid range."""
        assert torch.all(opinions >= opinion_range[0]), f"Opinions should be >= {opinion_range[0]}"
        assert torch.all(opinions <= opinion_range[1]), f"Opinions should be <= {opinion_range[1]}"
    
    @staticmethod
    def compute_opinion_diversity(opinions):
        """Compute diversity measure of opinion distribution."""
        return torch.std(opinions).item()
    
    @staticmethod
    def detect_convergence(opinion_history, threshold=1e-4):
        """Detect if opinions have converged."""
        if len(opinion_history) < 2:
            return False
        
        last_change = torch.abs(opinion_history[-1] - opinion_history[-2])
        return torch.all(last_change < threshold)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])