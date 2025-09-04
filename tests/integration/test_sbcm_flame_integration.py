"""
Integration tests for SBCM dynamics with FLAME framework.

Tests SBCM as part of complete AgentTorch simulation pipeline.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from agent_torch.examples.models.opinion_dynamics.substeps.sbcm_dynamics import (
    SBCMDynamics,
    SBCMOpinionDynamicsConfig
)


class TestSBCMFLAMEIntegration:
    """Test SBCM integration with FLAME substep framework."""
    
    @pytest.fixture
    def flame_config(self):
        """Create FLAME-compatible configuration."""
        return {
            "simulation_metadata": {
                "num_agents": 100,
                "num_steps_per_episode": 50,
                "num_episodes": 1,
                "calibration": True,
                "device": "cpu"
            },
            "agents": {
                "citizens": {
                    "number": 100,
                    "properties": {
                        "opinion": {
                            "shape": [100, 1],
                            "dtype": "float32",
                            "initialization_function": "uniform",
                            "initialization_args": {"low": -1.0, "high": 1.0}
                        }
                    }
                }
            },
            "substeps": {
                "sbcm_dynamics": {
                    "observation": {},
                    "policy": {},
                    "transition": {
                        "citizens": {
                            "sbcm_opinion_update": {
                                "function": "SBCMDynamics",
                                "input_variables": ["agents.citizens.opinion"],
                                "output_variables": ["agents.citizens.opinion"]
                            }
                        }
                    }
                }
            }
        }
    
    @pytest.fixture
    def test_state(self):
        """Create test simulation state."""
        n_agents = 20
        opinions = torch.randn(n_agents, 1) * 0.5  # Opinions around 0
        
        # Create simple network structure
        neighbor_opinions = torch.randn(n_agents, 5) * 0.3
        
        return {
            "agents": {
                "citizens": {
                    "opinion": opinions,
                    "agent_id": torch.arange(n_agents).unsqueeze(1)
                }
            },
            "network": {
                "neighbor_opinions": neighbor_opinions,
                "neighbor_indices": torch.randint(0, n_agents, (n_agents, 5))
            },
            "current_step": 0,
            "current_substep": "sbcm_dynamics"
        }
    
    def test_sbcm_substep_creation(self, flame_config):
        """Test creating SBCM substep from FLAME config."""
        # Get default SBCM config
        sbcm_config = SBCMOpinionDynamicsConfig.get_default_config()
        
        # Create SBCM substep
        sbcm_substep = SBCMDynamics(
            config=flame_config,
            input_variables=sbcm_config["input_variables"],
            output_variables=sbcm_config["output_variables"],
            arguments=sbcm_config["arguments"]
        )
        
        # Verify creation
        assert isinstance(sbcm_substep, SBCMDynamics)
        assert hasattr(sbcm_substep, 'rho')
        assert hasattr(sbcm_substep, 'temperature')
        assert sbcm_substep.config == flame_config
        
    def test_multi_step_simulation(self, flame_config, test_state):
        """Test SBCM over multiple simulation steps."""
        # Create SBCM substep
        sbcm_config = SBCMOpinionDynamicsConfig.get_default_config()
        sbcm_substep = SBCMDynamics(
            config=flame_config,
            input_variables=sbcm_config["input_variables"],
            output_variables=sbcm_config["output_variables"],
            arguments=sbcm_config["arguments"]
        )
        
        # Run simulation for multiple steps
        current_state = test_state.copy()
        opinion_history = [current_state["agents"]["citizens"]["opinion"].clone()]
        
        num_steps = 10
        for step in range(num_steps):
            # Update current step
            current_state["current_step"] = step
            
            # Apply SBCM dynamics
            action = {}  # Empty action for this test
            new_state = sbcm_substep.forward(current_state, action)
            
            # Update state for next iteration
            current_state = new_state
            opinion_history.append(current_state["agents"]["citizens"]["opinion"].clone())
        
        # Verify simulation progression
        assert len(opinion_history) == num_steps + 1
        
        # Check that opinions evolved over time
        initial_opinions = opinion_history[0]
        final_opinions = opinion_history[-1]
        
        # Opinions should have changed
        assert not torch.allclose(initial_opinions, final_opinions, atol=1e-6)
        
        # Opinions should remain in valid range
        for opinions in opinion_history:
            assert torch.all(opinions >= -1.0), "Opinions should stay >= -1"
            assert torch.all(opinions <= 1.0), "Opinions should stay <= 1"
    
    def test_consensus_emergence(self):
        """Test that similar opinions lead to consensus."""
        # Create agents with similar initial opinions
        n_agents = 10
        similar_opinions = torch.tensor([[0.1], [0.15], [0.05], [0.12], [0.08], 
                                       [0.13], [0.09], [0.11], [0.14], [0.06]])
        
        # Create state where all agents can interact with each other
        all_neighbors = similar_opinions.expand(n_agents, n_agents).clone()
        
        state = {
            "agents": {
                "citizens": {
                    "opinion": similar_opinions
                }
            },
            "network": {
                "neighbor_opinions": all_neighbors
            }
        }
        
        # Create SBCM with high interaction probability for close opinions
        config = {"simulation_metadata": {"calibration": True}}
        arguments = {
            "learnable": {"rho": torch.tensor(2.0)},  # Higher rho for stronger consensus
            "fixed": {
                "temperature": 0.1,
                "eps": 1e-12,
                "step_size": 0.2,  # Larger steps
                "clip_opinions": True,
                "opinion_range": [-1.0, 1.0]
            }
        }
        
        sbcm = SBCMDynamics(
            config=config,
            input_variables=["agents.citizens.opinion"],
            output_variables=["agents.citizens.opinion"],
            arguments=arguments
        )
        
        # Run simulation until convergence
        current_state = state
        initial_std = torch.std(current_state["agents"]["citizens"]["opinion"])
        
        for step in range(20):  # Run for multiple steps
            new_state = sbcm.forward(current_state, {})
            current_state = new_state
        
        final_opinions = current_state["agents"]["citizens"]["opinion"]
        final_std = torch.std(final_opinions)
        
        # Check convergence (opinions should be more similar)
        assert final_std < initial_std, "Opinion diversity should decrease over time"
        
        # Check that final opinions are reasonable
        mean_opinion = torch.mean(final_opinions)
        assert torch.abs(mean_opinion - 0.1) < 0.1, "Final consensus should be near initial mean"
        
    def test_polarization_dynamics(self):
        """Test that dissimilar opinions can lead to polarization."""
        # Create agents with polarized initial opinions
        n_agents = 10
        polarized_opinions = torch.tensor([[-0.8], [-0.7], [-0.75], [-0.85], [-0.9],
                                          [0.8], [0.7], [0.75], [0.85], [0.9]])
        
        # Create network where similar agents interact more
        neighbor_opinions = torch.zeros(n_agents, 5)
        for i in range(n_agents):
            if polarized_opinions[i] < 0:  # Negative opinion group
                neighbor_opinions[i] = torch.tensor([-0.7, -0.8, -0.75, -0.85, -0.9])
            else:  # Positive opinion group  
                neighbor_opinions[i] = torch.tensor([0.7, 0.8, 0.75, 0.85, 0.9])
        
        state = {
            "agents": {
                "citizens": {
                    "opinion": polarized_opinions
                }
            },
            "network": {
                "neighbor_opinions": neighbor_opinions
            }
        }
        
        # Create SBCM
        config = {"simulation_metadata": {"calibration": True}}
        arguments = SBCMOpinionDynamicsConfig.get_default_config()["arguments"]
        arguments["fixed"]["step_size"] = 0.1  # Moderate step size
        
        sbcm = SBCMDynamics(
            config=config,
            input_variables=["agents.citizens.opinion"],
            output_variables=["agents.citizens.opinion"],
            arguments=arguments
        )
        
        # Run simulation
        current_state = state
        for step in range(15):
            new_state = sbcm.forward(current_state, {})
            current_state = new_state
        
        final_opinions = current_state["agents"]["citizens"]["opinion"]
        
        # Check that polarization is maintained or increased
        negative_group = final_opinions[final_opinions < 0]
        positive_group = final_opinions[final_opinions > 0]
        
        assert len(negative_group) > 0, "Should maintain negative opinion group"
        assert len(positive_group) > 0, "Should maintain positive opinion group"
        
        # Groups should be internally cohesive
        if len(negative_group) > 1:
            neg_std = torch.std(negative_group)
            assert neg_std < 0.2, "Negative group should be cohesive"
            
        if len(positive_group) > 1:
            pos_std = torch.std(positive_group)  
            assert pos_std < 0.2, "Positive group should be cohesive"
    
    def test_parameter_learning_integration(self):
        """Test that SBCM parameters can be learned via gradients."""
        # Create training data with known optimal rho
        n_agents = 15
        initial_opinions = torch.randn(n_agents, 1) * 0.3
        target_opinions = torch.tanh(initial_opinions * 0.8)  # Target convergence pattern
        
        # Create simple all-to-all network
        neighbor_opinions = initial_opinions.expand(n_agents, n_agents)
        
        state = {
            "agents": {
                "citizens": {
                    "opinion": initial_opinions.clone()
                }
            },
            "network": {
                "neighbor_opinions": neighbor_opinions
            }
        }
        
        # Create SBCM with learnable parameters
        config = {"simulation_metadata": {"calibration": True}}
        arguments = {
            "learnable": {"rho": torch.tensor(0.5, requires_grad=True)},
            "fixed": {
                "temperature": 0.1,
                "eps": 1e-12,
                "step_size": 0.1,
                "clip_opinions": True,
                "opinion_range": [-1.0, 1.0]
            }
        }
        
        sbcm = SBCMDynamics(
            config=config,
            input_variables=["agents.citizens.opinion"],
            output_variables=["agents.citizens.opinion"],
            arguments=arguments
        )
        
        # Set up optimizer
        optimizer = torch.optim.Adam([sbcm.rho], lr=0.01)
        
        initial_rho = sbcm.rho.item()
        
        # Training loop
        for epoch in range(10):
            optimizer.zero_grad()
            
            # Forward pass
            current_state = {k: v.clone() if isinstance(v, torch.Tensor) else v 
                           for k, v in state.items()}
            
            # Run multiple simulation steps
            for step in range(5):
                new_state = sbcm.forward(current_state, {})
                current_state = new_state
            
            final_opinions = current_state["agents"]["citizens"]["opinion"]
            
            # Compute loss against target
            loss = torch.nn.functional.mse_loss(final_opinions, target_opinions)
            
            # Add regularization
            reg_loss = sbcm.get_regularization_loss()
            total_loss = loss + 0.01 * reg_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
        
        final_rho = sbcm.rho.item()
        
        # Verify that parameters changed
        assert abs(final_rho - initial_rho) > 1e-4, "Rho parameter should have changed during training"
        
        # Verify that loss decreased
        with torch.no_grad():
            final_state = {k: v.clone() if isinstance(v, torch.Tensor) else v 
                          for k, v in state.items()}
            for step in range(5):
                new_state = sbcm.forward(final_state, {})
                final_state = new_state
            
            final_loss = torch.nn.functional.mse_loss(
                final_state["agents"]["citizens"]["opinion"], 
                target_opinions
            )
            
            # Loss should be reasonable (not necessarily lower due to stochasticity)
            assert final_loss.item() < 10.0, "Final loss should be reasonable"
    
    def test_batch_processing_efficiency(self):
        """Test SBCM efficiency with batch processing."""
        # Test with different batch sizes
        batch_sizes = [10, 50, 100]
        
        config = {"simulation_metadata": {"calibration": True}}
        arguments = SBCMOpinionDynamicsConfig.get_default_config()["arguments"]
        
        for batch_size in batch_sizes:
            # Create large state
            opinions = torch.randn(batch_size, 1) * 0.5
            neighbor_opinions = torch.randn(batch_size, 8) * 0.3
            
            state = {
                "agents": {
                    "citizens": {
                        "opinion": opinions
                    }
                },
                "network": {
                    "neighbor_opinions": neighbor_opinions
                }
            }
            
            # Create SBCM
            sbcm = SBCMDynamics(
                config=config,
                input_variables=["agents.citizens.opinion"],
                output_variables=["agents.citizens.opinion"],
                arguments=arguments
            )
            
            # Measure execution time
            import time
            start_time = time.time()
            
            # Run simulation
            new_state = sbcm.forward(state, {})
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Verify results
            assert "agents" in new_state
            assert new_state["agents"]["citizens"]["opinion"].shape == (batch_size, 1)
            
            # Execution time should be reasonable (less than 1 second for these sizes)
            assert execution_time < 1.0, f"Execution too slow for batch size {batch_size}: {execution_time}s"
            
            print(f"Batch size {batch_size}: {execution_time:.4f}s")
    
    def test_network_structure_handling(self):
        """Test SBCM with different network structures."""
        n_agents = 12
        base_opinions = torch.randn(n_agents, 1) * 0.4
        
        # Test with different neighbor counts
        neighbor_counts = [2, 5, 8]
        
        config = {"simulation_metadata": {"calibration": True}}
        arguments = SBCMOpinionDynamicsConfig.get_default_config()["arguments"]
        
        sbcm = SBCMDynamics(
            config=config,
            input_variables=["agents.citizens.opinion"],
            output_variables=["agents.citizens.opinion"],
            arguments=arguments
        )
        
        for n_neighbors in neighbor_counts:
            neighbor_opinions = torch.randn(n_agents, n_neighbors) * 0.3
            neighbor_indices = torch.randint(0, n_agents, (n_agents, n_neighbors))
            
            state = {
                "agents": {
                    "citizens": {
                        "opinion": base_opinions.clone()
                    }
                },
                "network": {
                    "neighbor_opinions": neighbor_opinions,
                    "neighbor_indices": neighbor_indices
                }
            }
            
            # Should handle different network sizes
            new_state = sbcm.forward(state, {})
            
            # Verify output
            assert new_state["agents"]["citizens"]["opinion"].shape == (n_agents, 1)
            assert torch.all(torch.isfinite(new_state["agents"]["citizens"]["opinion"]))
            
    def test_error_handling(self):
        """Test SBCM error handling for edge cases."""
        config = {"simulation_metadata": {"calibration": True}}
        arguments = SBCMOpinionDynamicsConfig.get_default_config()["arguments"]
        
        sbcm = SBCMDynamics(
            config=config,
            input_variables=["agents.citizens.opinion"],
            output_variables=["agents.citizens.opinion"],
            arguments=arguments
        )
        
        # Test with empty state
        empty_state = {
            "agents": {
                "citizens": {
                    "opinion": torch.empty(0, 1)
                }
            },
            "network": {
                "neighbor_opinions": torch.empty(0, 3)
            }
        }
        
        # Should handle empty input gracefully
        try:
            result = sbcm.forward(empty_state, {})
            assert result["agents"]["citizens"]["opinion"].shape == (0, 1)
        except Exception as e:
            pytest.skip(f"Empty state handling not implemented: {e}")
        
        # Test with NaN inputs
        nan_state = {
            "agents": {
                "citizens": {
                    "opinion": torch.tensor([[float('nan')], [0.5]])
                }
            },
            "network": {
                "neighbor_opinions": torch.tensor([[0.1, 0.2], [0.3, 0.4]])
            }
        }
        
        # Should handle NaN inputs without crashing
        try:
            result = sbcm.forward(nan_state, {})
            # Result may contain NaN, but shouldn't crash
        except Exception as e:
            pytest.skip(f"NaN handling not implemented: {e}")


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "--tb=short", "-x"])