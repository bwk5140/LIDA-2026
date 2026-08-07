from typing import List
from data_models import Node

class PerceptualAssociativeMemory:
    """Recognizes spatial features, states, and injects feelings into the percept dynamically."""
    
    async def recognize(self, sensory_data: dict) -> List[Node]:
        state_idx = sensory_data["grid_position"]
        reward = sensory_data["reward"]
        terminated = sensory_data["terminated"]
        
        # Create a percept for the current environmental location
        percepts = [
            Node(id=f"state_{state_idx}", content=state_idx, activation=1.0)
        ]
        
        # Dynamically evaluate feelings based on environment feedback
        if reward > 0:  
            # Reached the goal
            percepts.append(Node(id="feel_joy", content="goal", activation=1.0, valence=1.0, is_feeling=True))
            
        elif terminated and reward == 0:  
            # Fell in a hole (episode ended without reward)
            percepts.append(Node(id="feel_danger", content="hole", activation=0.9, valence=-1.0, is_feeling=True))
            
        else:
            # Intrinsic Motivation: Fatigue / Step Cost
            # A tiny negative valence (-0.01) teaches the agent that wasting time and bouncing 
            # against walls is slightly painful, forcing it to constantly seek new paths.
            percepts.append(Node(id="feel_fatigue", content="step", activation=0.1, valence=-0.01, is_feeling=True))
            
        return percepts