from typing import List, Dict
from data_models import Scheme, InstantiatedScheme, Coalition

class ProceduralMemory:
    def __init__(self):
        self.schemes: Dict[str, List[Scheme]] = {}
        self.primitive_actions = [
            {"name": "Move Left", "code": 0},
            {"name": "Move Down", "code": 1},
            {"name": "Move Right", "code": 2},
            {"name": "Move Up", "code": 3}
        ]

    def _ensure_schemes_exist(self, context_id: str):
        if context_id not in self.schemes:
            self.schemes[context_id] = [
                Scheme(
                    context_id=context_id,
                    action_name=action["name"],
                    action_code=action["code"],
                    # FIX: Start at 0.0 so safe moves don't decay when reward is 0
                    base_level_activation=0.0 
                ) for action in self.primitive_actions
            ]

    async def recruit_schemes(self, broadcast: Coalition) -> List[InstantiatedScheme]:
        instantiated = []
        state_id = next((node.id for node in broadcast.nodes if node.id.startswith("state_")), None)
        
        if not state_id:
            return instantiated
            
        self._ensure_schemes_exist(state_id)
        
        for scheme in self.schemes[state_id]:
            bonus_activation = sum(n.activation for n in broadcast.nodes if n.is_feeling and n.valence > 0)
            penalty = sum(n.activation for n in broadcast.nodes if n.is_feeling and n.valence < 0)
            
            # FIX: Removed the 0.0 clamp! Negative values now correctly repel the agent.
            final_activation = scheme.base_level_activation + bonus_activation - penalty
            instantiated.append(InstantiatedScheme(scheme=scheme, current_activation=final_activation))
                
        return instantiated
        
    async def reinforce_scheme(self, context_id: str, action_code: int, reward: float, max_future_activation: float, is_terminal: bool):
        alpha = 0.1  
        gamma = 0.9  
        
        if context_id in self.schemes:
            for scheme in self.schemes[context_id]:
                if scheme.action_code == action_code:
                    current_activation = scheme.base_level_activation
                    
                    if is_terminal:
                        target = reward
                    else:
                        target = reward + (gamma * max_future_activation)
                        
                    scheme.base_level_activation += alpha * (target - current_activation)