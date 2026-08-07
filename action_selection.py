import random
from typing import List, Optional
from data_models import InstantiatedScheme

class ActionSelection:
    """Chooses a single behavior to dominate this cognitive cycle."""
    
    async def select_action(self, instantiated_schemes: List[InstantiatedScheme]) -> Optional[InstantiatedScheme]:
        if not instantiated_schemes:
            return None
        
        # 1. Find the highest current activation among all instantiated schemes
        max_activation = max(s.current_activation for s in instantiated_schemes)
        
        # 2. Gather ALL schemes that share this exact maximum activation
        best_schemes = [s for s in instantiated_schemes if s.current_activation == max_activation]
        
        # 3. Randomly select one from the tied winners (This creates natural exploration)
        selected_behavior = random.choice(best_schemes)
        
        return selected_behavior