from typing import List, Optional
from data_models import Coalition

class GlobalWorkspace:
    """Hosts the competition for consciousness and globally broadcasts the winner."""
    def __init__(self):
        self.coalitions: List[Coalition] = []

    async def receive_coalition(self, coalition: Coalition):
        self.coalitions.append(coalition)

    async def broadcast_conscious_contents(self) -> Optional[Coalition]:
        if not self.coalitions:
            return None
        
        winning_coalition = max(self.coalitions, key=lambda c: c.total_salience)
        self.coalitions.clear()
        
        return winning_coalition