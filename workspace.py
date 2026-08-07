from typing import List
from data_models import Node, Coalition

class Workspace:
    """Holds the Current Situational Model assembled pre-consciously."""
    def __init__(self):
        self.current_situational_model: List[Node] = []

    async def update_situational_model(self, percepts: List[Node], associations: List[Node]):
        # Structure building codelets synthesize the current reality here
        self.current_situational_model = percepts + associations

class AttentionCodelet:
    """Claims portions of the situational model to form coalitions."""
    async def form_coalition(self, situational_model: List[Node]) -> Coalition:
        # Sums up arousal/activations to determine the priority of the coalition
        salience = sum(node.activation for node in situational_model)
        return Coalition(nodes=situational_model, total_salience=salience)