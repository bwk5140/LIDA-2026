from pydantic import BaseModel, Field
from typing import Any, List

class Node(BaseModel):
    """A fundamental unit of representation, including feelings."""
    id: str
    content: Any
    activation: float = Field(default=0.0, ge=0.0, le=1.0)
    valence: float = Field(default=0.0, ge=-1.0, le=1.0)
    is_feeling: bool = False

class Coalition(BaseModel):
    """Formed by attention codelets competing for consciousness."""
    nodes: List[Node]
    total_salience: float = 0.0

class Scheme(BaseModel):
    """A procedural memory template for an action."""
    context_id: str
    action_name: str
    action_code: int
    base_level_activation: float = 0.0

class InstantiatedScheme(BaseModel):
    """A scheme instantiated with specific variables for the current situation."""
    scheme: Scheme
    current_activation: float = 0.0