from typing import List
from data_models import Node, Coalition

class EpisodicMemory:
    """Represents Transient Episodic Memory (TEM) for recent events."""
    
    def __init__(self):
        self.memory_store: List[List[Node]] = []
        self.max_capacity = 200 

    async def encode(self, broadcast: Coalition):
        """Memorizes events from the conscious broadcast."""
        if broadcast and broadcast.nodes:
            # FIX: Filter out associative nodes. We only want to remember core realities, 
            # not the act of remembering a memory.
            clean_memory = [
                n for n in broadcast.nodes 
                if not n.id.startswith("assoc_") and not n.id.startswith("dm_")
            ]
            
            if clean_memory:
                self.memory_store.append(clean_memory)
                if len(self.memory_store) > self.max_capacity:
                    self.memory_store.pop(0)

    async def retrieve_local_associations(self, cue_percepts: List[Node]) -> List[Node]:
        """Cues memory with current percepts to retrieve associated past events."""
        associations = []
        cue_ids = {node.id for node in cue_percepts}
        
        for memory in reversed(self.memory_store):
            memory_ids = {node.id for node in memory}
            
            if cue_ids.intersection(memory_ids):
                for node in memory:
                    if node.id not in cue_ids:
                        # FIX: Use safe Pydantic copying
                        assoc_node = node.model_copy() if hasattr(node, "model_copy") else node.copy()
                        assoc_node.id = f"assoc_{node.id}"
                        assoc_node.activation = max(0.1, assoc_node.activation * 0.5) 
                        associations.append(assoc_node)
                        
                break 
                
        return associations