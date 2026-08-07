from typing import List, Dict
from data_models import Node

class DeclarativeMemory:
    """
    Long-term memory module encompassing Autobiographical and Semantic Memory.
    """
    def __init__(self):
        self.long_term_store: List[List[Node]] = []
        self.semantic_facts: Dict[str, Node] = {}

    async def consolidate(self, transient_memory_store: List[List[Node]]):
        """Offline consolidation: transfers strong memories from TEM into DM."""
        for memory in transient_memory_store:
            has_strong_emotion = any(n.is_feeling and abs(n.valence) > 0.5 for n in memory)
            
            if has_strong_emotion:
                memory_ids = {n.id for n in memory}
                is_duplicate = any({n.id for n in stored_mem} == memory_ids for stored_mem in self.long_term_store)
                
                if not is_duplicate:
                    self.long_term_store.append(memory)
                    
                    state_node = next((n for n in memory if n.id.startswith("state_")), None)
                    feeling_node = next((n for n in memory if n.is_feeling and abs(n.valence) > 0.5), None)
                    
                    if state_node and feeling_node:
                        fact_id = f"fact_{state_node.id}_{feeling_node.content}"
                        fact_node = Node(
                            id=fact_id, 
                            content=f"Fact: {state_node.content} is {feeling_node.content}", 
                            activation=0.5, 
                            valence=feeling_node.valence
                        )
                        self.semantic_facts[state_node.id] = fact_node

    async def retrieve_local_associations(self, cue_percepts: List[Node]) -> List[Node]:
        """Cues DM to retrieve long-term episodic events and semantic facts."""
        associations = []
        cue_ids = {node.id for node in cue_percepts}
        
        # 1. Retrieve Semantic Facts
        for cue in cue_percepts:
            if cue.id in self.semantic_facts:
                fact_source = self.semantic_facts[cue.id]
                fact_node = fact_source.model_copy() if hasattr(fact_source, "model_copy") else fact_source.copy()
                fact_node.id = f"dm_{fact_node.id}"
                fact_node.activation *= 0.8  
                associations.append(fact_node)

        # 2. Retrieve Autobiographical Events
        for memory in reversed(self.long_term_store):
            memory_ids = {node.id for node in memory}
            if cue_ids.intersection(memory_ids):
                for node in memory:
                    if node.id not in cue_ids:
                        assoc_node = node.model_copy() if hasattr(node, "model_copy") else node.copy()
                        assoc_node.id = f"dm_assoc_{node.id}"
                        assoc_node.activation *= 0.3 
                        associations.append(assoc_node)
                break  
                
        return associations