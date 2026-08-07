import asyncio
from typing import Any, Optional

from sensory_memory import SensoryMemory
from perceptual_associative_memory import PerceptualAssociativeMemory
from episodic_memory import EpisodicMemory
from declarative_memory import DeclarativeMemory
from workspace import Workspace, AttentionCodelet
from global_workspace import GlobalWorkspace
from procedural_memory import ProceduralMemory
from action_selection import ActionSelection
from sensory_motor_memory import SensoryMotorMemory
from data_models import Coalition

class LidaAgent:
    def __init__(self):
        self.sensory_memory = SensoryMemory()
        self.pam = PerceptualAssociativeMemory()
        self.episodic_memory = EpisodicMemory()
        self.declarative_memory = DeclarativeMemory()
        self.workspace = Workspace()
        self.attention_codelet = AttentionCodelet()
        self.global_workspace = GlobalWorkspace()
        self.procedural_memory = ProceduralMemory()
        self.action_selection = ActionSelection()
        self.sensory_motor_memory = SensoryMotorMemory()
        
        self.previous_state_id = None
        self.previous_action_code = None

    async def offline_consolidation(self):
        """Called between episodes to consolidate TEM into Declarative Memory."""
        # Consolidate undecayed contents
        await self.declarative_memory.consolidate(self.episodic_memory.memory_store)
        
        # Clear TEM (Simulating the decay of short-term transient memories)
        self.episodic_memory.memory_store.clear()

    async def trigger_learning(self, broadcast: Coalition, is_terminal: bool):
        await self.episodic_memory.encode(broadcast)
        
        if self.previous_state_id is not None and self.previous_action_code is not None:
            reward_signal = sum(n.valence for n in broadcast.nodes if n.is_feeling)
            
            max_future = 0.0
            if not is_terminal:
                current_state_node = next((n for n in broadcast.nodes if n.id.startswith("state_")), None)
                if current_state_node:
                    self.procedural_memory._ensure_schemes_exist(current_state_node.id)
                    max_future = max(s.base_level_activation for s in self.procedural_memory.schemes[current_state_node.id])
                    
            await self.procedural_memory.reinforce_scheme(
                context_id=self.previous_state_id,
                action_code=self.previous_action_code,
                reward=reward_signal,
                max_future_activation=max_future,
                is_terminal=is_terminal
            )
            
        if is_terminal:
            self.previous_state_id = None
            self.previous_action_code = None

    async def cognitive_cycle(self, env_state: Any, reward: float = 0.0, terminated: bool = False, truncated: bool = False) -> Optional[int]:
        is_terminal = terminated or truncated
        
        # --- PHASE 1: Understanding ---
        raw_data = await self.sensory_memory.sense(env_state, reward, terminated)
        percepts = await self.pam.recognize(raw_data)
        
        # Cue BOTH Transient Episodic Memory and Declarative Memory
        tem_associations = await self.episodic_memory.retrieve_local_associations(percepts)
        dm_associations = await self.declarative_memory.retrieve_local_associations(percepts)
        
        # Combine associations into the situational model
        all_associations = tem_associations + dm_associations
        await self.workspace.update_situational_model(percepts, all_associations)

        # --- PHASE 2: Attention (Consciousness) ---
        coalition = await self.attention_codelet.form_coalition(self.workspace.current_situational_model)
        await self.global_workspace.receive_coalition(coalition)
        conscious_broadcast = await self.global_workspace.broadcast_conscious_contents()

        if not conscious_broadcast:
            return None

        # --- PHASE 3: Action Selection and Learning ---
        asyncio.create_task(self.trigger_learning(conscious_broadcast, is_terminal))

        if is_terminal:
            return None

        instantiated_schemes = await self.procedural_memory.recruit_schemes(conscious_broadcast)
        selected_behavior = await self.action_selection.select_action(instantiated_schemes)
        
        if selected_behavior:
            action_code = await self.sensory_motor_memory.execute(selected_behavior)
            
            current_state_node = next((n for n in conscious_broadcast.nodes if n.id.startswith("state_")), None)
            if current_state_node:
                self.previous_state_id = current_state_node.id
                self.previous_action_code = action_code
                
            return action_code
            
        return None