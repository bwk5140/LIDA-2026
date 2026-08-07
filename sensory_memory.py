from typing import Any

class SensoryMemory:
    async def sense(self, raw_env_state: Any, reward: float, terminated: bool) -> dict:
        return {
            "grid_position": raw_env_state,
            "reward": reward,
            "terminated": terminated
        }