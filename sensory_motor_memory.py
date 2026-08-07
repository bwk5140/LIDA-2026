from data_models import InstantiatedScheme

class SensoryMotorMemory:
    """Produces the final motor integer plan for the Gym environment."""
    async def execute(self, selected_behavior: InstantiatedScheme) -> int:
        print(f"[Motor Command] Intention locked: {selected_behavior.scheme.action_name}")
        return selected_behavior.scheme.action_code