import gymnasium as gym
import asyncio
from agent import LidaAgent

async def run_env():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="human")
    agent = LidaAgent()
    episodes = 500  
    
    for ep in range(episodes):
        print(f"\n--- Starting Episode {ep + 1} ---")
        obs, info = env.reset()
        reward = 0.0
        terminated = False
        truncated = False
        done = False
        
        while not done:
            action = await agent.cognitive_cycle(obs, reward, terminated, truncated)
            
            if action is None:
                action = env.action_space.sample()
                
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            await asyncio.sleep(0.01) 

        # Final cycle to internalize terminal feelings
        await agent.cognitive_cycle(obs, reward, terminated, truncated)
        
        if reward > 0:
            print(f"Goal reached on episode {ep + 1}!")
            await asyncio.sleep(1) # Pause to celebrate

        await agent.offline_consolidation()

    print("\n--- Simulation and Training Ended ---")
    env.close()

if __name__ == "__main__":
    asyncio.run(run_env())