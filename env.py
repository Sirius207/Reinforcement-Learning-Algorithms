import gym
import logging
import matplotlib.pyplot as plt

from dqn import DQN

logging.getLogger().setLevel(logging.INFO)


MEMORY_SIZE = 2000
BATCH_SIZE = 32
GAMMA = 0.85
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
ALPHA = 0.005
ALPHA_DECAY = 0.001

N_EPISODE = 10000

def main():
    env = gym.make("MountainCar-v0")
    # env._max_episode_steps=500
    # env = env.unwrapped

    print(env.action_space)
    print(env.observation_space)
    print(env.observation_space.high) # max position 0.6, max velocity 0.07
    print(env.observation_space.low)  # min position -1.2, min velocity -0.07

    agent = DQN(
        env=env,
        memory_size=MEMORY_SIZE,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        epsilon=EPSILON,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY,
        alpha=ALPHA,
        alpha_decay=ALPHA_DECAY,
    )

    reward_list = []
    for i_episode in range(N_EPISODE):
        state = agent.preprocess_state(env.reset())
        done = False
        steps = 0
        ep_reward = 0
        while not done:
            # env.render()
            action = agent.choose_action(state)
            new_state, reward, done, _ = env.step(action)

            # modify reward
            position, velocity = new_state
            if(position >= 0.5):
                reward = 100
            else:
                reward = abs(position - (-0.5))

            new_state = agent.preprocess_state(new_state)
            agent.remember(state, action, reward, new_state, done)
        
            agent.replay()
            agent.train_target()

            state = new_state
            if(done):
                reward_list.append(ep_reward)
                logging.info(f"Episode: {i_episode}, Steps: {steps}, Reward: {ep_reward}")
                break

            steps += 1
            ep_reward += reward
    
    plt.plot(reward_list)
    plt.show()


if(__name__ == "__main__"):
    main()