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

N_EPISODE = 20
UPDATE_TARGET_PERIOD = 1
START_REPLAY_EPISODE = 0

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
    steps_list = []
    for i_episode in range(N_EPISODE):
        state = agent.preprocess_state(env.reset())
        done = False
        steps = 0
        ep_reward = 0
        while not done:
            # env.render()
            action = agent.choose_action(state)
            # prevent output 1 to raise performance
            action_ = 2 if action == 1 else 0
            new_state, reward, done, _ = env.step(action_)

            # modify reward
            position, velocity = new_state
            if (action_ == 2 and velocity > 0):
                reward = 1
            elif(action_ == 0 and velocity < 0):
                reward = 1
            else:
                reward = -2
            
            if (position - (-0.5) > 0):
                reward += abs(position - (-0.5))

            if (position > 0.5):
                reward = (200-steps) * 100

            new_state = agent.preprocess_state(new_state)
            agent.remember(state, action, reward, new_state, done)
        
            if(i_episode > START_REPLAY_EPISODE):
                agent.replay()
                if(i_episode % UPDATE_TARGET_PERIOD == 0):
                    agent.train_target()

            state = new_state
            ep_reward += reward

            if(done):
                reward_list.append(ep_reward)
                steps_list.append(steps)
                logging.info(f"Episode: {i_episode}, Steps: {steps}, Reward: {ep_reward}, Epsilon: {agent.epsilon}")
                break

            steps += 1
    
    with open('./log/dqn.csv', 'w') as output:
        output.write('reward,steps\n')
        for episode in range(N_EPISODE):
            output.write(str(reward_list[episode])+',')
            output.write(str(steps_list[episode])+'\n')
    plt.plot(reward_list)
    plt.show()


if(__name__ == "__main__"):
    main()