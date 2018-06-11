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
