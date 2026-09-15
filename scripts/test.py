import matplotlib.pyplot as plt
import gym_super_mario_bros

env  = gym_super_mario_bros.make('SuperMarioBros-1-1-v3')

env.reset()
image = env.render(mode = 'rgb_array')
save_path = 'output/test.png'

# Save the image
plt.imsave(save_path, image)
print(f'Saved image to {save_path}')
env.close()