
from Quad_Env import Quad_Env
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
from stable_baselines.common.cmd_util import make_vec_env
import tensorflow as tf

env = Quad_Env()
env = make_vec_env(lambda: env, n_envs=64)
# If the environment don't follow the interface, an error will be thrown
# see stable baselines custom env documentation for more.
obs = env.reset()

#Define Neural Network Architecture and activation functions to use
policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[64, 64 , 64, 64])



model = PPO2(MlpPolicy, env ,nminibatches=32 ,
             policy_kwargs=policy_kwargs, tensorboard_log="./TB/" )

#model = PPO2.load("fileNameOfTrainedAgent")
#model.set_env(env)


model.learn(total_timesteps=10000000,  log_interval=10000)

model.save("QuadBlendingAgent")

print("Training complete - agent saved")