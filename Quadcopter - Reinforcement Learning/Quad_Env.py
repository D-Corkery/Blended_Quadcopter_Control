import gym
import time
from sklearn.preprocessing import normalize
import quadcopter,controller #,gui
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
from stable_baselines.common.cmd_util import make_vec_env
import tensorflow as tf
from gym import spaces

from gym.spaces import Tuple, Box, Discrete, MultiDiscrete, MultiBinary, Dict
totalNumberRuns = 0
allCumulativeRewards = [0]
allRuns = [0]
logInterval = 10
epNumberLevelUp = [0]
ep_status = 0
count = 0
avgOverAllDomains = []
#avgDomainPerf = {"None": [-1000], "Rotor": [-1000,-1000,-1000,-1000] , "Wind": [-1000,-1000,-1000,-1000] ,
avgDomainPerf = [ [1000,1000,1000,1000] ,  [1000,1000,1000,1000]  ,
                 [1000,1000,1000,1000]  ,[1000,1000,1000,1000] ]


domainProb = [[0.0625, 0.0625, 0.0625, 0.0625], [0.0625, 0.0625, 0.0625, 0.0625], [0.0625, 0.0625, 0.0625, 0.0625], [0.0625, 0.0625, 0.0625, 0.0625]]
domainProbDist = [0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625]

#allDomainPerf = {"None": [], "Rotor": [[],[],[],[]] , "Wind": [[],[],[],[]] ,
allDomainPerf = [ [[],[],[],[]] ,  [[],[],[],[]] ,
                 [[],[],[],[]] , [[],[],[],[]] ]

fig, ax1= plt.subplots()
#
# plt.bar(allRuns, allCumulativeRewards, color='blue')
# plt.xlabel("Episode Number")
# plt.ylabel("Reward")
# plt.title("Quadcopter Training under faults")
#
#
plt.ion()




Type ="None"
TypeInd = 0
Level = 0
converged = False
last_rewards = []
reward_threshold = 100 #all waypoints - 100 threshold
convergence_length = 20
stableAtGoal = 0

BLENDED_CONTROLLER_PARAMETERS = {'Motor_limits': [0, 9000],
                         'Tilt_limits': [-10, 10],
                         'Yaw_Control_Limits': [-900, 900],
                         'Z_XY_offset': 500,
                         'Linear_PID': {'P': [300, 300, 7500], 'I': [0.04, 0.04, 5], 'D': [450, 450, 5400]},
                         'Linear_PID2': {'P': [300, 300, 7000], 'I': [0.04, 0.04, 4.5], 'D': [450, 450, 5000]},
                         'Linear_To_Angular_Scaler': [1, 1, 0],
                         'Yaw_Rate_Scaler': 0.18,
                         'Angular_PID': {'P': [24000, 24000, 1500], 'I': [0, 0, 1.2], 'D': [12000, 12000, 0]},
                         'Angular_PID2': {'P': [4000, 4000, 1500], 'I': [0, 0, 1.2], 'D': [1500, 1500, 0]},
                         }

#original angular rate = {'P': [22000, 22000, 1500], 'I': [0, 0, 1.2], 'D': [12000, 12000, 0]}
stepsToGoal = 0
stepsToGoal = 0
steps = 4
# x_path = [0, 0, 5, 0, -5, 0, 5, 5]
# y_path = [0, 0, 0, 5, 0, -5, 0, 0]
# z_path = [0, 5, 5, 5, 5, 5, 5, 5]
limit = 10
x_dest = np.random.randint(-limit, limit)
y_dest = np.random.randint(-limit, limit)
z_dest = np.random.randint(5, limit)
x_path = [0, 0, x_dest, x_dest]
y_path = [0, 0, y_dest, y_dest]
z_path = [5, 5, z_dest, z_dest]
interval_steps = 50
yaws = np.zeros(steps)
goals = []
safe_region = []

for i in range(steps):
    if(i < steps-1 ):
        #create linespace between waypoint i and i+1
        x_lin = np.linspace(x_path[i], x_path[i+1], interval_steps)
        y_lin =  np.linspace(y_path[i], y_path[i+1], interval_steps)
        z_lin =  np.linspace(z_path[i], z_path[i+1], interval_steps)
    else:
        x_lin = np.linspace(x_path[i], x_path[i], interval_steps)
        y_lin = np.linspace(y_path[i], y_path[i], interval_steps)
        z_lin = np.linspace(z_path[i], z_path[i], interval_steps)

    goals.append([x_path[i], y_path[i], z_path[i]])
    #for each pos in linespace append a goal
    safe_region.append([])
    for j in range(interval_steps):
        safe_region[i].append([x_lin[j], y_lin[j], z_lin[j]])
        stepsToGoal +=1




class Quad_Env(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a simple env where the agent must learn to go always left.
    """
    # Because of google colab, we cannot implement the GUI ('human' render mode)
    #metadata = {'render.modes': ['console']}
    # Define constants for clearer code


    def __init__(self):
        super(Quad_Env, self).__init__()
        converged = False
        self.quad_id = 1

        weight = np.random.choice([0.8, 1.2, 1.6])

        QUADCOPTER = {str(self.quad_id): {'position': [0, 0, 6], 'orientation': [0, 0, 0], 'L': 0.3, 'r': 0.1, 'prop_size': [10, 4.5],
                            'weight': weight}}
        # Make objects for quadcopter
        self.quad = quadcopter.Quadcopter(QUADCOPTER)
        #self.gui_object = gui.GUI(quads=QUADCOPTER)
        self.sampleTime = 0.01
        #create blended controller and link it to quadcopter object
        self.ctrl = controller.Blended_PID_Controller(self.quad.get_state, self.quad.get_time,
                                                      self.quad.set_motor_speeds, self.quad.get_motor_speeds,
                                                      self.quad.stepQuad, self.quad.set_motor_faults, self.quad.setWind, self.quad.setNormalWind,
                                                      params=BLENDED_CONTROLLER_PARAMETERS, quad_identifier=str(self.quad_id))
        self.ctrl.setController("Agent")
        #self.gui_object = gui.GUI(quads=QUADCOPTER, ctrl=self.ctrl)

        generateRandomPath()
        self.current = 0
        self.ctrl.update_target(goals[self.current], safe_region[self.current])
        self.setRandomFault()
        self.stableAtGoal =0

        self.cumu_reward= 0
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions, we have two: left and right
        n_actions = 3
        #PID LOW and HIHG
        self.actionlow = 0.01
        self.actionhigh =1
        self.action_low_state = np.array([self.actionlow, self.actionlow, self.actionlow, self.actionlow], dtype=np.float)

        self.action_high_state = np.array([self.actionhigh, self.actionhigh, self.actionlow, self.actionlow], dtype=np.float)

        self.action_space = spaces.Box(low=self.action_low_state, high=self.action_high_state, dtype=np.float)


        # The observation will be the coordinate of the agent
        # this can be described both by Discrete and Box space

       # print(self.observation_space)
       #  self.low_error=-1
       #  self.high_error=1
        self.low_dest= -10
        self.high_dest =10
        # self.low_act = -10000
        # self.high_act = 10000
        # self.low = 0
        # self.high = 10000
        self.low_state = np.array([self.low_dest, self.low_dest,self.low_dest, self.low_dest, self.low_dest, self.low_dest,self.low_dest, self.low_dest, self.low_dest
                                   ], dtype=np.float32)
       # self.low_state = np.array([self.low_dest,self.low_dest,  self.low_error,self.low_error, self.low_act, self.low_act, self.low_act, self.low_act])
        #self.high_state = np.array([self.high_dest,self.high_dest, self.high_error, self.high_error, self.high_act, self.high_act, self.high_act,self.high_act])

        self.high_state = np.array([self.high_dest, self.high_dest,self.high_dest, self.high_dest, self.high_dest, self.high_dest, self.high_dest, self.high_dest, self.high_dest
                                    ], dtype=np.float32)

        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state,dtype=np.float32)





    def reset(self):
        converged= False
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        # Initialize the agent at the right of the grid
        self.quad_id += 1
        weight = np.random.choice([0.8, 1.2, 1.6])

        QUADCOPTER = {str(self.quad_id): {'position': [0, 0, 6], 'orientation': [0, 0, 0], 'L': 0.3, 'r': 0.1,
                                          'prop_size': [10, 4.5],
                                          'weight': weight}}

        self.quad = quadcopter.Quadcopter(QUADCOPTER)
        #self.gui_object = gui.GUI(quads=QUADCOPTER)
        self.ctrl = controller.Blended_PID_Controller(self.quad.get_state, self.quad.get_time,
                                                      self.quad.set_motor_speeds, self.quad.get_motor_speeds,
                                                      self.quad.stepQuad, self.quad.set_motor_faults,self.quad.setWind, self.quad.setNormalWind,
                                                      params=BLENDED_CONTROLLER_PARAMETERS, quad_identifier=str(self.quad_id))

        self.ctrl.setController("Agent")
        #self.gui_object.close()
       # self.gui_object = gui.GUI(quads=QUADCOPTER, ctrl=self.ctrl)
        self.stableAtGoal = 0
        #global totalNumberRuns
        #totalNumberRuns += 1
        generateRandomPath()

        self.current = 0
        self.ctrl.update_target(goals[self.current], safe_region[self.current])
        #print(type(self.quad.get_state("q1")))
        self.setRandomFault()
        self.cumu_reward = 0

        obs = self.ctrl.get_updated_observations()
        obs_array = []
#        for key, value in obs.items():
 #           obs_array.append(value)
        return obs
        # here we convert to float32 to make it more general (in case we want to use continuous actions)

    def step(self, action):
        print("==================================")
        print("Action=" + str(action))
        global last_rewards, maxFault, maxRotor,converged,convergence_length,\
            totalNumberRuns, allCumulativeRewards  , allRuns, epNumberLevelUp \
            , count , ep_status, avgOverAllDomains

        completionBonus = 0

        count += 1
        #print(count)
        #print("Action: " + str(action))
        #action will be probability distribution for blend space
        #one step is a full trajectory with given blending dist.

        obs = self.ctrl.set_action(action)
        # obs = ctrl.set_action([mu, sigma])

        #print("Env Step" + str(count))
        done = False

        if (self.ctrl.isAtPos(goals[self.current])):
            if self.current < steps - 1:
                self.current += 1
            else:
                self.current += 0

            if (self.current < steps - 1):
                self.ctrl.update_target(goals[self.current], safe_region[self.current-1])
                if(self.current < steps-1):
                    #gui_object.updatePathToGoal()
                    pass
            else:

                self.stableAtGoal += 1
                #if(self.stableAtGoal > 50):
                done = True
                #    if self.ctrl.isDone():
                        #didnt leave safe area
                #        print("inside safe bounds for whole ep fault size " + str(maxFault))
                #        completionBonus = 100 + (100 *maxFault)

                # else:
                #
                #     done = False
        else:
            self.stableAtGoal = 0


        reset = True

        # Null reward everywhere except when reaching the goal (left of the grid)
        reward = self.ctrl.getReward()
        failed =False
        if reward == -0.1:
            done = True
            failed = True
            reward = 0

            print("Ep failed : to much deviation")

        reward += completionBonus
        self.cumu_reward += reward


        if(done):
            #print(obs)

            #total time outside is 0 - nothing to learn
            stayedInsideSafetyBound = self.ctrl.isDone()
            ep_status = 0
            rewardBonus = 0
            if stayedInsideSafetyBound:
                #fault not significant enough to leave safebound
                rewardBonus = 0
                ep_status = 0
            elif failed:
                rewardBonus = 0 # total reward is the applied for each negative step outside
                # - no extra negative for failing
                ep_status = 2
            else:
                # cancel out the negative rewards and give large positive - proportional to the time spent outside.
                rewardBonus =  2 * self.ctrl.getTotalTimeOutside()
                ep_status = 1

            reward += rewardBonus
            print("Ep done| " + Type + " Level " + str(Level+1) + " Accumulated Reward = " + str(self.cumu_reward)
                 + " with " + str(self.ctrl.getTotalSteps()) + " steps")

            # if len(last_rewards) > convergence_length:
            #     last_rewards.pop(0)
            #
            allDomainPerf[TypeInd][Level].append((self.ctrl.getTotalTimeOutside()))

            #print(allDomainPerf)

            for t in range(4):
                for lev in range(4):
                    AvgLength = 20

                    if len(allDomainPerf[t][lev]) > AvgLength:
                        allRes = allDomainPerf[t][lev][-AvgLength:]
                        #print(allRes)
                    else:
                        allRes = allDomainPerf[t][lev]
                    if allRes == []:
                        avg = 2000
                    else:
                        avg = (np.average(allRes))
                    #print(int(avg))
                    avgDomainPerf[t][lev] = int(avg)

            totalNumberRuns += 1
            #cumulativePerformance = np.sum( last_rewards)
            #allCumulativeRewards.append(cumulativePerformance)
            allRuns.append(totalNumberRuns)

            avgOverAllDomains.append(np.sum(avgDomainPerf[0]) +np.sum(avgDomainPerf[1])+np.sum(avgDomainPerf[2])+ np.sum(avgDomainPerf[3]))
            updateRewardChart()


        # Optionally we can pass additional info, we are not using that for now
        info = {"CR" : self.cumu_reward}
        #self.render()
        return np.array(obs), reward, done, info

    def render(self,mode="console"):
        if mode != 'console':
            raise NotImplementedError()
        # #agent is represented as a cross, rest as a dot
        # self.gui_object.quads[self.quad_id]['position'] = self.quad.get_position(self.quad_id)
        # self.gui_object.quads[self.quad_id]['orientation'] = self.quad.get_orientation(self.quad_id)
        # self.gui_object.update()


    def close(self):
        pass

    def checkConverged(self):

        if len(last_rewards) < convergence_length:
            return False
        for rew in last_rewards:
            if rew < reward_threshold:
                return False

        return True

    def setRandomFault(self):
        global totalNumberRuns,Type,Level,avgDomainPerf,domainProb,TypeInd, domainProbDist

        faultType = "None"
        #===============UNIFORM DOMAIN RANDOMIZATION
        np.random.seed(int(time.time()))
        #domain = np.random.randint(1, 5)


        #==============Guided Randomization
        #get average performance over entire domain -
        #print(avgDomainPerf.values())
        TotalAveragePerf = 0
        allAvg = []
        for i in range(len(avgDomainPerf)):
            type = avgDomainPerf[i]
            allAvg.append([])
            for j in range(len(type)):
                TotalAveragePerf += max( abs(type[j]), 100)
                allAvg[i].append(max( abs(type[j]), 100))

        domainProbDist = []
        for i in range(len(avgDomainPerf)):
            type = avgDomainPerf[i]
            #domainProb.append([])
            for j in range(len(type)):
                domainProbDist.append(allAvg[i][j] / TotalAveragePerf)


        #print(TotalAveragePerf)
       # print(allAvg)

       #  for avg in allAvg:
       #      domainProb.append(avg/TotalAveragePerf)
        # for i in range(len(avgDomainPerf)):
        #     type = list(avgDomainPerf.values())[0]
        #     for j in range(len(type)):
        #         domainProb.append( max( (type[j] / TotalAveragePerf) , 0.0125))

       # print(domainProb)
        #print(np.arange(0, 16))
        #domainSelect = np.random.choice(np.arange(0, 16), p=domainProbDist)


        #// create 16 position array each containing the probability of sampling from that domain.
            #probability is linked to performance in domain.
            # avgPerf/Total

        # domainSelect, mod4
        # domain select / 4 = domain

        TypeInd = np.random.choice([0,1,2,3]) #represents the index in the matrix for a given fault 0=rotor , 1=wind, 2= pos , 3=att
        magnitude = np.random.randint(0,3) # represents magnitude of fault 0 smallest, 3 largest
        #print(magnitude)


        if(TypeInd == 0 ):
            faultType = "Rotor"

        elif(TypeInd == 1 ):
            faultType = "AttNoise"

        elif(TypeInd == 2 ):
            faultType = "PosNoise"

        else:
            faultType = "Wind"

        # print(TypeInd)
        #
        #
        # print(faultType)
        # print(magnitude)
        #print(faultType + str(magnitude))

        # magnitude = np.random.randint(0, numLevels) +1



        numLevels = 4


        # Random Rotor Selection
        # Fixed Start time and permanent
        # magnitude varies
        if(faultType == "Rotor"):
            #values = 0.05 , 0.1 , 0.15, 0.20
            faults = [0, 0, 0, 0]
            rotor = np.random.randint(0, 4)

            #magnitude = 4 # np.random.randint(0, numLevels) + 1

            Level = magnitude
            fault_mag = magnitude * 0.05
            starttime = 300
            endtime = 31000

            faults[rotor] = fault_mag
            #print(faults)
            self.ctrl.setMotorFault(faults)
            self.ctrl.setFaultTime(starttime, endtime)


        # Dryden Distrubance model Random + nominal part
        # Random Nominal portion Direction -X,+X,-Y,+Y
        # Random portion doesnt vary
        if (faultType == "Wind"):
            #TODO set nominal and random wind in controller
            # values = 3, 6, 9, 12
            # Direction = -X, +X, -Y, +Y

            direction = np.random.randint(0, 4)
           # magnitude = 4
            WindMag = 3 * magnitude
            Level = magnitude
            if (direction == 0):
                winds = [-WindMag , 0 ,  0]
            elif (direction == 1):
                winds = [WindMag, 0, 0]
            elif (direction == 2):
                winds = [0, -WindMag, 0]
            else:
                winds = [0, WindMag, 0]

            #print(winds)
            self.ctrl.setNormalWind(winds)

        #single variable of magnitude
        #uniform random noise from -noise to +noise
        if(faultType == "PosNoise"):
            #values = 1, 2, 3, 4

          #  magnitude = 4
            Level = magnitude
            self.ctrl.setSensorNoise(magnitude)
            #print(noise)
        # single variable of magnitude
        # uniform random noise from -noise to +noise
        if(faultType == "AttNoise"):
            #values = 0.3, 0.6, 0.9, 1.2

            #magnitude =4
            attNoise = 0.2 * magnitude
            Level = magnitude
            self.ctrl.setAttitudeSensorNoise(attNoise)
            #print(attNoise)


        Type = faultType

        self.ctrl.setFaultMode(faultType)



    def getTotalTimeOutside(self):
        return  self.ctrl.total_time_outside_safety


def updateRewardChart():
   # if(totalNumberRuns % logInterval == 0):
    if(totalNumberRuns % 50 == 0):
        TotalAveragePerf = 0
        allAvg = []
        AvgP = []
        numEps = []

        for i in range(len(avgDomainPerf)):
            type = avgDomainPerf[i]
            AvgP.append([])
            numEps.append([])
            allAvg.append([])
            for j in range(len(type)):
                TotalAveragePerf += min( max(abs(type[j]), 200) , 750)
                allAvg[i].append( min(max(abs(type[j]), 200) , 750)  )
                AvgP[i].append(type[j])
                numEps[i].append( len(allDomainPerf[i][j]) )

        # print(TotalAveragePerf)
        #print(allAvg)
        domainProb = []
        #colours = []
        for i in range(len(avgDomainPerf)):
            type = avgDomainPerf[i]
            domainProb.append([])
            #colours.append([])
            for j in range(len(type)):
                domainProb[i].append(allAvg[i][j] / TotalAveragePerf)
                #colours[i].append( )

        # for avg in allAvg:
        #     domainProb.append(avg / TotalAveragePerf)


        # ax.cla()
        ax1.cla()

        #AvgP = [ avgDomainPerf["Rotor"] , avgDomainPerf["Wind"], avgDomainPerf["PosNoise"], avgDomainPerf["AttNoise"] ]

        #print(AvgP)
        #print(domainProb)
        plotP = [AvgP[0], AvgP[3]]

        x = np.arange(0.5, 5.5, 1)  # len = 5 level
        y = np.arange(0.5, 3.5, 1)

        labelPosX = [1, 2, 3, 4]
        labelPosY = [1, 2 ]
        x_label = ["Level 1", "Level 2", "Level 3", "Level 4"]
        y_label = ["Rotor", "Att Noise"]
        # 5 types of domains
        # for txt in fig.texts:
        #     txt.set_visible(False)
        ind = -1
       # AvgP = [[500, 200, 300, 400], [550, 200, 300, 400], [600, 200, 300, 400], [700, 200, 300, 400]]
        #print("Plotting")
        for i in range(len(x_label)):

            for j in range(len(y_label)):
                ind+=1
                x_p = x[i] +0.4
                y_p = y[j] +0.4

               # ax.text(x_p, y_p, str(AvgP[j][i]), fontsize=12)
                x_p2 = x[i] + 0.4
                y_p2 = y[j] + 0.2
               # ax.text(x_p2, y_p2, "#ep:" +str(numEps[j][i]), fontsize=8)
                #x_p3 = x[i] + 0.4
                #y_p3 = y[j] + 0.6
                #plt.text(x_p3, y_p3, str(int(domainProb[j][i]*100))+"%" , fontsize=8)


        cmap = plt.get_cmap('YlGn').reversed()

        #ax.pcolormesh(x, y, plotP, cmap=cmap , vmin=0, vmax=1000, edgecolors="black")
        #ax.set_xticks(labelPosX, x_label)
        #ax.set_yticks(labelPosY, y_label)
        eps = np.arange(0,totalNumberRuns)
        ax1.plot(eps, avgOverAllDomains )
       # goodPer = np.ones(totalNumberRuns)*1000
       # ax1.plot(eps, goodPer)


        ax1.set_title("Average Performance")
        ax1.set_ylabel("Average Time outside Safezone")
        ax1.set_xlabel("Episodes")

       # ax.set_xlabel("Fault Magnitude")
      #  ax.set_ylabel("Domains")
       # ax.set_title("Performance Over Domain Space")

        plt.draw()
        plt.pause(0.001)
        plt.savefig("Training/"+str(totalNumberRuns)+".png")
        #np.savetxt('PolicyRandomization.csv', allDomainPerf, delimiter=',')


def generateRandomPath():

    global totalNumberRuns, stepsToGoal, steps, x_dest, y_dest, x_path, y_path, z_path, goals, safe_region, limit

    np.random.seed(totalNumberRuns)
    stepsToGoal = 0

    limit = 10
    x_dest = np.random.randint(-limit, limit)
    y_dest = np.random.randint(-limit, limit)
    z_dest = np.random.randint(5, limit)
    steps = 4
    # x_path = [0, 0, 5, 0, -5, 0, 5, 5]
    # y_path = [0, 0, 0, 5, 0, -5, 0, 0]
    # z_path = [0, 5, 5, 5, 5, 5, 5, 5]
    x_path = [0, 0, x_dest, x_dest]
    y_path = [0, 0, y_dest, y_dest]
    z_path = [5, 5, z_dest, z_dest]
    interval_steps = 50
    yaws = np.zeros(steps)
    goals = []
    safe_region = []
    #print("Goal : " + str(x_dest) + " , " + str(y_dest))
    for i in range(steps):
        if(i < steps-1 ):
            #create linespace between waypoint i and i+1
            x_lin = np.linspace(x_path[i], x_path[i+1], interval_steps)
            y_lin =  np.linspace(y_path[i], y_path[i+1], interval_steps)
            z_lin =  np.linspace(z_path[i], z_path[i+1], interval_steps)
        else:
            x_lin = np.linspace(x_path[i], x_path[i], interval_steps)
            y_lin = np.linspace(y_path[i], y_path[i], interval_steps)
            z_lin = np.linspace(z_path[i], z_path[i], interval_steps)

        goals.append([x_path[i], y_path[i], z_path[i]])
        #for each pos in linespace append a goal
        safe_region.append([])
        for j in range(interval_steps):
            safe_region[i].append([x_lin[j], y_lin[j], z_lin[j]])
            stepsToGoal +=1
   # print(goals)
