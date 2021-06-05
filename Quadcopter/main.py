import quadcopter,controller
import numpy as np
import time
import gui

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


fault_mag = 0.15
starttime = 300
n = 1
avgDomainPerf1 = {"None": [0], "Rotor": [0,0,0,0] , "Wind": [0,0,0,0] ,
                 "PosNoise" : [0,0,0,0] , "AttNoise": [0,0,0,0]}

allDomainPerf1 = {"None": [], "Rotor": [[],[],[],[]] , "Wind": [[],[],[],[]] ,
                 "PosNoise" : [[],[],[],[]] , "AttNoise": [[],[],[],[]]}




Type ="None"
Level = 0


controller1 = []

controller2 = []

steps1 = []
steps2 = []
total_steps = [ ]

trajectories = []
trajectories2 = []
trajectories3 = []


stepsToGoal = 0
steps = 7
limit = 7

yaws = np.zeros(steps)
goals = []
safe_region = []

def setRandomPath():
    global stepsToGoal, steps, x_dest, y_dest, x_path, y_path, z_path, goals, safe_region, limit
    millis = int(round(time.time()))

    np.random.seed(millis)


    limit = 10
    x_dest = np.random.randint(-limit, limit)
    y_dest = np.random.randint(-limit, limit)
    z_dest = np.random.randint(5, limit)
    steps = 4
    x_path = [0, 0, x_dest, x_dest]
    y_path = [0, 0, y_dest, y_dest]
    z_path = [5, 5, z_dest, z_dest]

    # ===================================================


    interval_steps = 50
    yaws = np.zeros(steps)
    goals = []
    safe_region = []
    # print("Goal : " + str(x_dest) + " , " + str(y_dest))
    for i in range(steps):
        if (i < steps - 1):
            # create linespace between waypoint i and i+1
            x_lin = np.linspace(x_path[i], x_path[i + 1], interval_steps)
            y_lin = np.linspace(y_path[i], y_path[i + 1], interval_steps)
            z_lin = np.linspace(z_path[i], z_path[i + 1], interval_steps)
        else:
            x_lin = np.linspace(x_path[i], x_path[i], interval_steps)
            y_lin = np.linspace(y_path[i], y_path[i], interval_steps)
            z_lin = np.linspace(z_path[i], z_path[i], interval_steps)

        goals.append([x_path[i], y_path[i], z_path[i]])
        # for each pos in linespace append a goal
        safe_region.append([])
        for j in range(interval_steps):
            safe_region[i].append([x_lin[j], y_lin[j], z_lin[j]])
            stepsToGoal += 1
        #print(goals)

#for DType in ["Rotor", "Wind", "PosNoise", "AttNoise"]:
for DType in ["Rotor"]:

    # for mag in [1,2,3,4]:
        mag = 1
        control = "Dirichlet"
        for i in range(n):
            quad_id = i

            QUADCOPTER = {
                str(quad_id): {'position': [0, 0, 0], 'orientation': [0, 0, 0], 'L': 0.3, 'r': 0.1,
                               'prop_size': [10, 4.5],
                               'weight': 1.2}}

            # Make objects for quadcopter

            quad = quadcopter.Quadcopter(QUADCOPTER)
            Quads = {str(quad_id): QUADCOPTER[str(quad_id)]}

            # create blended controller and link it to quadcopter object
            ctrl = controller.Blended_PID_Controller(quad.get_state, quad.get_time,
                                                     quad.set_motor_speeds, quad.get_motor_speeds,
                                                     quad.stepQuad, quad.set_motor_faults, quad.setWind,
                                                     quad.setNormalWind,
                                                     params=BLENDED_CONTROLLER_PARAMETERS, quad_identifier=str(quad_id))

            gui_object = gui.GUI(quads=Quads, ctrl=ctrl)

            setRandomPath()
            current = 0
            ctrl.update_target(goals[current], safe_region[current])

            faultType = "None"
            np.random.seed(int(time.time()))

            # Random Rotor Selection
            # Fixed Start time and permanent
            # magnitude varies
            if (DType == "Rotor"):
                # values = 0.05 , 0.1 , 0.15, 0.20
                faults = [0, 0, 0, 0]
                rotor = np.random.randint(0, 4)
                #magnitude = np.random.randint(0, numLevels) + 1
                magnitude = mag # np.random.randint(0, numLevels) + 1
                # rotor = 1
                Level = magnitude
                fault_mag = magnitude * 0.05
                starttime =300
                endtime = 31000
                faultType = "Rotor"
                faults[rotor] = fault_mag
                # print(faults)
                ctrl.setMotorFault(faults)
                ctrl.setFaultTime(starttime, endtime)

            # Dryden Distrubance model Random + nominal part
            # Random Nominal portion Direction -X,+X,-Y,+Y
            # Random portion doesnt vary
            if (DType == "Wind"):
                WindScalar = 5
                WindMag = magnitude * WindScalar
                direction = np.random.randint(0, 4)
                # magnitude = 4
                if (direction == 0):
                    winds = [-WindMag, 0, 0]
                elif (direction == 1):
                    winds = [WindMag, 0, 0]
                elif (direction == 2):
                    winds = [0, -WindMag, 0]
                else:
                    winds = [0, WindMag, 0]

                faultType = "Wind"
                Level = magnitude

                ctrl.setNormalWind(winds)

                # ============= Noise config===============

            if (DType == "PosNoise") :
                faultType ="PosNoise"
                Level  = magnitude
                ctrl.setSensorNoise(Level)

            # single variable of magnitude
            # uniform random noise from -noise to +noise
            if (DType == "AttNoise"):
                # values = 0.3, 0.6, 0.9, 1.2
                magnitude = mag
                # magnitude =4
                attNoise = 0.3 * magnitude
                Level = magnitude
                ctrl.setAttitudeSensorNoise(attNoise)
                # print(attNoise)
                faultType = "AttNoise"

            Type = faultType
            # print(Type)
            # print(magnitude)
            ctrl.setFaultMode(faultType)

            ctrl.setController(control)
            done = False
            stepcount = 0
            stableAtGoal = 0
            failed = False
            cumu_reward = 0
            obs = ctrl.get_updated_observations()



            while not done:
                print(stepcount)
                stepcount += 1

                obs = ctrl.step()
                rew = ctrl.getReward()

                if (stepcount % 20 == 0):
                    gui_object.quads[str(quad_id)]['position'] = [obs[0], obs[1], obs[2]]
                    gui_object.update()



                if (rew != -0.1):
                    cumu_reward += rew
                else:
                    done = True

                    failed = True

                if (ctrl.isAtPos(goals[current])):

                    if current < steps - 1:
                        current += 1
                    else:
                        current += 0

                    if (current < steps - 1):
                        ctrl.update_target(goals[current], safe_region[current - 1])
                        if (current < steps - 1):
                            #gui_object.updatePathToGoal()
                            pass
                    else:

                        stableAtGoal += 1
                        if (stableAtGoal > 100):

                            done = True
                        else:

                            done = False

                else:
                    stableAtGoal = 0






