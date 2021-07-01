import quadcopter,controller
import numpy as np
import time
import gui

# --------------------------------------------------------
# MAIN CONFIGURATION OF BLENDED QUADCOPTER CONTROLLER

# The angular PID controller tuning is the most interesting. e.g.
# 'Angular_PID': {'P': [4000, 4000, 1500], 'I': [0, 0, 1.2], 'D': [1500, 1500, 0]},
#  This corresponds to a:
#  Roll PID config of [4000, 0, 1500] - first element of each list
#  Pitch PID config of [4000, 0, 1500]- second element of each list
#  Yaw   PID config of [1500, 1.2, 0]- third element of each list
#  - similarly for PID position controllers, remaining parameters should stay fixed.

#  Angular_PID index defines the first set of PID controllers to use - for example for nominal conditions
#  Angular_PID2 index defines the fault PID configuration - can be tuned to any fault condition.
#  ... for a larger set of controllers simply extend this set.

#  IMPORTANT : For blended control to be effective there must be two DIFFERENTLY tuned controllers
#  Position Blending is implemented but disabled as the SAME controller tuning is used.
# --------------------------------------------------------

BLENDED_CONTROLLER_PARAMETERS = {'Motor_limits': [0, 9000],
                         'Tilt_limits': [-10, 10],
                         'Yaw_Control_Limits': [-900, 900],
                         'Z_XY_offset': 500,
                         'Linear_PID': {'P': [300, 300, 7500], 'I': [0.04, 0.04, 5], 'D': [450, 450, 5400]},
                         'Linear_PID2': {'P': [300, 300, 7500], 'I': [0.04, 0.04, 5], 'D': [450, 450, 5400]},
                         'Linear_To_Angular_Scaler': [1, 1, 0],
                         'Yaw_Rate_Scaler': 0.18,
                         'Angular_PID': {'P': [24000, 24000, 1500], 'I': [0, 0, 1.2], 'D': [12000, 12000, 0]},
                         'Angular_PID2': {'P': [4000, 4000, 1500], 'I': [0, 0, 1.2], 'D': [1500, 1500, 0]},
                         }



#number of runs of each environment.
n = 1


Type ="None"
Level = 0

controller1 = []
steps1 = []
total_steps = [ ]

trajectories = []
stepsToGoal = 0
steps = 0
limit = 0
goals = []
safe_region = []

def setDiamondPath():
    global stepsToGoal, steps, x_dest, y_dest, x_path, y_path, z_path, goals, safe_region, limit

    x_path = [0, 0, 5, 0, -5, 0, 5, 5]
    y_path = [0, 0, 0, 5, 0, -5, 0, 0]
    z_path = [0, 5, 5, 5, 5, 5, 5, 5]
    steps = len(x_path)
    interval_steps = 50
    goals = []
    safe_region = []

    #create line of points between all waypoints in the trajectory.
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

        #defines the centre of the spheres used for safe zone calculation
        for j in range(interval_steps):
            safe_region[i].append([x_lin[j], y_lin[j], z_lin[j]])
            stepsToGoal += 1


#=======================================
#====      MAIN SIMULATION LOOP     ====
#=======================================


# Set the fault condition of interest - if no fault mode needed use "" instead.
# Several fault modes can be set together using a list of strings.

#for DType in ["Rotor", "Wind", "PosNoise", "AttNoise"]:
for DType in ["Rotor"]:


    # for mag in [1,2,3,4]:
        mag = 1




        #control = "C1" # use the first controller only
        #control = "C2"  # use the second controller only
        control = "Uniform" # use uniform randomized blended control between 2 controllers
        #control = "Dirichlet"  # randomized blended control between N controllers using dirichlet dist
        #control = "Agent"  # let a RL agent parameterize the distribution to use

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

            setDiamondPath()
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
                starttime = 300
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

            #-----------------------------------------------------
            #this while loop actually steps through the simulation.
            # -----------------------------------------------------
            while not done:
                #print(stepcount)
                stepcount += 1

                # -----------------------------------------------------
                # this line steps the controller object which in turn
                # steps the quadcopter object. The obs vector contains
                # the current state of the quadcopter.
                obs = ctrl.step()
                # -----------------------------------------------------
                #
                
                if (stepcount % 20 == 0):
                    gui_object.quads[str(quad_id)]['position'] = [obs[0], obs[1], obs[2]]
                    gui_object.update()



                # -----------------------------------------------------
                rew = ctrl.getReward()
                # Below code is relevant for showing the GUI and checking
                # if the last waypoint has been reached (sim run ends).

                # a reward of -0.1 is used to halt the simulation from
                # the controller file based on some criteria such as too many steps
                # outside of the safe zone.
                #
                # NOTE: this can be useful even without training.
                if (rew != -0.1):
                    cumu_reward += rew
                else:
                    # Quadcopter failed - end of run.
                    done = True
                    failed = True

                if (ctrl.isAtPos(goals[current])):

                    if current < steps - 1:
                        current += 1
                    else:
                        current += 0

                    # -----------------------------------------------------
                    # Check if the another waypoint exists on the trajectory
                    # if so update the next goal otherwise stabilize on the goal and end.
                    if (current < steps - 1):
                        ctrl.update_target(goals[current], safe_region[current - 1])
                        if (current < steps - 1):
                            #gui_object.updatePathToGoal()
                            pass
                    else:

                        stableAtGoal += 1
                        if (stableAtGoal > 100):
                            #Goal position has been reached - end of run.
                            done = True
                        else:

                            done = False

                else:
                    stableAtGoal = 0






