import numpy as np
import math
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as Axes3D
import sys

class GUI():
    # 'quad_list' is a dictionary of format: quad_list = {'quad_1_name':{'position':quad_1_position,'orientation':quad_1_orientation,'arm_span':quad_1_arm_span}, ...}
    def __init__(self, quads, ctrl):
        self.quads = quads
        self.pos = []
        self.ctrl = ctrl
        self.fig = plt.figure()
        self.ax = Axes3D.Axes3D(self.fig)
        self.ax.set_xlim3d([-10.0, 10.0])
        self.ax.set_xlabel('X')
        self.ax.set_ylim3d([-10.0, 10.0])
        self.ax.set_ylabel('Y')
        self.ax.set_zlim3d([0, 10.0])
        self.ax.set_zlabel('Z')
        self.ax.set_title('Quadcopter Simulation')
        self.init_plot()
        self.currentPoints = []
        self.fig.canvas.mpl_connect('key_press_event', self.keypress_routine)
        self.scatter = 0
    def rotation_matrix(self,angles):
        ct = math.cos(angles[0])
        cp = math.cos(angles[1])
        cg = math.cos(angles[2])
        st = math.sin(angles[0])
        sp = math.sin(angles[1])
        sg = math.sin(angles[2])
        R_x = np.array([[1,0,0],[0,ct,-st],[0,st,ct]])
        R_y = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
        R_z = np.array([[cg,-sg,0],[sg,cg,0],[0,0,1]])
        R = np.dot(R_z, np.dot( R_y, R_x ))
        return R

    def init_plot(self):
        for key in self.quads:
            self.quads[key]['l1'], = self.ax.plot([],[],[],color='blue',linewidth=3,antialiased=False)
            self.quads[key]['l2'], = self.ax.plot([],[],[],color='red',linewidth=3,antialiased=False)
            self.quads[key]['hub'], = self.ax.plot([],[],[],marker='o',color='green', markersize=6,antialiased=False)

    def update(self):
        for key in self.quads:
            R = self.rotation_matrix(self.quads[key]['orientation'])
            L = self.quads[key]['L']
            points = np.array([ [-L,0,0], [L,0,0], [0,-L,0], [0,L,0], [0,0,0], [0,0,0] ]).T
            points = np.dot(R,points)
            points[0,:] += self.quads[key]['position'][0]
            points[1,:] += self.quads[key]['position'][1]
            points[2,:] += self.quads[key]['position'][2]
            self.quads[key]['l1'].set_data(points[0,0:2],points[1,0:2])
            self.quads[key]['l1'].set_3d_properties(points[2,0:2])
            self.quads[key]['l2'].set_data(points[0,2:4],points[1,2:4])
            self.quads[key]['l2'].set_3d_properties(points[2,2:4])
            self.quads[key]['hub'].set_data(points[0,5],points[1,5])
            self.quads[key]['hub'].set_3d_properties(points[2,5])
            self.pos.append([self.quads[key]['position'][0],
                             self.quads[key]['position'][1],
                             self.quads[key]['position'][2]])
        self.updateLine()

       # self.showPathToGoal()
       # self.addMinDistLine()
        plt.pause(0.000000000000001)

    def addMinDistLine(self):
        p2 = self.ctrl.getLatestMinDistPoint()

        x_c1 = [p2[0] , self.pos[-1][0]]
        y_c1 = [p2[1], self.pos[-1][1]]
        z_c1 = [p2[2], self.pos[-1][2]]
        c = "k"
        if self.ctrl.getLatestMinDist() > 1:
            c = "r"
        self.ax.plot3D(x_c1, y_c1, z_c1, linewidth=1, c=c)

    def updateLine(self):
        x_c1 = []
        y_c1 = []
        z_c1 = []

        for i in range(len(self.pos)):
            x_c1.append(self.pos[i][0])
            y_c1.append(self.pos[i][1])
            z_c1.append(self.pos[i][2])

        self.ax.plot3D(x_c1,y_c1,z_c1,linewidth=1, c="b")

    def showPathToGoal(self):

        path = self.ctrl.getCurrentSafeBounds()
        ind = 0
        for i in range(len(path)):
            pos = path[i]
            ind += 1
            self.ax.scatter(pos[0], pos[1], pos[2], marker='o', color='green', alpha=0.05, linewidth=50)
            if ind == (len(path) - 1):
                self.ax.scatter(pos[0], pos[1], pos[2], marker='o', color='black', alpha=0.5, linewidth=50)


    def show(self):
        x_c1 = []
        y_c1 = []
        z_c1 = []

        for i in range(len(self.pos)):
            x_c1.append(self.pos[i][0])
            y_c1.append(self.pos[i][1])
            z_c1.append(self.pos[i][2])

        self.ax.plot3D(x_c1, y_c1, z_c1, linewidth=1, c="b")

    def keypress_routine(self,event):
        sys.stdout.flush()
        if event.key == 'x':
            y = list(self.ax.get_ylim3d())
            y[0] += 0.2
            y[1] += 0.2
            self.ax.set_ylim3d(y)
        elif event.key == 'w':
            y = list(self.ax.get_ylim3d())
            y[0] -= 0.2
            y[1] -= 0.2
            self.ax.set_ylim3d(y)
        elif event.key == 'd':
            x = list(self.ax.get_xlim3d())
            x[0] += 0.2
            x[1] += 0.2
            self.ax.set_xlim3d(x)
        elif event.key == 'a':
            x = list(self.ax.get_xlim3d())
            x[0] -= 0.2
            x[1] -= 0.2
            self.ax.set_xlim3d(x)
    def close(self):

        plt.close(self.fig)