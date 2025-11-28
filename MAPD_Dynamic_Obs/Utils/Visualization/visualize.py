#!/usr/bin/env python3
import yaml
import matplotlib
# matplotlib.use("Agg")
from matplotlib.patches import Circle, Rectangle, Arrow, RegularPolygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
import matplotlib.animation as manimation
import argparse
import math
import json
import os
import pickle
import RoothPath


Colors = ['orange', 'blue', 'green']


class Animation:
    def __init__(self, map, schedule, slow_factor=10):
        self.map = map
        self.schedule = schedule
        self.slow_factor = slow_factor
        self.combined_schedule = {}
        self.combined_schedule.update(self.schedule["schedule"])
        self.dynamic_obstacles = schedule['dyn_ob_path']

        aspect = map["map"]["dimension"][0] / map["map"]["dimension"][1]

        self.fig = plt.figure(frameon=False, figsize=(5 * aspect, 5))
        self.ax = self.fig.add_subplot(111, aspect='equal')
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=None, hspace=None)
        self.ax.set_title(f"Time: {0:.2f} seconds")

        self.patches = []
        self.artists = []
        self.agents = dict()
        self.agent_names = dict()
        self.tasks = dict()
        self.dyn_ob = []
        # Create boundary patch
        xmin = -0.5
        ymin = -0.5
        xmax = map["map"]["dimension"][0] - 0.5
        ymax = map["map"]["dimension"][1] - 0.5

        # self.ax.relim()
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)

        # self.dynamic_obstacles = [[43,1],[22,1]]
        # self.dynamic_ob = Rectangle((22 - 0.5, 5 - 0.5), 1, 1, facecolor='black', edgecolor='black', alpha= 0)
        # self.patches.append(self.dynamic_ob)

        self.patches.append(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, facecolor='none', edgecolor='red'))
        for x in range(map["map"]["dimension"][0]):
            for y in range(map["map"]["dimension"][1]):
                self.patches.append(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor='none', edgecolor='none'))
        for o in map["map"]["obstacles"]:
            x, y = o[0], o[1]
            self.patches.append(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor='black', edgecolor='black'))
            # if (x,y) in [(4,8), (4,9), (4,10), (5,8), (5,10), (4,12), (4,13), (4,14), (5,12), (5,14), (4,26), (4,27), (4,28), (5,26), (5,28), (4,30), (4,31), (4,32), (5,30), (5,32)]:
            #     continue
            # if y not in [11,9,10,12,13] or x not in [64,65,52,53,39,40,27,28,22,21,34,33,47,46,59,58,23,24,25,26,35,36,37,38,48,49,50,51,60,61,62,63]:
            #     self.patches.append(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor='black', edgecolor='black'))
        # for o in [21.5, 33.5, 46.5, 58.5]:
        #     xy = (o,9.5)
        #     self.patches.append(Rectangle(xy, 0.2, 4, facecolor='black', edgecolor='black'))
        # for o in [27.5, 39.5, 52.5, 64.5]:
        #     xy = (o-0.2,9.5)
        #     self.patches.append(Rectangle(xy, 0.2, 4, facecolor='black', edgecolor='black'))

        # for o in [24, 36, 49, 61]:
        #     self.patches.append(Rectangle((o+0.15, 9.5), 0.7, 4, facecolor='black', edgecolor='black'))

        for e in map["map"]["non_task_endpoints"]:
            x, y = e[0], e[1]
            self.patches.append(Circle((x, y), 0.4, facecolor='green', edgecolor='black'))

        task_colors = np.random.rand(len(map["tasks"]), 3)
        for t, i in zip(map["tasks"], range(len(map["tasks"]))):
            x_s, y_s = t['waypoints'][0][0], t['waypoints'][0][1]
            self.tasks[t['task_name']] = [Rectangle((x_s - 0.25, y_s - 0.25), 0.5, 0.5, facecolor=task_colors[i], edgecolor='black', alpha=0)]
            self.patches.append(self.tasks[t['task_name']][0])
        
        for t, i in zip(map["tasks"], range(len(map["tasks"]))):
            for j in range(1, len(t['waypoints'])-1):
                x_i, y_i = t['waypoints'][j][0], t['waypoints'][j][1]
                self.tasks[t['task_name']].append(RegularPolygon(xy=(x_i, y_i - 0.05), numVertices=3, radius=0.2, facecolor=task_colors[i], edgecolor='black', alpha=0))
                self.patches.append(self.tasks[t['task_name']][-1])

        for t, i in zip(map["tasks"], range(len(map["tasks"]))):
            x_g, y_g = t['waypoints'][-1][0], t['waypoints'][-1][1]
            self.tasks[t['task_name']].append(RegularPolygon(xy=(x_g, y_g - 0.05), numVertices=3, radius=0.2, facecolor=task_colors[i], edgecolor='black', alpha=0))
            self.patches.append(self.tasks[t['task_name']][-1])

        # Create agents:
        self.T = 0
        # Draw goals first
        for d, i in zip(map["agents"], range(0, len(map["agents"]))):
            if 'goal' in d:
                self.patches.append(
                    Rectangle((d["goal"][0] - 0.25, d["goal"][1] - 0.25), 0.5, 0.5, facecolor=Colors[0], edgecolor='black',
                              alpha=0.5))
        for d, i in zip(map["agents"], range(0, len(map["agents"]))):
            name = d["name"]
            self.agents[name] = Circle((d["start"][0], d["start"][1]), 1.2, facecolor=Colors[0], edgecolor='black')
            self.agents[name].original_face_color = Colors[0]
            self.patches.append(self.agents[name])
            self.T = max(self.T, schedule["schedule"][name][-1]["t"])
            self.agent_names[name] = self.ax.text(d["start"][0], d["start"][1], name.replace('agent', ''))
            self.agent_names[name].set_horizontalalignment('center')
            self.agent_names[name].set_verticalalignment('center')
            self.artists.append(self.agent_names[name])
        
        for i in range(len(self.dynamic_obstacles)):
            path = self.dynamic_obstacles['obstacle'+str(i)]
            patch = Circle((path[0]['x'], path[0]['y']), 0.45, facecolor='red', edgecolor='black')
            self.dyn_ob.append(patch)
            self.patches.append(patch)


        self.anim = animation.FuncAnimation(self.fig, self.animate_func,
                                            init_func=self.init_func,
                                            frames=int(self.T + 1) * self.slow_factor,
                                            interval=10,
                                            blit=False,
                                            repeat=False)

    def save(self, file_name, speed):
        self.anim.save(
            file_name,
            "ffmpeg",
            fps=10 * speed,
            dpi=200),
        # savefig_kwargs={"pad_inches": 0, "bbox_inches": "tight"})

    def show(self):
        plt.show()
    
    def init_func(self):
        # if self.layout == 2:
        #     paths = [[(58,7), (61,7)], [(7,7), (7,9)], [(7,9), (7,7)], [(7,15), (7,13)], [(7,13), (7,15)], [(9.0, 31.0), (7.0, 31.0), (5, 31)], [(5, 31), (7.0, 31.0), (9.0, 31.0)], [(7, 13), (5, 13)], [(5, 13), (7, 13)], [(9.0, 31.0), (9.5, 28.25), (10.0, 25.5), (10.5, 22.75), (11.0, 20.0), (11.5, 17.25), (12.0, 14.5), (12.5, 11.75), (13, 9)], [(10.0, 11.0), (7, 13)], [(7, 13), (10.0, 11.0)], [(10.0, 11.0), (11.5, 10.0), (13, 9)], [(13, 9), (15, 7)], [(5, 27), (7, 27)], [(7, 27), (5, 27)], [(5, 9), (7, 9)], [(7, 9), (5, 9)], [(10.0, 11.0), (9.5, 13.67), (9.0, 16.33), (8.5, 19.0), (8.0, 21.67), (7.5, 24.33), (7, 27)], [(11.0, 7.0), (9.0, 8.0), (7, 9)], [(11.0, 7.0), (10, 11)], [(16.0, 5.0), (13.5, 6.0), (11, 7)], [(21, 7), (23, 9), (23, 12)], [(26, 12), (26, 9)], [(26, 9), (26, 12)], [(26, 9), (24, 7)], [(24, 7), (26, 9)], [(24, 7), (22, 5)], [(33, 7), (35, 9), (35, 12)], [(38, 12), (38, 9)], [(38, 9), (38, 12)], [(38, 9), (36, 7)], [(36, 7), (38, 9)], [(36, 7), (34, 5)], [(46, 7), (48, 9), (48, 12)], [(51, 12), (51, 9)], [(51, 9), (51, 12)], [(51, 9), (49, 7)], [(49, 7), (51, 9)], [(49, 7), (47, 5)], [(58, 7), (60, 9), (60, 12)], [(63, 12), (63, 9)], [(63, 9), (63, 12)], [(63, 9), (61, 7)], [(61, 7), (63, 9)], [(61, 7), (59, 5)], [(15.0, 7.0), (18.0, 7.0), (21, 7)], [(21, 7), (24, 7)], [(24.0, 7.0), (27.0, 7.0), (30.0, 7.0), (33, 7)], [(33, 7), (36, 7)], [(36.0, 7.0), (38.5, 7.0), (41.0, 7.0), (43.5, 7.0), (46, 7)], [(46, 7), (49, 7)], [(49.0, 7.0), (52.0, 7.0), (55.0, 7.0), (58, 7)], [(59.0, 5.0), (56.0, 5.0), (53.0, 5.0), (50.0, 5.0), (47, 5)], [(47.0, 5.0), (44.4, 5.0), (41.8, 5.0), (39.2, 5.0), (36.6, 5.0), (34, 5)], [(34.0, 5.0), (31.0, 5.0), (28.0, 5.0), (25.0, 5.0), (22, 5)], [(22.0, 5.0), (19.0, 5.0), (16, 5)], [(23, 12), (23, 9)], [(23, 9), (24, 7)], [(35, 12), (35, 9)], [(35, 9), (36, 7)], [(48, 12), (48, 9)], [(48, 9), (49, 7)], [(60, 12), (60, 9)], [(60, 9), (61, 7)], [(7.0, 27.0), (7.0, 29.0), (7, 31)], [(7.0, 9.0), (7.0, 11.0), (7, 13)], [(7.0, 27.0), (8.5, 26.25), (10.0, 25.5)], [(13, 9), (11, 7)], [(14, 31), (12, 31)], [(12, 31), (14, 31)], [(14, 29), (12, 29)], [(12, 29), (14, 29)], [(14, 27), (12, 27)], [(12, 27), (14, 27)], [(14, 33), (12, 33)], [(12, 33), (14, 33)], [(12.0, 27.0), (12.0, 29.0), (12.0, 31.0), (12, 33)], [(12, 33), (12.0, 31.0), (12.0, 29.0), (12.0, 27.0)], [(12, 31), (9, 31)], [(9, 31), (12, 31)]]
        # else:
        #     paths = [[(58,7), (61,7)], [(7,7), (7,9)], [(7,9), (7,7)], [(7,15), (7,13)], [(7,13), (7,15)], [(9.0, 31.0), (7.0, 31.0), (5, 31)], [(5, 31), (7.0, 31.0), (9.0, 31.0)], [(7, 13), (5, 13)], [(5, 13), (7, 13)], [(9.0, 31.0), (9.5, 28.25), (10.0, 25.5), (10.5, 22.75), (11.0, 20.0), (11.5, 17.25), (12.0, 14.5), (12.5, 11.75), (13, 9)], [(10.0, 11.0), (7, 13)], [(7, 13),  (10.0, 11.0)], [(10.0, 11.0), (13, 9)], [(13, 9), (15, 7)], [(5, 27), (7, 27)], [(7, 27), (5, 27)], [(5, 9), (7, 9)], [(7, 9), (5, 9)], [(10.0, 11.0), (9.5, 13.67), (9.0, 16.33), (8.5, 19.0), (8.0, 21.67), (7.5, 24.33), (7, 27)], [(11.0, 7.0), (9.0, 8.0), (7, 9)], [(11.0, 7.0), (10.5, 9.0), (10, 11)], [(16.0, 5.0), (13.5, 6.0), (11, 7)], [(21, 7), (23, 9), (23, 12)], [(26, 12), (26, 9)], [(26, 9), (26, 12)], [(26, 9), (24, 7)], [(24, 7), (26, 9)], [(24, 7), (22, 5)], [(31, 5), (33, 7)], [(33, 7), (35, 9), (35, 12)], [(38, 12), (38, 9)], [(38, 9), (38, 12)], [(38, 9), (36, 7)], [(36, 7), (38, 9)], [(36, 7), (34, 5)], [(34, 5), (32, 3)], [(44, 5), (46, 7)], [(46, 7), (48, 9), (48, 12)], [(51, 12), (51, 9)], [(51, 9), (51, 12)], [(51, 9), (49, 7)], [(49, 7), (51, 9)], [(49, 7), (47, 5)], [(47, 5), (45, 3)], [(56, 5), (58, 7)], [(58, 7), (60, 9), (60, 12)], [(63, 12), (63, 9)], [(63, 9), (63, 12)], [(63, 9), (61, 7)], [(61, 7), (63, 9)], [(61, 7), (59, 5)], [(59, 5), (57, 3)], [(15.0, 7.0), (18.0, 7.0), (21, 7)], [(21, 7), (24, 7)], [(24.0, 7.0), (27.0, 7.0), (30.0, 7.0), (33, 7)], [(33, 7), (36, 7)], [(36.0, 7.0), (38.5, 7.0), (41.0, 7.0), (43.5, 7.0), (46, 7)], [(46, 7), (49, 7)], [(49.0, 7.0), (52.0, 7.0), (55.0, 7.0), (58, 7)], [(47.0, 5.0), (50.0, 5.0), (53.0, 5.0), (56.0, 5.0), (59, 5)], [(59, 5), (56.0, 5.0), (53.0, 5.0), (50.0, 5.0), (47.0, 5.0)], [(34.0, 5.0), (36.6, 5.0), (39.2, 5.0), (41.8, 5.0), (44.4, 5.0), (47, 5)], [(47, 5), (44.4, 5.0), (41.8, 5.0), (39.2, 5.0), (36.6, 5.0), (34.0, 5.0)], [(22.0, 5.0), (25.0, 5.0), (28.0, 5.0), (31.0, 5.0), (34, 5)], [(34, 5), (31.0, 5.0), (28.0, 5.0), (25.0, 5.0), (22.0, 5.0)], [(16.0, 5.0), (19.0, 5.0), (22, 5)], [(22, 5), (19.0, 5.0), (16.0, 5.0)], [(18, 7), (19, 5)], [(19, 5), (18, 7)], [(21, 3), (19, 5)], [(57.0, 3.0), (54.0, 3.0), (51.0, 3.0), (48.0, 3.0), (45, 3)], [(45.0, 3.0), (42.4, 3.0), (39.8, 3.0), (37.2, 3.0), (34.6, 3.0), (32, 3)], [(32.0, 3.0), (29.25, 3.0), (26.5, 3.0), (23.75, 3.0), (21, 3)], [(23, 12), (23, 9)], [(23, 9), (24, 7)], [(35, 12), (35, 9)], [(35, 9), (36, 7)], [(48, 12), (48, 9)], [(48, 9), (49, 7)], [(60, 12), (60, 9)], [(60, 9), (61, 7)], [(7.0, 27.0), (7.0, 29.0), (7, 31)], [(7.0, 9.0), (7.0, 11.0), (7, 13)], [(7.0, 27.0), (8.5, 26.25), (10.0, 25.5)], [(13, 9), (11, 7)], [(14, 31), (12, 31)], [(12, 31), (14, 31)], [(14, 29), (12, 29)], [(12, 29), (14, 29)], [(14, 27), (12, 27)], [(12, 27), (14, 27)], [(14, 33), (12, 33)], [(12, 33), (14, 33)], [(12.0, 27.0), (12.0, 29.0), (12.0, 31.0), (12, 33)], [(12, 33), (12.0, 31.0), (12.0, 29.0), (12.0, 27.0)], [(12, 31), (9, 31)], [(9, 31), (12, 31)]]          
        with open('paths.pkl', 'rb') as f:
            paths = pickle.load(f)
        
        for path in paths:
            path_x = [path[i][0] for i in range(len(path))]
            path_y = [path[i][1] for i in range(len(path))]
            plt.plot(path_x, path_y, '-', linewidth='2', color='#13ae00', alpha=1.0)
        for p in self.patches:
            self.ax.add_patch(p)
        for a in self.artists:
            self.ax.add_artist(a)
        return self.patches + self.artists

    def animate_func(self, i):
        self.ax.set_title(f"Time: {i/ self.slow_factor:.2f} seconds")
        for agent_name, agent in self.combined_schedule.items():
            pos = self.getState(i / self.slow_factor, agent)
            p = (pos[0], pos[1])
            self.agents[agent_name].center = p
            self.agent_names[agent_name].set_position(p)

        for j in range(len(self.dynamic_obstacles)):
            ob_path = self.dynamic_obstacles['obstacle'+str(j)]
            patch = self.dyn_ob[j]
            path_length = len(ob_path)
            time = i / self.slow_factor
            index = time % path_length
            index_int = int(index)
            index_frac = index - index_int

            # Current and next positions
            current_pos = np.array([ob_path[index_int]['x'], ob_path[index_int]['y']])
            next_pos = np.array([ob_path[(index_int+1)%path_length]['x'], ob_path[(index_int+1)%path_length]['y']])

            # Linear interpolation
            interpolated_pos = current_pos + index_frac * (next_pos - current_pos)
            patch.center = interpolated_pos

        # Reset all colors
        for _, agent in self.agents.items():
            agent.set_facecolor(agent.original_face_color)

        # if 20 < i/self.slow_factor + 1 < 80:
        #     self.dynamic_ob.set_alpha(1)
        # elif 80 <= i/self.slow_factor + 1:
        #     self.dynamic_ob.set_alpha(0)
        # Make tasks visible at the right time
        for t in map["tasks"]:
            if t['start_time'] <= i / self.slow_factor + 1 <= self.schedule['completed_tasks_times'][t['task_name']]:
                for task in self.tasks[t['task_name']]:
                    task.set_alpha(0.5)
            else:
                for task in self.tasks[t['task_name']]:
                    task.set_alpha(0)

        # Check drive-drive collisions
        agents_array = [agent for _, agent in self.agents.items()]
        for i in range(0, len(agents_array)):
            for j in range(i + 1, len(agents_array)):
                d1 = agents_array[i]
                d2 = agents_array[j]
                pos1 = np.array(d1.center)
                pos2 = np.array(d2.center)
                if np.linalg.norm(pos1 - pos2) < 0.7:
                    d1.set_facecolor('red')
                    d2.set_facecolor('red')
                    print("COLLISION! (agent-agent) ({}, {})".format(i, j))

        return self.patches + self.artists

    def getState(self, t, d):
        idx = 0
        while idx < len(d) and d[idx]["t"] < t:
            idx += 1
        if idx == 0:
            return np.array([float(d[0]["x"]), float(d[0]["y"])])
        elif idx < len(d):
            posLast = np.array([float(d[idx - 1]["x"]), float(d[idx - 1]["y"])])
            posNext = np.array([float(d[idx]["x"]), float(d[idx]["y"])])
        else:
            return np.array([float(d[-1]["x"]), float(d[-1]["y"])])
        dt = d[idx]["t"] - d[idx - 1]["t"]
        t = (t - d[idx - 1]["t"]) / dt
        pos = (posNext - posLast) * t + posLast
        return pos


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-map", help="input file containing map")
    parser.add_argument("-layout", help="2-line or 3-line layout", default=3, type=int)
    parser.add_argument("-schedule", help="schedule for agents")
    parser.add_argument('-slow_factor', help='Slow factor of visualization', default=1, type=int)
    parser.add_argument('--video', dest='video', default=None,
                        help="output video file (or leave empty to show on screen)")
    parser.add_argument("--speed", type=int, default=1, help="speedup-factor")
    args = parser.parse_args()

    if args.map is None:
        with open(os.path.join(RoothPath.get_root(), 'config.json'), 'r') as json_file:
            config = json.load(json_file)
        args.map = os.path.join(RoothPath.get_root(), os.path.join(config["Defaults"]['input_path'], config["Defaults"]['input_name'] + config["Defaults"]['visual_postfix'],))
        args.schedule = os.path.join(RoothPath.get_root(), 'output.yaml')

    with open(args.map) as map_file:
        map = yaml.load(map_file, Loader=yaml.FullLoader)

    with open(args.schedule) as states_file:
        schedule = yaml.load(states_file, Loader=yaml.FullLoader)

    animation = Animation(map, schedule, slow_factor=args.slow_factor)
    # animation.show()
    
    if args.video:
        animation.save(args.video, args.speed)
 
    


