from pathlib import Path
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from queue import PriorityQueue
from rl.data import Trajectory


class AStar:
    def __init__(self, mesh, start, goal, delta=0.004, step_size=0.006):
        self.mesh = mesh
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.delta = delta
        self.step_size = step_size
        self.open_list = PriorityQueue()
        self.open_list.put((0, tuple(self.start)))
        self.came_from = {}
        self.g_cost = {tuple(self.start): 0}
        self.f_cost = {tuple(self.start): self.heuristic(self.start)}

        # Visualization setup
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_xlim(mesh.bounds[0][0], mesh.bounds[1][0])
        self.ax.set_ylim(mesh.bounds[0][1], mesh.bounds[1][1])
        self.ax.set_zlim(mesh.bounds[0][2], mesh.bounds[1][2])
        self.ax.scatter(*self.start, c="g", s=100)  # start in green
        self.ax.scatter(*self.goal, c="r", s=100)  # goal in red

    def heuristic(self, point):
        return np.linalg.norm(self.goal - np.array(point))

    def get_path(self):
        path = []
        current = tuple(self.goal)  # Ensure the goal is in tuple form

        while current in self.came_from:
            path.append(current)
            current = self.came_from[current]

        path.append(
            tuple(self.start)
        )  # Append the start position as well and ensure it's a tuple
        path.reverse()  # Reverse the path to start from the start position
        return path

    def get_neighbors(self, point):
        neighbors = []
        for dx in [-self.step_size, 0, self.step_size]:
            for dy in [-self.step_size, 0, self.step_size]:
                for dz in [-self.step_size, 0, self.step_size]:
                    if dx == dy == dz == 0:
                        continue
                    new_point = np.array(point) + np.array([dx, dy, dz])
                    if self.mesh.contains(np.expand_dims(new_point, 0)):
                        neighbors.append(tuple(new_point))
        return neighbors

    def build(self):
        while not self.open_list.empty():
            _, current = self.open_list.get()

            if np.linalg.norm(current - self.goal) < self.delta:
                path = self.reconstruct_path(current)
                print("Goal reached!")
                return path

            for neighbor in self.get_neighbors(current):
                tentative_g_cost = self.g_cost[current] + np.linalg.norm(
                    np.array(neighbor) - np.array(current)
                )

                if (
                    neighbor not in self.g_cost
                    or tentative_g_cost < self.g_cost[neighbor]
                ):
                    self.came_from[neighbor] = current
                    self.g_cost[neighbor] = tentative_g_cost
                    self.f_cost[neighbor] = tentative_g_cost + self.heuristic(neighbor)
                    self.open_list.put((self.f_cost[neighbor], neighbor))

                    # Visualization: plot the neighbor node and connecting edge
                    self.ax.scatter(*neighbor, c="b", s=10)
                    self.ax.plot(
                        [current[0], neighbor[0]],
                        [current[1], neighbor[1]],
                        [current[2], neighbor[2]],
                        c="k",
                    )
                    plt.draw()
                    plt.pause(0.001)

        print("Path not found.")

    def reconstruct_path(self, current):
        path = []
        while current in self.came_from:
            path.append(current)
            prev = self.came_from[current]
            self.ax.plot(
                [current[0], prev[0]],
                [current[1], prev[1]],
                [current[2], prev[2]],
                c="y",
                linewidth=2,
            )
            current = prev
        path.append(tuple(self.start))  # Append the start position as well
        path.reverse()  # Reverse the path to start from the start position
        return path

    def show_plot(self):
        plt.show()


if __name__ == "__main__":
    path = Path.cwd() / Path("scratch/phantom3.stl")
    mesh = trimesh.load_mesh(path)
    start = [0.00028165, 0.01593208, 0.00023918]
    goal = [-0.043272, 0.136586, 0.034102]
    astar = AStar(mesh, start, goal)
    path = astar.build()
    astar.show_plot()
    # path = astar.get_path()
    traj = Trajectory.from_dict({"head_pos": path})
    traj.save("./astar_path")
