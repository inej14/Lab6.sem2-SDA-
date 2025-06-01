import math
import random
import tkinter as tk
from typing import List, Tuple

Coord = Tuple[float, float]
Matrix = List[List[int]]

VARIANT = 4104
PANEL_SIZE = 600
OUTER_RADIUS = 0.40 * PANEL_SIZE
NODE_RADIUS = 22
EDGE_WIDTH = 3

n1, n2, n3, n4 = map(int, str(VARIANT).zfill(4))
vertex_count = 10 + n3
k = 1.0 - n3 * 0.01 - n4 * 0.01 - 0.3


def generate_weight_matrix(size: int, seed: int) -> Matrix:
    random.seed(seed)
    W = [[0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i + 1, size):
            if random.random() < k:
                weight = random.randint(1, 200)
                W[i][j] = W[j][i] = weight
    return W


def generate_adjacency_matrix(weight_matrix: Matrix) -> Matrix:
    size = len(weight_matrix)
    A = [[0] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            if weight_matrix[i][j] > 0:
                A[i][j] = 1
    return A


def node_positions_triangle(count: int, offset_x: int) -> List[Coord]:
    cx, cy, r = offset_x + PANEL_SIZE / 2, PANEL_SIZE / 2, OUTER_RADIUS
    angles = [math.pi / 2 + i * 2 * math.pi / 3 for i in range(3)]
    verts = [(cx + r * math.cos(a), cy - r * math.sin(a)) for a in angles]
    base, extra = divmod(count, 3)

    def points_on_edge(p1, p2, n):
        return [(p1[0] + (p2[0] - p1[0]) * i / n, p1[1] + (p2[1] - p1[1]) * i / n) for i in range(n)]

    points = []
    for i in range(3):
        n_nodes = base + (1 if i < extra else 0)
        points += points_on_edge(verts[i], verts[(i + 1) % 3], n_nodes)
    return points[:count]


class DisjointSet:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        xr, yr = self.find(x), self.find(y)
        if xr == yr:
            return False
        self.parent[yr] = xr
        return True


def kruskal_mst(weight_matrix: Matrix) -> List[Tuple[int, int, int]]:
    edges = [(i, j, weight_matrix[i][j])
             for i in range(len(weight_matrix))
             for j in range(i + 1, len(weight_matrix))
             if weight_matrix[i][j] > 0]
    edges.sort(key=lambda x: x[2])

    print("\nAll edges sorted by weight:")
    for u, v, w in edges:
        print(f"Edge {u+1}-{v+1}, weight: {w}")

    dsu = DisjointSet(len(weight_matrix))
    mst = []

    for u, v, w in edges:
        if dsu.union(u, v):
            mst.append((u, v, w))

    print("\nEdges in the minimum spanning tree:")
    for u, v, w in mst:
        print(f"Edge {u+1}-{v+1}, weight: {w}")

    total_weight = sum(w for _, _, w in mst)
    print(f"\nTotal weight of MST: {total_weight}")

    return mst


class GraphRenderer:
    def __init__(self, canvas: tk.Canvas, positions: List[Coord]):
        self.canvas = canvas
        self.positions = positions
        self.drawn_edges = {}

    def draw_node(self, x: float, y: float, label: str):
        self.canvas.create_oval(x - NODE_RADIUS, y - NODE_RADIUS, x + NODE_RADIUS, y + NODE_RADIUS,
                                fill="lightblue", outline="black", width=2)
        self.canvas.create_text(x, y, text=label, font=("Arial", 12, "bold"))

    def draw_edge(self, u: int, v: int, weight: int, color: str = "black", tag: str = ""):
        x1, y1 = self.positions[u]
        x2, y2 = self.positions[v]

        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length == 0:
            return

        offset = NODE_RADIUS
        nx = dx / length
        ny = dy / length

        start_x = x1 + nx * offset
        start_y = y1 + ny * offset
        end_x = x2 - nx * offset
        end_y = y2 - ny * offset

        self.canvas.create_line(start_x, start_y, end_x, end_y, width=EDGE_WIDTH, fill=color, tags=tag)

        mx, my = (start_x + end_x) / 2, (start_y + end_y) / 2
        offset_x, offset_y = 10, -10
        self.canvas.create_text(mx + offset_x, my + offset_y, text=str(weight), font=("Arial", 10, "bold"), fill="darkblue")

        self.drawn_edges[(u, v)] = tag

    def draw_graph(self, matrix: Matrix):
        for i in range(len(matrix)):
            for j in range(i + 1, len(matrix)):
                if matrix[i][j]:
                    self.draw_edge(i, j, matrix[i][j])
        for idx, (x, y) in enumerate(self.positions):
            self.draw_node(x, y, str(idx + 1))

    def draw_next_mst_edge(self, edge: Tuple[int, int, int]):
        u, v, w = edge
        self.draw_edge(u, v, w, color="red")


def print_matrix(matrix: Matrix, title: str):
    print(f"\n{title}:")
    for row in matrix:
        print(" ".join(f"{v:3}" for v in row))


if __name__ == "__main__":
    weight_matrix = generate_weight_matrix(vertex_count, VARIANT)
    adjacency_matrix = generate_adjacency_matrix(weight_matrix)

    print_matrix(adjacency_matrix, "Adjacency matrix A")
    print_matrix(weight_matrix, "Weight matrix W")

    mst_edges = kruskal_mst(weight_matrix)

    root = tk.Tk()
    root.title(f"Kruskal MST · Variant {VARIANT}")
    canvas = tk.Canvas(root, width=PANEL_SIZE, height=PANEL_SIZE + 60, bg="white")
    canvas.pack()

    positions = node_positions_triangle(vertex_count, offset_x=0)
    renderer = GraphRenderer(canvas, positions)
    renderer.draw_graph(weight_matrix)

    step_index = 0

    def next_step():
        global step_index
        if step_index < len(mst_edges):
            renderer.draw_next_mst_edge(mst_edges[step_index])
            step_index += 1

    button = tk.Button(root, text="Next Step", command=next_step, font=("Arial", 12, "bold"))
    button.pack(pady=10)

    root.bind("<space>", lambda event: next_step())
    root.mainloop()
