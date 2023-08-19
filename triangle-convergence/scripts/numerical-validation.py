import matplotlib.pyplot as plt
import numpy as np
from rich import print


# ==============================================================================
# Classes
# ==============================================================================
class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"

    def __repr__(self) -> str:
        return f"Point({self.x}, {self.y})"

    def __eq__(self, other: "Point") -> bool:
        if not isinstance(other, Point):
            return NotImplemented

        x_check = np.isclose(self.x, other.x)
        y_check = np.isclose(self.y, other.y)

        return x_check and y_check

    def length(self) -> float:
        return np.sqrt(self.x**2 + self.y**2)

    def length_squared(self) -> float:
        return self.x**2 + self.y**2

    def distance(self, other: "Point") -> float:
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


class Triangle:
    def __init__(self, A: Point, B: Point, C: Point):
        self.A = A
        self.B = B
        self.C = C

    def circumcentre(self) -> Point:
        D = 2 * (
            self.A.x * (self.B.y - self.C.y)
            + self.B.x * (self.C.y - self.A.y)
            + self.C.x * (self.A.y - self.B.y)
        )

        A2 = self.A.length_squared()
        B2 = self.B.length_squared()
        C2 = self.C.length_squared()

        Ux = (
            1
            / D
            * (
                A2 * (self.B.y - self.C.y)
                + B2 * (self.C.y - self.A.y)
                + C2 * (self.A.y - self.B.y)
            )
        )

        Uy = (
            1
            / D
            * (
                A2 * (self.C.x - self.B.x)
                + B2 * (self.A.x - self.C.x)
                + C2 * (self.B.x - self.A.x)
            )
        )

        return Point(Ux, Uy)

    def center_of_mass(self) -> Point:
        return Point(
            (self.A.x + self.B.x + self.C.x) / 3,
            (self.A.y + self.B.y + self.C.y) / 3,
        )

    def midpoints(self) -> list[Point]:
        return [
            Point((self.A.x + self.B.x) / 2, (self.A.y + self.B.y) / 2),
            Point((self.B.x + self.C.x) / 2, (self.B.y + self.C.y) / 2),
            Point((self.C.x + self.A.x) / 2, (self.C.y + self.A.y) / 2),
        ]


# ==============================================================================
# Auxiliary functions
# ==============================================================================


def plot_point(ax: plt.Axes, point: Point, color: str = "black"):
    ax.scatter(point.x, point.y, color=color)


def plot_triangle(ax: plt.Axes, triangle: Triangle, **kwargs):
    ax.plot(
        [triangle.A.x, triangle.B.x, triangle.C.x, triangle.A.x],
        [triangle.A.y, triangle.B.y, triangle.C.y, triangle.A.y],
        **kwargs,
    )


# ==============================================================================
# Tests
# ==============================================================================
def test_point_equivalence():
    print("Testing point equivalence...")
    A = Point(0, 0)
    B = Point(0, 0)

    assert A == B


def test_circumcentre():
    print("Testing circumcentre...")
    A = Point(0, 0)
    B = Point(1, 0)
    C = Point(0, 1)

    triangle = Triangle(A, B, C)
    circumcentre = triangle.circumcentre()
    expected = Point(1 / 2, 1 / 2)

    assert circumcentre == expected


if __name__ == "__main__":
    test_point_equivalence()
    test_circumcentre()

    figsize = (8, 8)
    _, axes = plt.subplots(nrows=2, figsize=figsize)

    axes = axes.flatten()

    A = Point(0, 0)
    B = Point(4, 3.9)
    C = Point(0, 7)

    triangle = Triangle(A, B, C)

    ax = axes[0]

    plot_point(ax, A)
    plot_point(ax, B)
    plot_point(ax, C)
    plot_triangle(ax, triangle)

    # Plot center of mass
    center_of_mass = triangle.center_of_mass()
    plot_point(ax, center_of_mass, color="red")

    N = 25

    previous_triangle = triangle
    distances = []
    for idx in range(N):
        midpoints = previous_triangle.midpoints()
        triangle_new = Triangle(midpoints[0], midpoints[1], midpoints[2])

        circumcentre = triangle_new.circumcentre()
        distance = circumcentre.distance(center_of_mass)
        distances.append(distance)

        plot_triangle(
            ax,
            triangle_new,
            color="black",
            linewidth=0.5,
        )
        previous_triangle = triangle_new

    ax.set_title("Nagore's Point")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax = axes[1]

    ax.semilogy(range(N), distances, marker="o")
    ax.set_title("Distance circumcentres to center of mass")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Distance")
    ax.grid()

    plt.tight_layout()
    plt.show()
