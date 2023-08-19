from dataclasses import dataclass

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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Point):
            return NotImplemented

        x_check = np.isclose(self.x, other.x)
        y_check = np.isclose(self.y, other.y)

        return x_check and y_check

    def length(self) -> float:
        return np.sqrt(self.x**2 + self.y**2)

    def length_squared(self) -> float:
        return self.x**2 + self.y**2


def ccw(A: Point, B: Point, C: Point) -> bool:
    return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x)


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


def plot_triangle(ax: plt.Axes, triangle: Triangle):
    ax.plot(
        [triangle.A.x, triangle.B.x, triangle.C.x, triangle.A.x],
        [triangle.A.y, triangle.B.y, triangle.C.y, triangle.A.y],
        color="black",
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

    _, ax = plt.subplots()

    A = Point(0, 0)
    B = Point(1, 0)
    C = Point(0, 1)

    triangle = Triangle(A, B, C)
    circumcentre = triangle.circumcentre()
    midpoints = triangle.midpoints()

    plot_point(ax, A)
    plot_point(ax, B)
    plot_point(ax, C)

    for midpoint in midpoints:
        plot_point(ax, midpoint, color="green")

    plot_triangle(ax, triangle)

    ax.grid()

    plt.show()
