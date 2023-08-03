class Polygon:
    def __init__(self, vertices: list) -> None:
        self.vertices = vertices
        self.edges = self._calculate_edges()

    def _calculate_edges(self) -> list:
        """
        Calculate and return the edges of the polygon.
        """
        edges = []
        num_vertices = len(self.vertices)

        for i in range(num_vertices):
            edge = (self.vertices[i], self.vertices[(i + 1) % num_vertices])
            edges.append(edge)

        return edges

    def is_point_inside(self, x, y):
        """
        Check if a point (x, y) is inside the polygon.

        Returns:
            bool: True if the point is inside the polygon, False otherwise.
        """
        num_intersections = 0

        for edge in self.edges:
            (x1, y1), (x2, y2) = edge

            # Check if the point lies on the edge
            if (x1 == x and y1 == y) or (x2 == x and y2 == y):
                return True

            # Check for intersection
            if (y1 > y) != (y2 > y):
                if x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
                    num_intersections += 1

        return num_intersections % 2 == 1


# poly1 = Polygon([(0, 0), (10, 30), (20, 0)])
# print(poly1.edges)
# print(poly1.is_point_inside(30, 15))
