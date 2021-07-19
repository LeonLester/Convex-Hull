import math
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def set_correct_normal(possible_internal_points, plane):  # Make the orientation of Normal correct
    for point in possible_internal_points:
        dist = dotProduct(plane.normal, point - plane.pointA)
        if (dist != 0):
            if (dist > 10 ** -10):
                plane.normal.x = -1 * plane.normal.x
                plane.normal.y = -1 * plane.normal.y
                plane.normal.z = -1 * plane.normal.z
                return


def printV(vec):  # Print points
    print(vec.x, vec.y, vec.z)


def crossProduct(pointA, pointB):  # Cross product
    x = (pointA.y * pointB.z) - (pointA.z * pointB.y)
    y = (pointA.z * pointB.x) - (pointA.x * pointB.z)
    z = (pointA.x * pointB.y) - (pointA.y * pointB.x)
    return Point(x, y, z)


def dotProduct(pointA, pointB):  # Dot product
    return pointA.x * pointB.x + pointA.y * pointB.y + pointA.z * pointB.z


def checker_plane(a, b):  # Check if two planes are equal or not

    if ((a.pointA.x == b.pointA.x) and (a.pointA.y == b.pointA.y) and (a.pointA.z == b.pointA.z)):
        if ((a.pointB.x == b.pointB.x) and (a.pointB.y == b.pointB.y) and (a.pointB.z == b.pointB.z)):
            if ((a.pointC.x == b.pointC.x) and (a.pointC.y == b.pointC.y) and (a.pointC.z == b.pointC.z)):
                return True

        elif ((a.pointB.x == b.pointC.x) and (a.pointB.y == b.pointC.y) and (a.pointB.z == b.pointC.z)):
            if ((a.pointC.x == b.pointB.x) and (a.pointC.y == b.pointB.y) and (a.pointC.z == b.pointB.z)):
                return True

    if ((a.pointA.x == b.pointB.x) and (a.pointA.y == b.pointB.y) and (a.pointA.z == b.pointB.z)):
        if ((a.pointB.x == b.pointA.x) and (a.pointB.y == b.pointA.y) and (a.pointB.z == b.pointA.z)):
            if ((a.pointC.x == b.pointC.x) and (a.pointC.y == b.pointC.y) and (a.pointC.z == b.pointC.z)):
                return True

        elif ((a.pointB.x == b.pointC.x) and (a.pointB.y == b.pointC.y) and (a.pointB.z == b.pointC.z)):
            if ((a.pointC.x == b.pointA.x) and (a.pointC.y == b.pointA.y) and (a.pointC.z == b.pointA.z)):
                return True

    if ((a.pointA.x == b.pointC.x) and (a.pointA.y == b.pointC.y) and (a.pointA.z == b.pointC.z)):
        if ((a.pointB.x == b.pointA.x) and (a.pointB.y == b.pointA.y) and (a.pointB.z == b.pointA.z)):
            if ((a.pointC.x == b.pointB.x) and (a.pointC.y == b.pointB.y) and (a.pointC.z == b.pointB.z)):
                return True

        elif ((a.pointB.x == b.pointC.x) and (a.pointB.y == b.pointC.y) and (a.pointB.z == b.pointC.z)):
            if ((a.pointC.x == b.pointB.x) and (a.pointC.y == b.pointB.y) and (a.pointC.z == b.pointB.z)):
                return True

    return False


def checker_edge(a, b):  # Check if 2 edges have same 2 vertices

    if ((a.pointA == b.pointA) and (a.pointB == b.pointB)) or ((a.pointB == b.pointA) and (a.pointA == b.pointB)):
        return True

    return False


class Edge:  # Make a object of type Edge which have two points denoting the vertices of the edges
    def __init__(self, pointA, pointB):
        self.pointA = pointA
        self.pointB = pointB

    def __str__(self):
        string = "Edge"
        string += "\n\tA: " + str(self.pointA.x) + "," + str(self.pointA.y) + "," + str(self.pointA.z)
        string += "\n\tB: " + str(self.pointB.x) + "," + str(self.pointB.y) + "," + str(self.pointB.z)
        return string

    def __hash__(self):
        return hash((self.pointA, self.pointB))

    def __eq__(self, other):
        # print "comparing Edges"
        return checker_edge(self, other)


class Point:  # Point class denoting the points in the space
    def __init__(self, x=None, y=None, z=None):
        self.x = x
        self.y = y
        self.z = z

    def __sub__(self, pointX):
        return Point(self.x - pointX.x, self.y - pointX.y, self.z - pointX.z)

    def __add__(self, pointX):
        return Point(self.x + pointX.x, self.y + pointX.y, self.z + pointX.z)

    def length(self):
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def __str__(self):
        return str(self.x) + "," + str(self.y) + "," + str(self.z)

    def __hash__(self):
        return hash((self.x, self.y, self.z))

    def __eq__(self, other):
        # print "Checking equality of Point"
        return (self.x == other.x) and (self.y == other.y) and (self.z == other.z)


class Plane:  # Plane class having 3 points for a triangle
    def __init__(self, pointA, pointB, pointC):
        self.pointA = pointA
        self.pointB = pointB
        self.pointC = pointC
        self.normal = None
        self.distance = None
        self.calcNorm()
        self.to_do = set()
        self.edge1 = Edge(pointA, pointB)
        self.edge2 = Edge(pointB, pointC)
        self.edge3 = Edge(pointC, pointA)

    def calcNorm(self):  # calculates a normal vector for the plane
        vec1 = self.pointA - self.pointB
        vec2 = self.pointB - self.pointC
        normVector = crossProduct(vec1, vec2)
        length = normVector.length()
        normVector.x = normVector.x / length
        normVector.y = normVector.y / length
        normVector.z = normVector.z / length
        self.normal = normVector
        self.distance = dotProduct(self.normal, self.pointA)

    def dist(self, pointX):  # Returns the distance of the plane from a point X
        return dotProduct(self.normal, pointX - self.pointA)

    def get_edges(self):  # Returns the edges that make up the plane
        return [self.edge1, self.edge2, self.edge3]

    def calculate_to_do(self, temp=None):  # Adds points to the to_do list
        if temp is not None:
            for p in temp:
                dist = self.dist(p)
                if dist > 10 ** (-10):
                    self.to_do.add(p)

        else:
            for p in points:
                dist = self.dist(p)
                if dist > 10 ** (-10):
                    self.to_do.add(p)

    def __eq__(self, other):  # Checks plane equality. Returns true if equal
        return checker_plane(self, other)

    def __str__(self):  # Returns the coordinates of the three points making up the plane and that of the normal vec
        string = "Plane : "
        string += "\n\tX: " + str(self.pointA.x) + "," + str(self.pointA.y) + "," + str(self.pointA.z)
        string += "\n\tY: " + str(self.pointB.x) + "," + str(self.pointB.y) + "," + str(self.pointB.z)
        string += "\n\tZ: " + str(self.pointC.x) + "," + str(self.pointC.y) + "," + str(self.pointC.z)
        string += "\n\tNormal: " + str(self.normal.x) + "," + str(self.normal.y) + "," + str(self.normal.z)
        return string

    def __hash__(self):  # Returns the hash of the plane
        return hash((self.pointA, self.pointB, self.pointC))


# functions


def adjacent_plane(main_plane, edge):  # Finding adjacent planes to an edge
    for plane in list_of_planes:  # for every plane
        edges = plane.get_edges()  # get its edges in the edges list
        if (plane != main_plane) and (edge in edges):  # if that plane isn't the one whose the adjacent we want to find
            return plane  # and the input edge belongs to its edges, it is adjacent to the main_plane


def calc_horizon(visited_planes, plane, eye_point, edge_list):  # Calculating the horizon for an eye to make new faces
    if plane.dist(eye_point) > 10 ** -10:  # If the distance of the eye_point to the plane is big enough
        visited_planes.append(plane)  # 1. Append the plane to the visited planes
        edges = plane.get_edges()  # 2. Get the edges of that plane
        for edge in edges:  # for every edge of the plane
            neighbour = adjacent_plane(plane, edge)  # get the adjacent planes
            if neighbour not in visited_planes:  # if the adjacent planes haven't been visited
                result = calc_horizon(visited_planes, neighbour, eye_point, edge_list)  # call the function
                if result == 0:
                    edge_list.add(edge)

        return 1

    else:
        return 0


def distLine(pointA, pointB, pointX):  # Calculate the distance of a point from a line
    vec1 = pointX - pointA
    vec2 = pointX - pointB
    vec3 = pointB - pointA
    vec4 = crossProduct(vec1, vec2)
    if vec2.length() == 0:
        return None

    else:
        return vec4.length() / vec2.length()


def max_dist_line_point(pointA, pointB):  # Calculate the maximum distant point from a line for initial simplex
    maxDist = 0
    for point in points:  # for every point
        if (pointA != point) and (pointB != point):  # if that point isn't part of the line
            dist = abs(distLine(pointA, pointB, point))  # find it's distance from the line AB
            if dist > maxDist:  # keep track of the max
                maxDistPoint = point
                maxDist = dist

    return maxDistPoint


def max_dist_plane_point(plane):  # Calculate the maximum distance from the plane
    maxDist = 0
    dist = 0
    for point in points:  # for every point
        if (plane.pointA != point) and (plane.pointB != point) and (
                plane.pointC != point):  # if that point isn't part of the plane
            dist = abs(plane.dist(point))  # get it's distance from the plane
        if dist > maxDist:  # keep track of the max
            maxDist = dist
            maxDistPoint = point

    return maxDistPoint


def find_eye_point(plane, to_do_list):  # Calculate the maximum distance from the plane
    maxDist = 0
    for point in to_do_list:
        dist = plane.dist(point)
        if dist > maxDist:
            maxDist = dist
            maxDistPoint = point

    return maxDistPoint


def initial_dis(p, q):  # Gives the Euclidean distance of two points
    return math.sqrt((p.x - q.x) ** 2 + (p.y - q.y) ** 2 + (p.z - q.z) ** 2)


def initial_max(now):  # From the extreme points calculate the 2 most distant points
    maxi = -1
    found = [[], []]
    for i in range(6):
        for j in range(i + 1, 6):
            dist = initial_dis(now[i], now[j])
            if dist > maxi:
                found = [now[i], now[j]]

    return found


def initial():  # To calculate the extreme points to make the initial simplex

    x_min_temp = 10 ** 9
    x_max_temp = -10 ** 9

    y_min_temp = 10 ** 9
    y_max_temp = -10 ** 9

    z_min_temp = 10 ** 9
    z_max_temp = -10 ** 9

    for i in range(len(points)):
        # find max x
        if points[i].x > x_max_temp:
            x_max_temp = points[i].x
            x_max = points[i]
        # find min x
        if points[i].x < x_min_temp:
            x_min_temp = points[i].x
            x_min = points[i]
        # find max y
        if points[i].y > y_max_temp:
            y_max_temp = points[i].y
            y_max = points[i]
        # find min y
        if points[i].y < y_min_temp:
            y_min_temp = points[i].y
            y_min = points[i]
        # find max z
        if points[i].z > z_max_temp:
            z_max_temp = points[i].z
            z_max = points[i]
        # find min z
        if points[i].z < z_min_temp:
            z_min_temp = points[i].z
            z_min = points[i]

    return x_max, x_min, y_max, y_min, z_max, z_min


def fileToList():  # To get the file of coordinates into a list
    coordinates = []
    with open('3Dpoints.txt') as fileInput:
        for line in fileInput.readlines():
            coordinates.append(float(line))
    return coordinates


def draw_3d_points(points, convex_hull_3d):
    from mpl_toolkits.mplot3d import Axes3D
    x = []
    y = []
    z = []
    for point in points:
        x.append(point.x)
        y.append(point.y)
        z.append(point.z)

    # x,y and z axes value list

    hull_x = []
    hull_y = []
    hull_z = []

    for point in convex_hull_3d:
        hull_x.append(point.x)
        hull_y.append(point.y)
        hull_z.append(point.z)

    hull_x.append(convex_hull_3d[0].x)
    hull_y.append(convex_hull_3d[0].y)
    hull_z.append(convex_hull_3d[0].z)

    fig = plt.figure()
    ax = Axes3D(fig)

    ax.scatter(x, y, z, 'o')
    ax.scatter(hull_x, hull_y, hull_z, color='green')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.savefig('3Dplot.png', dpi=1200)
    plt.show()
    ax = fig.add_subplot(111, projection='3d')


def draw_3d_convex_hull(points, list_of_planes):
    plt.figure()
    custom = plt.subplot(111, projection='3d')
    # 1. add all the points
    x = []
    y = []
    z = []
    for point in points:
        x.append(point.x)
        y.append(point.y)
        z.append(point.z)
    custom.scatter(x, y, z)

    # 2.
    hull_x = []
    hull_y = []
    hull_z = []

    for plane in list_of_planes:
        hull_x.append(plane.pointA.x)
        hull_x.append(plane.pointB.x)
        hull_x.append(plane.pointC.x)

        hull_y.append(plane.pointA.y)
        hull_y.append(plane.pointB.y)
        hull_y.append(plane.pointC.y)

        hull_z.append(plane.pointA.z)
        hull_z.append(plane.pointB.z)
        hull_z.append(plane.pointC.z)

    hull_vertices = [list(zip(hull_x, hull_y, hull_z))]
    srf = Poly3DCollection(hull_vertices, alpha=.25, facecolor='#800000')
    plt.gca().add_collection3d(srf)
    custom.set_xlabel('X')
    custom.set_ylabel('Y')
    custom.set_zlabel('Z')
    plt.savefig('3Dplotttt.png', dpi=1200)
    plt.show()


coordinates = fileToList()
points = []  # List to store the points


# fill the x,y,z coordinates
def get_3d_point_coordinates(coordinates):
    x_coordinates = coordinates[0::3]
    y_coordinates = coordinates[1::3]
    z_coordinates = coordinates[2::3]
    return x_coordinates, y_coordinates, z_coordinates


x_coordinates, y_coordinates, z_coordinates = get_3d_point_coordinates(coordinates)

index = 0
for i in range(len(x_coordinates)):
    points.append(Point(x_coordinates[index], y_coordinates[index], z_coordinates[index]))
    index += 1

# Check if we have enough points
if len(points) < 2:
    print("No points found")
    sys.exit()
elif len(points) < 4:
    print("Less than 4 points")

extremes = initial()  # calculate the extreme points for every axis.
initial_line = initial_max(extremes)  # Make the initial line by joining farthest 2 points
third_point = max_dist_line_point(initial_line[0], initial_line[1])  # Calculate the 3rd point to make a plane
first_plane = Plane(initial_line[0], initial_line[1],
                    third_point)  # Make the initial plane by joining 3rd point to the line

fourth_point = max_dist_plane_point(first_plane)  # Make the fourth plane to make a tetrahedron

possible_internal_points = [initial_line[0], initial_line[1], third_point,
                            fourth_point]  # List that helps in calculating orientation of point

second_plane = Plane(initial_line[0], initial_line[1], fourth_point)  # The other planes of the tetrahedron
third_plane = Plane(initial_line[0], fourth_point, third_point)
fourth_plane = Plane(initial_line[1], third_point, fourth_point)

set_correct_normal(possible_internal_points, first_plane)  # Setting the orientation of normal correct
set_correct_normal(possible_internal_points, second_plane)
set_correct_normal(possible_internal_points, third_plane)
set_correct_normal(possible_internal_points, fourth_plane)

first_plane.calculate_to_do()  # Calculating the to_do list which stores the point for which  eye_point have to be found
second_plane.calculate_to_do()
third_plane.calculate_to_do()
fourth_plane.calculate_to_do()

list_of_planes = [first_plane, second_plane, third_plane,
                  fourth_plane]  # List containing all the planes of the tetrahedron

any_left = True  # Checking if planes with to do list is over

# we have made the initial tetrahedron, now we check for remaining points
while any_left:
    any_left = False
    for working_plane in list_of_planes:  # for every plane in the list of planes
        if len(working_plane.to_do) > 0:  # if there are any points at its horizon
            any_left = True
            eye_point = find_eye_point(working_plane, working_plane.to_do)  # Calculate the eye point of the face

            edge_list = set()
            visited_planes = []

            calc_horizon(visited_planes, working_plane, eye_point, edge_list)  # Calculate the horizon

            for internal_plane in visited_planes:  # Remove the internal planes
                list_of_planes.remove(internal_plane)

            for edge in edge_list:  # Make new planes
                new_plane = Plane(edge.pointA, edge.pointB, eye_point)
                set_correct_normal(possible_internal_points, new_plane)

                temp_to_do = set()
                for internal_plane in visited_planes:
                    temp_to_do = temp_to_do.union(internal_plane.to_do)

                new_plane.calculate_to_do(temp_to_do)

                list_of_planes.append(new_plane)

# after this while, the list_of_planes contains the final planes of the hull


final_vertices = set()  # a set of Point objects which will contain the final vertices of the hull
for plane in list_of_planes:  # for every plane in the list of final planes, add each of its points
    final_vertices.add(plane.pointA)
    final_vertices.add(plane.pointB)
    final_vertices.add(plane.pointC)

# list_of_planes has the final planes
# final_vertices has the points that make up the hull


'''
for plane in list_of_planes:
    ax.plot_surface(plane.pointA, plane.pointB, plane.pointC)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.savefig('image.png', bbox_inches='tight')
plt.show()
'''
print("Pοints in convex hull----------------------")
convex_hull_3d = list(final_vertices)
index = 0
for point in convex_hull_3d:
    print(index, point.x, point.y, point.z)
    index += 1
print("All points -----------------------")
index = 0
for point in points:
    print(index, point)
    index += 1
draw_3d_points(points, convex_hull_3d)
# draw_3d_convex_hull(points, list_of_planes)
