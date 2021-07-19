import matplotlib.pyplot as plt
from matplotlib import interactive

interactive(True)


# point class with x, y as point
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def Left_most_point(points):
    '''
    Finding the left most point
    '''
    minn = 0
    for i in range(1, len(points)):
        if points[i].x < points[minn].x:
            minn = i
        elif points[i].x == points[minn].x:
            if points[i].y > points[minn].y:
                minn = i
    return minn


def orientation(p, q, r):  # implementation of the cross product equation
    '''
    0 --> p, q and r are colinear
    1 --> Clockwise
    2 --> Counterclockwise
    '''
    val = (q.y - p.y) * (r.x - q.x) - \
          (q.x - p.x) * (r.y - q.y)

    if val == 0:
        return 0
    elif val > 0:
        return 1
    else:
        return 2


def convexHull(points, n):
    # There must be at least 3 points
    if n < 3:
        print("Not enough points")
        return

    # Find the leftmost point
    l = Left_most_point(points)

    # make a list for the hull
    hull = []

    ''' 
    Start from leftmost point, keep moving counterclockwise 
    until reach the start point again. 
    '''
    p = l
    q = 0
    while (True):

        # Add current point to result
        hull.append(p)

        ''' 
        Search for a point 'q' such that orientation(p, x, 
        q) is counterclockwise for all points 'x'. The idea 
        is to keep track of last visited most counterclock- 
        wise point in q. If any point 'i' is more counterclock- 
        wise than q, then update q. 
        '''
        q = (p + 1) % n

        for i in range(n):

            # If i is more counterclockwise
            # than current q, then update q
            if (orientation(points[p],
                            points[i], points[q]) == 2):
                q = i

        ''' 
        Now q is the most counterclockwise with respect to p 
        Set p as q for next iteration, so that q is added to 
        result 'hull' 
        '''
        p = q

        # While we don't come to first point
        if (p == l):
            break

    # Print Result
    print("Points in Convex Hull")
    for each in hull:
        print(points[each].x, points[each].y)
    convex_hull = []
    for i in hull:
        convex_hull.append(Point(points[i].x, points[i].y))
    return convex_hull


# function to get the file of coordinates into a list
def fileToList():
    coordinates = []
    with open('2Dpoints.txt') as fileInput:
        for line in fileInput.readlines():
            coordinates.append(float(line))
    return coordinates


# unused but useful
def get_2d_point_coordinates(coordinates):
    x_coordinates = coordinates[0::2]
    y_coordinates = coordinates[1::2]

    return x_coordinates, y_coordinates


def draw_2d_convex_hull(points, convex_hull):  # draw the points with the hull
    # initialize the lists that will contain the point coordinates
    x = []
    y = []
    hull_x = []
    hull_y = []

    # get all the points
    for point in points:
        x.append(point.x)
        y.append(point.y)

    # get the points of the convex hull
    for point in convex_hull:
        hull_x.append(point.x)
        hull_y.append(point.y)

    # add the first element to have a closed loop
    hull_x.append(convex_hull[0].x)
    hull_y.append(convex_hull[0].y)

    # Set chart title.
    plt.title("All points")

    # Draw point based on above x, y axis values.
    plt.scatter(x, y)
    plt.scatter(hull_x,hull_y, color='red')
    plt.plot(hull_x, hull_y, color='red', linewidth=1)
    plt.savefig('2Dplot.png', dpi=1200)
    plt.show()


# Driver Code


# 1. get the coordinates in a list
coordinates = fileToList()

# 2. get the points into a list of Point objects
points = []
index = 0
x_coordinates, y_coordinates = get_2d_point_coordinates(coordinates)
for i in range(len(x_coordinates)):
    # fill the x,y with the coordinates
    points.append(Point(x_coordinates[index], y_coordinates[index]))
    index += 1

# and draw the convex hull by calling the convexHull function for the points
draw_2d_convex_hull(points, convexHull(points, len(points)))