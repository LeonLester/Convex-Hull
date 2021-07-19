# Python code to generate
# random numbers,
# append them to a list
# and save them to a file


import numpy as np
# Function to generate
# and append them
# start = starting range,
# end = ending range
# num = number of elements needs to be appended


# function to create coordinates for points in general since it's one coordinate per line
def create2Dpoints(start, end, num):
    f = open("2Dpoints.txt", "w+")
    # f.close()  # close the file after creation

    list = np.random.uniform(start, end, 2 * num)  # get the list from Rand

    with open('2Dpoints.txt', 'w') as output:  # with statement manages file closure later
        for x in range(len(list)):
            output.write(str(list[x]) + "\n")  # write the number


def create3Dpoints(start, end, num):
    f = open("3Dpoints.txt", "w+")
    # f.close()  # close the file after creation

    list = np.random.uniform(start, end, 3 * num)  # get the list from Rand

    with open('3Dpoints.txt', 'w') as output:  # with statement manages file closure later
        for x in range(len(list)):
            output.write(str(list[x]) + "\n")  # write the number



# Driver Code

point_user_input = int(input("Choose number of points: "))
number_of_dimensions = int(input("Choose number of dimensions: "))
upper_limit_user_input = int(input("Choose upper limit: "))
lower_limit_user_input = int(input("Choose lower limit: "))

if number_of_dimensions == 2:
    create2Dpoints(upper_limit_user_input, lower_limit_user_input, point_user_input)
elif number_of_dimensions == 3:
    create3Dpoints(upper_limit_user_input, lower_limit_user_input, point_user_input)
