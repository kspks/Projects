
# # Project: Linear Regression
# Reggie is a mad scientist who has been hired by the local fast food joint to build their newest ball pit in the play area. As such, he is working on researching the bounciness of different balls so as to optimize the pit. He is running an experiment to bounce different sizes of bouncy balls, and then fitting lines to the data points he records. He has heard of linear regression, but needs your help to implement a version of linear regression in Python.
# Linear Regression is when you have a group of points on a graph, and you find a line that approximately resembles that group of points. A good Linear Regression algorithm minimizes the error, or the distance from each point to the line. A line with the least error is the line that fits the data the best( a line of best fit).
# Let's use loops, lists, and arithmetic to create a function that will find a line of best fit when given a set of data.

# First Step: calculate the Error
# m is the slope of the line and b is the intercept, where the line crosses the y-axis.

def get_y(m, b, x):
  y = m*x + b
  return y

print(get_y(1, 0, 7) == 7)
print(get_y(5, 10, 3) == 25)

# To calculate error between a point and a line, let's use a function `calculate_error()`
def calculate_error(m, b, point):
    x_point, y_point = point
    get_y = m*x_point + b
    distance = abs(get_y - y_point)
    return distance


# Let's test this function:
# For example: this is a line that looks like y = x, so (3, 3) should lie on it. thus, error should be 0:
print(calculate_error(1, 0, (3, 3)))
# Or the point (3, 4) should be 1 unit away from the line y = x:
print(calculate_error(1, 0, (3, 4)))
# Or the point (3, 3) should be 1 unit away from the line y = x - 1:
print(calculate_error(1, -1, (3, 3)))
# Finally the point (3, 3) should be 5 units away from the line y = -x + 1:
print(calculate_error(-1, 1, (3, 3)))

# For instance, Reggie ran an experiment comparing the width of bouncy balls to how high they bounce:
datapoints = [(1, 2), (2, 0), (3, 4), (4, 4), (5, 3)]
# Here the first datapoint, (1, 2), means that his 1cm bouncy ball bounced 2 meters. The 4cm bouncy ball bounced 4 meters.

# For fitting a line to this data, let's use a function called calculate_all_error
def calculate_all_error(m, b, points):
    total_error = 0
    for point in datapoints:
        point_error = calculate_error(m, b, point)
        total_error += point_error
    return total_error


#Let's test this function
#every point in this dataset lies upon y=x, so the total error should be zero:
datapoints = [(1, 1), (3, 3), (5, 5), (-1, -1)]
print(calculate_all_error(1, 0, datapoints))
#every point in this dataset is 1 unit away from y = x + 1, so the total error should be 4:
datapoints = [(1, 1), (3, 3), (5, 5), (-1, -1)]
print(calculate_all_error(1, 1, datapoints))
#every point in this dataset is 1 unit away from y = x - 1, so the total error should be 4:
datapoints = [(1, 1), (3, 3), (5, 5), (-1, -1)]
print(calculate_all_error(1, -1, datapoints))
#the points in this dataset are 1, 5, 9, and 3 units away from y = -x + 1, respectively, so total error should be
# 1 + 5 + 9 + 3 = 18
datapoints = [(1, 1), (3, 3), (5, 5), (-1, -1)]
print(calculate_all_error(-1, 1, datapoints))

# Now we have a function that can take in a line and Reggie's data and return how much error that line produces when we try to fit it to the data.
# The next step is to find the `m` and `b` that minimizes this error, and thus fits the data best!


# Second Step: try a bunch of slopes and intercepts
# The way to find a line of best fit is by trial and error. Let's try a bunch of different slopes (m values) and a bunch of different intercepts (b values) and see which one produces the smallest error value for his dataset.
# Using a list comprehension, let's create a list of possible m values to try. Make the list possible_ms that goes from -10 to 10 inclusive, in increments of 0.1.

possible_ms = [m * 0.1 for m in range(-100, 101)]
# Now, let's make a list of `possible_bs` to check that would be the values from -20 to 20 inclusive, in steps of 0.1:
possible_bs = [b * 0.1 for b in range(-200, 201)]

#We are going to find the smallest error. First, we will make every possible y = m*x + b line by pairing all of the possible ms with all of the possible bs. Then, we will see which y = m*x + b line produces the smallest total error with the set of data stored in datapoint.
datapoints = [(1, 2), (2, 0), (3, 4), (4, 4), (5, 3)]
smallest_error = float("inf")
best_m = 0
best_b = 0

for m in possible_ms:
    for b in possible_bs:
        error = calculate_all_error(m, b, datapoints)
        if error < smallest_error:
            best_m = m
            best_b = b
            smallest_error = error
    
print(best_m, best_b, smallest_error)


#Third Step: Model predictions 
# Now we have seen that for this set of observations on the bouncy balls, the line that fits the data best has an m of 0.3 and a b of 1.7:
# Using this m and this b, what does your line predict the bounce height of a ball with a width of 6 to be? 
get_y(0.3, 1.7, 6)

# Our model predicts that the 6cm ball will bounce 3.5m. So, we can use this model to predict the bounce of all kinds of sizes of balls he may choose to include in the ball pit!

