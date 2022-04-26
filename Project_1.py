# Project based on using Loops for define Medical Insurance Estimates vs. Costs Project.
# I am interested in analyzing medical insurance cost data efficiently without writing repetitive code.
# In this project, I was use my knowledge of Python loops to iterate through and analyze medical insurance cost data.

# The names of fiveteen individuals.
names = ["Judith", "Abel", "Tyson", "Martha", "Beverley", "David", "Anabel", "Liam", "Noah", "Emma", "Olivia", "Oliver", "Ava", " Charlotte", " Elijah"]
# Estimated medical insurance costs for the individuals.
estimated_insurance_costs = [4321.0, 6440.0, 5599.0, 2193.0, 5773.0, 7694.0, 3797.0, 3662.0, 2402.0, 4535.0, 3154.0, 3122.0, 4801.0, 4758.0, 4488.0]
# The actual insurance costs paid by the individuals.
actual_insurance_costs = [4753.1, 7084.0, 6158.9, 2412.3, 6350.3, 8463.4, 4176.7, 4028.2, 2642.2, 4988.5, 3469.4, 3434.2, 5281.1, 5233.8, 4936.8]

# Add your code here
total_cost = 0  
for insurance_costs in actual_insurance_costs:
  total_cost += insurance_costs
average_cost = total_cost/len(actual_insurance_costs)
print("Average Insurance Cost: " + str(average_cost) + " dollars.")

for i in range(len(names)):
  name = names[i]
  insurance_cost = actual_insurance_costs[i]
  print("The insurance cost for " + name + " is " + str(insurance_cost) + " dollars.")
# checks if insurance cost is above average
  if insurance_cost > average_cost:
    print("The insurance cost for " + name + "is above average.")
# checks if insurance cost is below average
  elif insurance_cost < average_cost:
    print("The insurance cost for " + name + "is below average.")
# checks if insurance cost is equal to the average
  else:
    print("The insurance cost for " + name + "is equal to the average.")

# each of the actual insurance costs are 10% higher than the estimated insurance costs, so
updated_estimated_costs = [estimated_cost * 11/10 for estimated_cost in estimated_insurance_costs]
print(updated_estimated_costs)

























