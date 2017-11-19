
# Importing the dataset
dataset = read.csv('Lesson_1/Salary_Data.csv')

# Splitting the dataset into the Training set and Test set
library(caTools)
set.seed(123)
split <- sample.split(dataset$Purchased, SplitRatio = 2/3)
training_set <- subset(dataset, split == TRUE)
test_set <- subset(dataset, split == FALSE)

# Feature Scaling
# training_set[, 2:3] <- scale(training_set[, 2:3])
# test_set[, 2:3] <- scale(test_set[, 2:3])