
# Importing the dataset
dataset = read.csv('Lesson_1/Salary_Data.csv')

# Splitting the dataset into the Training set and Test set
library(caTools)
set.seed(123)
split <- sample.split(dataset$Salary, SplitRatio = 2/3) 
# This is a vector of boolean values. True means it goes to training set and False goes to test set.
training_set <- subset(dataset, split == TRUE)
test_set <- subset(dataset, split == FALSE)

# The model will learn the correlation between salary and yrs of experience in the training set and we will apply/ 
# use its power of prediction on the test set.

# Fitting Simple Linear Regression to the Training set
# This is where the coefficients are calculated. 
regressor <- lm(Salary ~ YearsExperience, data = training_set)
summary(regressor) # *** means it is highly statistically significant.

# Predicting the Test set results.
y_pred <- predict(regressor, newdata = test_set)

# Visualising the Training set results.
library(ggplot2)
ggplot() +
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour = 'green') +
  ggtitle('Salary vs Experience (Training set)') +
  xlab('Years of experience') +
  ylab('Salary')

# Visualising the Test set results.
ggplot() +
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary),
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour = 'green') +
  ggtitle('Salary vs Experience (Test set)') +
  xlab('Years of experience') +
  ylab('Salary')


# Feature Scaling
# training_set[, 2:3] <- scale(training_set[, 2:3])
# test_set[, 2:3] <- scale(test_set[, 2:3])