Practical Machine Learning: Project Writeup
========================================================

## Load data

```r
set.seed(123)

training <- read.csv("pml-training.csv", stringsAsFactors = FALSE)
testing <- read.csv("pml-testing.csv", stringsAsFactors = FALSE)
```


## Preprocessing
- divide data into 10 parts, 9 for training, 1 for cross validation

```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
folds <- createFolds(y = training$classe, k = 10, list = FALSE)
training_trainset <- training[folds != 10, ]
training_testset <- training[folds == 10, ]
```

- due to PC memory issue, use 1% data instead

```r
inTrain <- createDataPartition(y = training_trainset$classe, p = 0.01, list = FALSE)
inTest <- createDataPartition(y = training_testset$classe, p = 0.01, list = FALSE)
```

- take column 8,9,10,11 as predictor, classe as outcome

```r
training_trainset_predictor <- training_trainset[inTrain, c(8:11, 160)]
training_testset_predictor <- training_testset[inTest, c(8:11, 160)]
testing_predictor <- testing[, c(8:11, 160)]
```

- transform classe into factor

```r
training_trainset_predictor <- transform(training_trainset_predictor, classe = as.factor(classe))
training_testset_predictor <- transform(training_testset_predictor, classe = as.factor(classe))
```


## Training
Apply "Tree Classification"

```r
set.seed(123)
modelFit <- train(classe ~ ., method = "rpart", data = training_trainset_predictor)
```

```
## Loading required package: rpart
```

```r
print(modelFit$finalModel)
```

```
## n= 179 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
##  1) root 179 128 A (0.28 0.2 0.17 0.16 0.18)  
##    2) roll_belt< 132 166 115 A (0.31 0.21 0.19 0.17 0.12)  
##      4) yaw_belt>=169 15   0 A (1 0 0 0 0) *
##      5) yaw_belt< 169 151 115 A (0.24 0.23 0.21 0.19 0.13)  
##       10) yaw_belt< -85.75 74  50 A (0.32 0.19 0.23 0.041 0.22) *
##       11) yaw_belt>=-85.75 77  51 D (0.16 0.27 0.18 0.34 0.052) *
##    3) roll_belt>=132 13   0 E (0 0 0 0 1) *
```


## Inside Test 

```r
y1 <- predict(modelFit, newdata = training_trainset_predictor)
table(training_trainset_predictor$classe)
```

```
## 
##  A  B  C  D  E 
## 51 35 31 29 33
```

```r
table(y1)
```

```
## y1
##  A  B  C  D  E 
## 89  0  0 77 13
```

```r
table(y1 == training_trainset_predictor$classe)
```

```
## 
## FALSE  TRUE 
##   101    78
```

Train Set Accuracy

```r
table(y1 == training_trainset_predictor$classe)[2]/sum(table(y1 == training_trainset_predictor$classe))
```

```
##   TRUE 
## 0.4358
```


## Cross Validation

```r
y2 <- predict(modelFit, newdata = training_testset_predictor)
table(training_testset_predictor$classe)
```

```
## 
## A B C D E 
## 6 4 4 4 4
```

```r
table(y2)
```

```
## y2
##  A  B  C  D  E 
## 15  0  0  5  2
```

```r
table(y2 == training_testset_predictor$classe)
```

```
## 
## FALSE  TRUE 
##    11    11
```

Cross Validation Set Accuracy

```r
table(y2 == training_testset_predictor$classe)[2]/sum(table(y2 == training_testset_predictor$classe))
```

```
## TRUE 
##  0.5
```


## Outside Test

```r
y3 <- predict(modelFit, newdata = testing_predictor)
table(y3)
```

```
## y3
##  A  B  C  D  E 
## 14  0  0  6  0
```

Prediction Outcome

```r
y3
```

```
##  [1] D A A D A A A A A D A A A A D A D A D A
## Levels: A B C D E
```


