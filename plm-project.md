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

- experiment with 1%, 5%, 10%, 20%, 50%, 100% of data for training

```r
# ratio = 0.01 ratio = 0.05 ratio = 0.1 ratio = 0.2
ratio = 0.5
# ratio = 1.0
inTrain <- createDataPartition(y = training_trainset$classe, p = ratio, list = FALSE)
inTest <- createDataPartition(y = training_testset$classe, p = ratio, list = FALSE)
```

- take non-NA numeric/integer column 8,9,10,11,... as predictor, classe as outcome

```r
nIndex = 1
j = 1
predictor = vector("numeric", length = 0)
for (i in training) {
    if (nIndex >= 7 && (class(i) == "numeric" || class(i) == "integer") && sum(is.na(i)) == 
        0) {
        predictor[j] <- nIndex
        j = j + 1
    }
    nIndex = nIndex + 1
}

# training_trainset_predictor <- training_trainset[inTrain,c(8:11,160)]
# training_testset_predictor <- training_testset[inTest,c(8:11,160)]
training_trainset_predictor <- training_trainset[inTrain, c(predictor, 160)]
training_testset_predictor <- training_testset[inTest, c(predictor, 160)]
testing_predictor <- testing[, c(predictor, 160)]
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
## n= 8831 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
##  1) root 8831 6320 A (0.28 0.19 0.17 0.16 0.18)  
##    2) roll_belt< 130.5 8107 5606 A (0.31 0.21 0.19 0.18 0.11)  
##      4) pitch_forearm< -26.75 808   33 A (0.96 0.041 0 0 0) *
##      5) pitch_forearm>=-26.75 7299 5573 A (0.24 0.23 0.21 0.2 0.12)  
##       10) num_window>=45.5 6956 5230 A (0.25 0.24 0.22 0.2 0.092)  
##         20) num_window< 241.5 1604  762 A (0.52 0.12 0.11 0.19 0.06) *
##         21) num_window>=241.5 5352 3869 B (0.17 0.28 0.25 0.2 0.1)  
##           42) magnet_dumbbell_z< -24.5 1456  781 A (0.46 0.36 0.046 0.12 0.0096)  
##             84) num_window< 686.5 677  123 A (0.82 0.14 0.0044 0.03 0.0074) *
##             85) num_window>=686.5 779  352 B (0.16 0.55 0.082 0.2 0.012) *
##           43) magnet_dumbbell_z>=-24.5 3896 2599 C (0.054 0.25 0.33 0.23 0.14)  
##             86) magnet_dumbbell_x< -446.5 2769 1540 C (0.061 0.16 0.44 0.25 0.084) *
##             87) magnet_dumbbell_x>=-446.5 1127  602 B (0.036 0.47 0.06 0.18 0.26) *
##       11) num_window< 45.5 343   72 E (0 0 0 0.21 0.79) *
##    3) roll_belt>=130.5 724   10 E (0.014 0 0 0 0.99) *
```


## Inside Test 

```r
y1 <- predict(modelFit, newdata = training_trainset_predictor)
table(training_trainset_predictor$classe)
```

```
## 
##    A    B    C    D    E 
## 2511 1709 1540 1448 1623
```

```r
table(y1)
```

```
## y1
##    A    B    C    D    E 
## 3089 1906 2769    0 1067
```

```r
table(y1 == training_trainset_predictor$classe)
```

```
## 
## FALSE  TRUE 
##  3494  5337
```

Train Set Accuracy

```r
table(y1 == training_trainset_predictor$classe)[2]/sum(table(y1 == training_trainset_predictor$classe))
```

```
##   TRUE 
## 0.6043
```


## Cross Validation
The selection of predictor might not be suitable to outside test set, so apply cross validation to see the out of sample error.

```r
y2 <- predict(modelFit, newdata = training_testset_predictor)
table(training_testset_predictor$classe)
```

```
## 
##   A   B   C   D   E 
## 279 190 171 161 181
```

```r
table(y2)
```

```
## y2
##   A   B   C   D   E 
## 343 220 297   0 122
```

```r
table(y2 == training_testset_predictor$classe)
```

```
## 
## FALSE  TRUE 
##   393   589
```

Cross Validation Set Accuracy

```r
table(y2 == training_testset_predictor$classe)[2]/sum(table(y2 == training_testset_predictor$classe))
```

```
##   TRUE 
## 0.5998
```


## Outside Test

```r
y3 <- predict(modelFit, newdata = testing_predictor)
table(y3)
```

```
## y3
##  A  B  C  D  E 
## 11  3  6  0  0
```

Prediction Outcome

```r
y3
```

```
##  [1] A A A A A C C C A A C C B A C B A A A B
## Levels: A B C D E
```


