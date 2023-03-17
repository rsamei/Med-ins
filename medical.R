data <- read.csv("D:/R/project/Medical insurance/dataTrain.csv")
library(tidyverse)
# the average of charges for people who are smoker is more.
data %>% 
        ggplot(aes(charges,group = smoker)) + 
        geom_density(aes(fill = smoker),alpha = 0.5)
table(data$region) # 4 regions
# it seems that those in soytheast region are charged slightly more. how ever this difference is not huge and there are lots of outliers
data %>% 
        ggplot(aes(reorder(region,charges),charges)) + 
        geom_boxplot(fill = 'orange')
# also Men are charged alightly more than women
data %>% 
        group_by(sex) %>% 
        summarise(mean = mean(charges))
data %>% 
        ggplot(aes(sex,charges)) + 
        geom_boxplot(aes(fill =sex))
# seems no special releatin between  number of children and insurance charges
data %>% 
        ggplot(aes(reorder(factor(children),charges),charges)) + 
        geom_boxplot()
cor(data$charges,data$children) # around 7% correlation
# split data into test and train sets for training  and testing the model

set.seed(55,sample.kind = 'Rejection')
index = sample(1:nrow(data),0.7*nrow(data),replace = F)

train = data[index,]
test = data[-index,]

# check for being balanced
train %>% 
        group_by(sex) %>% 
        summarise(mean = mean(charges))
test %>% 
        group_by(sex) %>% 
        summarise(mean = mean(charges))
# make some variables ready for training
data$sex = factor(data$sex)
data$smoker = factor(data$smoker)
data$region = factor(data$region)
table(data$children)

# simple linear regression
mod_lm = lm(charges ~ .,
            data = train)
summary(mod_lm)
lm_pred = predict(mod_lm,test)
(lm_perf = data.frame(
        MSE = mean((test$charges - lm_pred)^2),
        CORR = cor(test$charges,lm_pred)
)) #performance of lm model



# regression tree
library(rpart)
library(rpart.plot)
mod_tree = rpart(charges ~ .,
                 data= train,
                 method = 'anova')
rpart.plot(mod_tree)
tree_pred = predict(mod_tree,test)
(tree_perf = data.frame(
        MSE = mean((test$charges - tree_pred)^2),
        CORR = cor(test$charges,tree_pred)
)) #performance of lm model
# let's prune the tree
cptable = data.frame(mod_tree$cptable)
cptable
mincpindex = which.min(cptable[,"xerror"])
LL = cptable[mincpindex,"xerror"] - cptable[mincpindex,"xstd"]
UL = cptable[mincpindex,"xerror"] + cptable[mincpindex,"xstd"]
which(cptable[,"xerror"] > LL & cptable[,"xerror"] < UL) # which n. of nodes to take into consideration?
which(cptable[,"xerror"] > LL & cptable[,"xerror"] < UL) %>% min() # which is the smallest three among them?
bestcp = cptable$CP[4]
mod_tree_pruned = prune(mod_tree,cp=bestcp) #pruned tree
rpart.plot(mod_tree_pruned)
pruned_tree_pred = predict(mod_tree_pruned,test)
(pruned_tree_perf = data.frame(
        MSE = mean((test$charges - pruned_tree_pred)^2),
        CORR = cor(test$charges,pruned_tree_pred)
)) # the pruned tree performs slightly worse, but it is simpler


## bagging
library(randomForest)
set.seed(55,sample.kind = 'Rejection')
mod_bag = randomForest(charges ~ .,
                       data = train,
                       mtry = ncol(train) - 1,
                       importance = T,
                       ntree = 500)
bag_pred = predict(mod_bag,test)
(bag_perf = data.frame(
        MSE = mean((test$charges - bag_pred)^2),
        CORR = cor(test$charges,bag_pred)
)) # performance of bagging model
varImpPlot(mod_bag, type=2) # this shows the most important variables effecting the price of insurance are #smoker, #bmi and #age
importance(mod_bag, type=2) # same as last function

# let's change the number of trees in each bag to see if we can improve the performance.
bagtree = seq(10,1000,by=20)
error = c()
for( i in 1:length(bagtree)){
        set.seed(55,sample.kind = 'Rejection')
        mod = randomForest(charges ~ .,
                           data = train,
                           mtry = ncol(train)- 1,
                           importance = T,
                           ntree = bagtree[i])
        
        error[i] = mod$mse[bagtree[i]]
}
plot1 <- data.frame(bagtree,error) %>%
        ggplot() +
        geom_line(aes(bagtree,error))+
        xlab("Number of trees")+
        ylab("OOB MSE")

plotly::ggplotly(plot1) # seems that starting from B=210 the mse is stable so it's ok to run the model with B=210


set.seed(55,sample.kind = 'Rejection')
mod_bag_tuned = randomForest(charges ~ .,
                       data = train,
                       mtry = ncol(train) - 1,
                       importance = T,
                       ntree = 210)
bag_tuned_pred = predict(mod_bag_tuned,test)
(bag_perf = data.frame(
        MSE = mean((test$charges - bag_tuned_pred)^2),
        CORR = cor(test$charges,bag_tuned_pred)
)) # performance of bagging model


#random Forest
rantree = seq(10,1000,by=20)
error_ran = c()
for( i in 1:length(rantree)){
        set.seed(55,sample.kind = 'Rejection')
        mod = randomForest(charges ~ .,
                           data = train,
                           mtry = sqrt(ncol(train)- 1),
                           importance = T,
                           ntree = rantree[i])
        
        error_ran[i] = mod$mse[rantree[i]]
}
plot2 <- data.frame(rantree,error_ran) %>%
        ggplot() +
        geom_line(aes(rantree,error_ran))+
        xlab("Number of trees")+
        ylab("OOB MSE")

plotly::ggplotly(plot2) #510 tree seems fine
set.seed(55,sample.kind = 'Rejection')
mod_ran = randomForest(charges ~ .,
                   data = train,
                   mtry = sqrt(ncol(train)- 1),
                   importance = T,
                   ntree = 510)
ran_pred = predict(mod_ran,test)
(ran_perf = data.frame(
        MSE = mean((test$charges - ran_pred)^2),
        CORR = cor(test$charges,ran_pred)
))

# Gradient Boosting
library(gbm)
hyper_grid = expand.grid(
        shrinkage = c(0.005, .01, .1),
        interaction.depth = c(1, 3, 5,7)
) 
train$sex = factor(train$sex,
                   labels = c(0,1))
test$sex = factor(test$sex,
                  labels = c(0,1))
train$smoker = factor(train$smoker,
                   labels = c(0,1))
test$smoker = factor(test$smoker,
                  labels = c(0,1))

for(i in 1:nrow(hyper_grid)){
        
        print(paste("Iteration n.",i))
        set.seed(55, sample.kind="Rejection") #we use CV
        mod = gbm(formula = charges ~ . - region, 
                         data = train,
                         distribution = "gaussian",
                         n.trees = 5000, #B
                         shrinkage = hyper_grid$shrinkage[i], #lambda
                         interaction.depth = hyper_grid$interaction.depth[i], #d
                         cv.folds = 5)
        
        hyper_grid$minMSE[i] = min(mod$cv.error)
        hyper_grid$bestB[i] = which.min(mod$cv.error)
}
hyper_grid %>% 
        arrange(minMSE)
set.seed(55, sample.kind="Rejection") #we use CV
mod_gbm = gbm(formula = charges ~ .- region, 
                data = train,
                distribution = "gaussian",
                n.trees = 10000, #B
                shrinkage = 0.005, #lambda
                interaction.depth = 5, #d
                cv.folds = 5)
bestB = gbm.perf(mod_gbm) #find the best B
gbm_pred =predict(mod_gbm,test,n.trees = bestB) #test could be any new data
(gbm_perf = data.frame(
        MSE = mean((test$charges - gbm_pred)^2),
        CORR = cor(test$charges,gbm_pred)
))

perf = rbind(lm = lm_perf,
             gbm = gbm_perf,
             bag = bag_perf,
             ran = ran_perf,
             tree = tree_perf,
             pruned = pruned_tree_perf)
perf %>% 
        arrange(desc(CORR)) # best performance for Gradient Boosting Model