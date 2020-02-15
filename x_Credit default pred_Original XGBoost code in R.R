library(xgboost)
library(magrittr)
library(dplyr)
library(Matrix)
library(pdp)

setwd("G:/work/Loans")

smp_size <- floor(0.7 * nrow(data1))
set.seed(123)
train_ind <- sample(seq_len(nrow(data1)), size = smp_size)
train <- data1[train_ind, ]
test <- data1[-train_ind, ]

options(na.action='na.pass') # corrective action for handling NAs while building sparse model matrix

#create Sparse matrix
trainm<- sparse.model.matrix (Outcome~., data= train)
train_label <- train [ , 'Outcome']
train_matrix <- xgb.DMatrix (data= as.matrix(trainm), label = train_label)
testm<- sparse.model.matrix (Outcome~., data= test)
test_label <- test [ , 'Outcome']
test_matrix <- xgb.DMatrix (data= as.matrix(testm), label = test_label)


xgb_params <- list ( objective= " binary:logistic ",
eval_metric= "error")
watchlist <- list (train= train_matrix , test= test_matrix)
bst_model <- xgb.train(params = xgb_params,
data = train_matrix,
nrounds =100,
watchlist =watchlist,
eta=0.1, verbose = FALSE)
e <- data.frame (bst_model$ evaluation_log)

#plotting training error Vs test error
plot(e$iter, e$train_mlogloss, col="#2166AC")
lines (e$iter, e$test_mlogloss, col = "#B2182B")
min(e$test_mlogloss)
e[e$test_mlogloss == 0.637967, ]

#variable importance
imp <- xgb.importance ( colnames (train_matrix), model = bst_model)
print(imp)
xgb.plot.importance(imp)

#Predicting
p <- predict (bst_model, newdata = test_matrix)
pred <- matrix (p, nrow=nc, ncol = length(p)/nc) %>%
t() %>%
data.frame() %>%
mutate(label=test_label, max_prob = max.col(., 'las')-1)
table(prediction=pred$max_prob, Actual=pred$label)



#partial dependance plots
bst_model %>%
partial(pred.var = "Monthlyincome",center = TRUE, prob=TRUE,train = trainm) %>%
plotPartial(rug = TRUE, train = trainm)
ice
bst_model %>%
partial(pred.var = "Monthlyincome", ice = TRUE, prob=TRUE,train = trainm) %>%
plotPartial(rug = TRUE,alpha = 0.1, train = trainm)



#ROC Curve and Cutoff
ROC Method (ROCR package)
xgb.pred <- prediction(pred, test_label)
xgb.perf <- performance(xgb.pred, "tpr", "fpr")
plot(xgb.perf,
avg="threshold",
colorize=TRUE,
lwd=1,
main="ROC Curve w/ Thresholds",
print.cutoffs.at=seq(0, 1, by=0.05),
text.adj=c(-0.5, 0.5),
text.cex=0.5)
grid(col="lightgray")
axis(1, at=seq(0, 1, by=0.1))
axis(2, at=seq(0, 1, by=0.1))
abline(v=c(0.1, 0.3, 0.5, 0.7, 0.9), col="lightgray", lty="dotted")
abline(h=c(0.1, 0.3, 0.5, 0.7, 0.9), col="lightgray", lty="dotted")
lines(x=c(0, 1), y=c(0, 1), col="black", lty="dotted")

#Optimum cut-off
opt.cut = function (xgb.perf, xgb.pred) {
cut.ind = mapply (FUN=function(x,y,p){
d=(x-0)^2+(y-1)^2
ind=which(d==min(d))
c(sensitivity=y[[ind]], specificity=1-x[[ind]],
cutoff=p[[ind]])
}, xgb.perf@x.values, xgb.perf@y.values, xgb.pred@cutoffs)
}
print(opt.cut(xgb.perf, xgb.pred))

confusion Matrix (caret package)
threshold<-0.6656946
prednew <- ifelse(pred >= threshold, 1, 0)
confusionMatrix(table(prednew,test_label))

#Hyper-parameter tuning
# create hyperparameter grid
hyper_grid <- expand.grid(
eta = c(.01, .05, .1, .3),
max_depth = c(1, 3, 5, 7),
min_child_weight = c(1, 3, 5, 7),
subsample = c(.65, .8, 1),
colsample_bytree = c(.8, .9, 1),
optimal_trees = 0, # a place to dump results
min_logloss = 0 # a place to dump results
)
hyper_grid$optimal_trees <- NULL # corrective action for handling NAs
hyper_grid$min_logloss <- NULL # corrective action for handling NAs
# grid search
for(i in 1:nrow(hyper_grid)) {
# create parameter list
params <- list(
eta = hyper_grid$eta[i],
max_depth = hyper_grid$max_depth[i],
min_child_weight = hyper_grid$min_child_weight[i],
subsample = hyper_grid$subsample[i],
colsample_bytree = hyper_grid$colsample_bytree[i]
)
# reproducibility
set.seed(123)
# train model
tuned_model <- xgb.train(params = params,
data = train_matrix,
nrounds =1000,
nfold = 5,
objective = "binary:logistic",
watchlist =watchlist,
verbose = FALSE)
# add min training error and trees to grid
hyper_grid$optimal_trees[i] <- which.min(tuned_model$evaluation_log$test_error)
hyper_grid$min_logloss[i] <- min(tuned_model$evaluation_log$test_error)
}
#Outcome
hyper_grid %>%
dplyr::arrange(min_logloss) %>%
head(10)
After tuning fitting final Model
paramsfinal <- list(
eta =0.1,
max_depth =5 ,
min_child_weight =1 ,
subsample =0.8 ,
colsample_bytree = 0.8,
eval_metric= "error"
)

#final model
bst_modelfinal <- xgb.train(params = paramsfinal,
data = train_matrix,
objective= "binary:logistic",
nrounds =292,
watchlist =watchlist,
verbose = FALSE)
e <- data.frame (bst_modelfinal$ evaluation_log)
plot(e$iter, e$train_error, col="#2166AC", ylim=c(0,0.35))
lines (e$iter, e$test_error, col = "#B2182B")
