num_runs = 100

TT_prep_clean <- read.csv("/media/kap/8e22f6f8-c4df-4d97-a388-0adcae3ec1fb/Python/Thesis/TT_prep_clean.csv", header=FALSE)
KeepColRaw = read.csv("/media/kap/8e22f6f8-c4df-4d97-a388-0adcae3ec1fb/Python/Thesis/Results/tt_full.csv")

y <- TT_prep_clean[, 2]
y <- as.matrix(y)
y <- y-1
x <- TT_prep_clean[, 3:4562]
x <- as.matrix(x)

library("liso", lib.loc="~/R/x86_64-pc-linux-gnu-library/3.2")

KeepCol = as.vector(KeepColRaw[,1])
Xred = x[,KeepCol]
Xred = as.matrix(Xred)

res = matrix(nrow = num_runs, ncol = 2)

for(i in seq(1:num_runs)){
  sz = dim(Xred)[1]  # 805
  shuffleind = sample(seq(1:sz))
  TrainInd = shuffleind[1:500]
  TestInd = shuffleind[501:805]
  trnx = Xred[TrainInd,]
  trny = y[TrainInd]
  tstx = Xred[TestInd,]
  tsty = y[TestInd]
  
  CVobj <- cv.liso(trnx, trny)
  fitobj = liso.backfit(trnx, trny, CVobj$optimlam)
  
  y_hat = fitobj*tstx  # gives the prediction. Need to find a cutoff value
  y_m = y_hat >= 0.5
  pred_acc = sum((y_m - tsty) == 0)/305  # prediction accuracy is only 52.23%
  res[i, 1] = CVobj$optimlam
  res[i, 2] = pred_acc
}

# write to CSV
write.csv(res, file = 'R_Liso_res.csv', row.names = FALSE)
