library(ggplot2)
tt_box <- read.csv("/media/kap/8e22f6f8-c4df-4d97-a388-0adcae3ec1fb/Python/Thesis/Results/aa_results/tt_box.csv")

x = tt_box$Parameter..c.
y = tt_box$Basis
x_new = rep(0, 1000)

for (i in 1:1000){
  if (x[i] <.2) {x_new[i]=.1}
  else if (x[i] <.4) {x_new[i]=.3}
  else if (x[i] <.6) {x_new[i]=.5}
  else if (x[i] <.8) {x_new[i]=.7}
  else if (x[i] <1.0){x_new[i]=.9}
  else if (x[i] <1.2){x_new[i]=1.1}
  else if (x[i] <1.4){x_new[i]=1.3}
  else if (x[i] <1.6){x_new[i]=1.5}
  else if (x[i] <1.8){x_new[i]=1.7}
  else if (x[i] <2.0){x_new[i]=1.9}
  else if (x[i] <2.2){x_new[i]=2.1}
  else if (x[i] <2.4){x_new[i]=2.3}
  else if (x[i] <2.6){x_new[i]=2.5}
  else if (x[i] <2.8){x_new[i]=2.7}
  else if (x[i] <3.0){x_new[i]=2.9}
  else if (x[i] <3.2){x_new[i]=3.1}
  else if (x[i] <3.4){x_new[i]=3.3}
  else if (x[i] <3.6){x_new[i]=3.5}
  else if (x[i] <3.8){x_new[i]=3.7}
  else if (x[i] <= 4.0){x_new[i]=3.9}
}
tt = data.frame(x_new, y)

ggplot(tt, aes(factor(x_new), y=y)) + geom_boxplot()+labs(x='Hyperparameter (C)', y = 'Number of Coefficients') + 
  theme(axis.text=element_text(size=12), axis.title=element_text(size=16, face='bold'))
