num_runs = 201
set.seed(41)

x = seq(0,5,5/(num_runs-1))
x_sim = runif(num_runs, min = 0, max = 5)

y_t = function(x){
  sin(x) + cos(sqrt(3)*x)
}
y= y_t(x)  # truth data
# y_sim = y_t(x_sim) + rnorm(1001, mean = runif(1001,-.5,.5), sd=sin(.1*(x_sim-2.2)^2)+.8)
y_sim = y_t(x_sim) + rnorm(num_runs, mean = runif(num_runs,-.5,.5), sd=sin(.1*(y_t(x_sim)-2.2)^2)+cos(sqrt(abs(y_t(x_sim)))))

y_lin = rep(mean(y_sim), num_runs)

sfit = smooth.spline(x_sim, y=y_sim, df = 20)
y_spl = predict(sfit, x)
y_spl = y_spl$y

df = data.frame(x, y, x_sim, y_sim, y_lin, y_spl)
# over regularized
ggplot(df) + geom_path(aes(x=x, y=y), size=3, col='black') + geom_point(aes(x=x_sim, y=y_sim)) + geom_path(aes(x=x_sim, y=y_lin), col='red', size=2)

# under regularized
ggplot(df) + geom_path(aes(x=x, y=y), size=3, col='black') + geom_point(aes(x=x_sim, y=y_sim)) + geom_path(aes(x=x, y=y_spl), col='red', size=2)


summary(fit)
