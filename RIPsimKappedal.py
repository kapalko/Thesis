from numpy.random import randn, choice
from numpy.linalg import norm
from numpy import arange, ones, sqrt, dot, square, zeros

# Create a random Gaussian sensing matrix and make the columns unit length 
# (put your normalized sensing matrix here)

m = 100
n = 10000
X = randn(m,n)
j = 0
while j < n:
    X[:,j] = X[:,j]/norm(X[:,j],2)
    j += 1

# sets up max runs and max sparsity
maxRuns = 10
maxS = 10


ind = arange(0,n) # creates an array of column indexes to sample from
delta = zeros((maxRuns,maxS)) # initalizes the delta matrix (yes I know first 2 columns are zeros)

s = 2

# main loop

while s < maxS:
    run = 0
    while run < maxRuns:
        samp = choice(ind,s)
        vect = ones(s)/sqrt(s)
        delta[s,run] = abs(sum(square(dot(X[:,samp],vect))) - 1)
        run = run + 1
    s = s + 1

print "you are awesome"
