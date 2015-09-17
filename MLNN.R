
library(R6)

HiddenLayer <- R6Class('Hidden Layer',
                       public = list(
                           # init
                           initialize = function(n_in,n_out,f_act,f_deriv){
                               self$n_in <- n_in
                               self$n_out <- n_out
                               self$f_act <- f_act
                               self$f_deriv <- f_deriv
                               self$W <- matrix(runif(n_in*n_out, 
                                                      min = -sqrt(6/(n_in+n_out)),
                                                      max = sqrt(6/(n_in+n_out))),
                                                nrow = n_out)
                               self$b <- runif(n_out, 
                                               min = -sqrt(6/n_out),
                                               max = sqrt(6/n_out))
                           },
                           # propagate x to the next layer. x can be a matrix
                           forward_prop <- function(x){
                               act(self$W %*% x + self$b)
                           },
                           # backprop grad to prev layer. x,z has to be a vector
                           back_prop <- function(grad,z){
                               if(length(grad)!=self$n_out) stop('Gradient Term (grad) Wrong Size')
                               if(length(z)!=self$n_out) stop('Input Term (z) Wrong Size')
                               t(self$W) %*% (grad * self$f_deriv(z))
                           },
                           # update weights and bias
                           update_w <- function(grad_w, learning_rate){
                               self$W <- self$W - learning_rate*grad_w
                           },
                           update_b <- function(grad_b, learning_rate){
                               self$W <- self$W - learning_rate*grad_b
                           }),
                       # hidden layers' parameters
                       private = list(
                           n_in,
                           n_out,
                           W,
                           b,
                           f_act,
                           f_deriv)
)

MLNN <- R6Class('Multi-Layered Perceptron',
                public = list(),
                private = list())

