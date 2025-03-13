#! /usr/bin/Rscript
# XXX document & doctrings
suppressPackageStartupMessages(library(dplyr))
library(survival)

fitted_times <- read.csv('fitted_survival_times.csv', sep=',')

surv_est <- function(formula, data) {
    fit <- survfit(formula, data)                      # survival analysis (KM est.)
    fit.wbl <- survreg(formula, data, dist='weibull')  # survival regression
    
    return(
        list(
            fit=fit,
            fit.wbl=fit.wbl,
            lam.w=exp(-coef(fit.wbl)),
            gam=1/fit.wbl$scale
        )
    )
}


estimate_weibull_param <- function(data) {
    x <- surv_est(Surv(survival_time_hat, event_hat) ~ 1, data = data)
    return(list(w_shape = x$gam, w_scale = 1/x$lam.w))
}


variable <- fitted_times$variable_parameter[1]

results <- fitted_times %>%
    group_by(across(all_of(variable))) %>%
    do(data.frame(val = estimate_weibull_param(.))) %>%
    rename('weibull_shape_hat' = 'val.w_shape', 'weibull_scale_hat' = 'val.w_scale')


write.csv(results, 'survival_analysis.csv', row.names=FALSE)

## Only to plot the survival curve on the real dataset
#library(ggfortify)

fwbl <- function(t) { pweibull(t, shape=x$gam, scale=1/x$lam.w, low=FALSE) }

plot_surv <- function(x, ...) {
    xmax <- seq(0, max(x$fit$time), len=99)
    #print(x$gam)
    #print(1/x$lam.w)
    autoplot(x$fit, surv.colour = '#E69F00', censor.colour = '#BC6C25') +
       geom_function(fun = fwbl, colour = 'black', linetype='dashed') +
       xlab('time') + ylab('survival') +
       theme_bw() +
       theme(
           legend.title = element_blank(),
           legend.position = 'none',
           text=element_text(size=16),
       )
}

x <- surv_est(Surv(survival_time_hat, event_hat) ~ 1, data = fitted_times)
#print(summary(x$fit))
#print(summary(x$fit.wbl))
#p <- plot_surv(x)
#ggsave('survival_curve.pdf', p)
