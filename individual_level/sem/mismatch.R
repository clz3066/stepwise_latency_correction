rm(list=ls())
library(readxl)
library(lavaan)
library(regsem)
library(semPlot)
library(corrplot)
excel_file_path <- "~/Documents/GitHub/latency_variability/results/individual_sem/subject_Fsp_c.xlsx"
data <- read_excel(excel_file_path, sheet=1)

step4 <- ' 
          # latent variable definitions
          FMA =~ macfb_A + mdrf_A + mewf_A    
          face_speed =~ Imdnmf_L + Imrsfb_L + Imverf_L + Immf_L
          step =~ step4_0 + step4_1 + step4_2
          
          # regressions
          FMA ~ step
          face_speed ~ step '

fit <- sem(step4, data=data, estimator="ML", missing="ml", std.lv=1, meanstructure=TRUE)
semPaths(fit, what='std', nCharNodes=0, fade=FALSE, residuals=TRUE, intercepts=FALSE)
results <- standardizedSolution(fit)
print("step0")
print(inspect(fit, what="cor.all"))
resid(fit, "cor")
plot_matrix <- function(matrix_toplot){
  corrplot::corrplot(matrix_toplot, is.corr = FALSE,
                     type = 'lower',
                     order = "original",
                     tl.col='black', tl.cex=.75)
}
r <- resid(fit, type="cor")$cov
plot_matrix(r)
