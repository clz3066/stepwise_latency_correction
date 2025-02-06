rm(list=ls())
library(readxl)
library(lavaan)
library(regsem)
library(semPlot)

excel_file_path <- "~/Documents/GitHub/latency_variability/results/individual_multi_label/subjects.xlsx"
data <- read_excel(excel_file_path, sheet=1)

combined_step0 <- ' 
          # latent variable definitions
          FMA =~ macfb_A + mdrf_A + mewf_A    
          face_speed =~ Imdnmf_L + Imrsfb_L + Imverf_L + Immf_L
          
          # variance
          Facc_step ~~ Fsp_step
          FMA ~~ face_speed
          
          # regressions
          FMA ~ Facc_step + Fsp_step
          face_speed ~ Facc_step + Fsp_step '

fit <- sem(combined_step0, data=data, estimator="ML", missing="ml", std.lv=1, meanstructure=TRUE)
semPaths(fit, what='std', nCharNodes=0, fade=FALSE, residuals=TRUE, intercepts=FALSE)
results <- standardizedSolution(fit)
print(results)
print("step0")
print(results[c(10, 12, 11, 13), c(1,2,3,4,7)])


