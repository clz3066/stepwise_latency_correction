rm(list=ls())
library(readxl)
library(lavaan)
library(regsem)
library(semPlot)
library(bruceR)

excel_file_path <- "~/Documents/GitHub/latency_variability/results/individual_sem/subject_Facc_s_r.xlsx"

data <- read_excel(excel_file_path, sheet=1)


s <- '    # latent variable definitions
          FPA =~ msmfu_A + msmfi_A
          FMA =~ macfb_A + mdrf_A + mewf_A    
          face_speed =~ Imdnmf_L + Imrsfb_L + Imverf_L + Immf_L
          step =~ s_component_0 + s_component_1 + s_component_2
          
          # regressions
          FPA ~ step
          FMA ~ step
          face_speed ~ step '

r <- ' 
          # latent variable definitions
          FMA =~ macfb_A + mdrf_A + mewf_A    
          face_speed =~ Imdnmf_L + Imrsfb_L + Imverf_L + Immf_L
          step =~ r_component_0 + r_component_1 + r_component_2
          
          # regressions
          FMA ~ step
          face_speed ~ step '


fit1 <- sem(s, data=data, estimator="ML", missing="ml", std.lv=1, meanstructure=TRUE)
print("s_component")
lavaan_summary(fit1)
#print(summary(fit, fit.measures = TRUE, standardized = TRUE))

#fit2 <- sem(r, data=data, estimator="ML", missing="ml", std.lv=1, meanstructure=TRUE)
#print("r_component")
#lavaan_summary(fit)
#print(summary(fit, fit.measures = TRUE, standardized = TRUE))


