rm(list=ls())
library(readxl)
library(lavaan)
library(regsem)
library(semPlot)
library(bruceR)
library(semTools)

excel_file_path <- "~/Documents/GitHub/latency_variability/results/individual_sem/subjects.xlsx"

data <- read_excel(excel_file_path, sheet=1)
# data[, c(2:7)] <- apply(data[, c(2:7)],  2, scale)


step0 <- ' 
          # latent variable definitions
          aFPA =~ msmfu_A + msmfi_A
          aFMA =~ macfb_A + mdrf_A + mewf_A    
          aface_speed =~ Imdnmf_L + Imrsfb_L + Imverf_L + Immf_L
          astep =~ step0_0a + step0_1a + step0_2a
          
          # regressions
          aFPA ~ astep
          aFMA ~ astep
          aface_speed ~ astep 

          bFPA =~ msmfu_A + msmfi_A
          bFMA =~ macfb_A + mdrf_A + mewf_A    
          bface_speed =~ Imdnmf_L + Imrsfb_L + Imverf_L + Immf_L
          bstep =~ step0_0s + step0_1s + step0_2s
          
          # regressions
          bFPA ~ bstep
          bFMA ~ bstep
          bface_speed ~ bstep 
          
          astep ~~ bstep '


fit0 <- sem(step0, data=data, estimator="ML", missing="ml", std.lv=1, meanstructure=TRUE)
print("step0")
lavaan_summary(fit0)
print(summary(fit0, fit.measures = TRUE, standardized = TRUE))
