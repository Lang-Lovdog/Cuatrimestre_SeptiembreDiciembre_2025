# En este archivo, se encuentra el pseudocódigo para la prueba estadística completa



```
 function statistical_analysis(opts)
   res["normality_test"][opts.ds_01_name] <- Normality_Test(opts.ds_01)
   res["overview"][opts.ds_01_name]       <- DatasetOverview(opts.ds_01)
   ds_02? entonces
     res["normality_test"][opts.ds_02_name] <- Shapiro-Wilk(opts.ds_02)
     res["overview"][opts.ds_02_name]       <- DatasetOverview(opts.ds_02) 
     res["variance_test"]   <- Variance_Test(opts, res)
     res["difference_test"] <- Statistical_Difference_Test(opts, res)
     dump(res)
   si no, entonces
     dump(res)
     exit


 function Normality_Test(ds)
   out <- Shapiro-Wilk(ds)
   out["method"] <- "Shapiro-Wilk"
   export qqplot(ds)
   export boxplot(ds)
   if out["p-value"] < 0.05 then
     out["normal"] <- false
   si no, entonces
     out["normal"] <- true
   return out

 function DatasetOverview(ds)
   out["samples"] <- length(ds)
   out["mean"]    <- mean(ds)
   out["median"]  <- median(ds)
   return out
 
 function Variance_Test(opts, res)
   res["normality_test"][opts.ds_01_name] and res["normality_test"][opts.ds_02_name] then
     out <- F-Test(opts.ds_01, opts.ds_02)
     out["method"] <- "F-Test"
     out["p-value"] < 0.05 then
       out["equal"] <- false
     si no, entonces
       out["equal"] <- true
     return out
   si no, entonces
     out <- Levene(opts.ds_01, opts.ds_02)
     out["method"] <- "Levene"
     out["p-value"] < 0.05 then
       out["equal"] <- false
     si no, entonces
       out["equal"] <- true
     return out

 function Statistical_Difference_Test(opts, res)
   res["normality_test"][opts.ds_01_name]["normal"] and res["normality_test"][opts.ds_02_name]["normal"] and res["variance_test"]["equal"] then
     out["alpha_grid"] <- alpha_optimizer(opts.ds_01, opts.ds_02, opts)
     out <- T-Test(opts.ds_01, opts.ds_02, res["alpha_grid"]["best_alpha"])
     out["method"] <- "T-Test"
     out["p-value"] < out["alpha_grid"]["best_alpha"] then
       out["equal"] <- false
     si no, entonces
       out["equal"] <- true
     return out
   si no, entonces
     out <- Wilcoxon(opts.ds_01, opts.ds_02)
     out["method"] <- "Wilcoxon"
     return out

 function alpha_optimizer(x,y,opts)
   alpha_list=opts["alpha"]["values"] or 
   P=opts["alpha"]["P"] or 0.4
   k=opts["alpha"]["k"] or 1
   sample_size=opts["alpha"]["sample_size"] or 6
   is_sampling_with_replacement=opts["alpha"]["sampling_with_replacement"] or TRUE
   iterations=opts["alpha"]["iterations"] or 500
   for alpha in alpha_list
     set.seed(random_integer)
     for i in 1 .. iterations
       sample_x <- sample(x, size = sample_size, replace = is_sampling_with_replacement)
       sample_y <- sample(y, size = sample_size, replace = is_sampling_with_replacement)
       pvalue <- t.test(x = sample_x, y=sample_y)$p.value
       pvalue <- alpha?
         rejection_count <- append(rejection_count,1)
       si no, entonces
         rejection_count <- append(rejection_count,0)
   
     rejected <- sum(rejection_count)
     total <- length(rejection_count)
   
     power <- rejected/total
     beta <- 1 - power
   
     beta_list <- append(beta_list,beta)
   
     expected_losses <- (P*alpha)+((1-P)*k*beta)
   
     expected_losses_list <- append(expected_losses_list,expected_losses)
   out <- { "alpha" : alpha_list, "beta" : beta_list, "expected_losses" : expected_losses_list }
   out["best_alpha"] <- out[expected_losses == min(expected_losses)]["alpha"]
   return out
```
