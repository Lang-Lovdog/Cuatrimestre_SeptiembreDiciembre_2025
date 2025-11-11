import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import json

# Set default style to avoid custom style issues
plt.style.use('default')

def statistical_analysis(opts):
    """
    Función principal para el análisis estadístico completo
    """
    res = {"normality_test": {}, "overview": {}}
    
    # Análisis del primer dataset
    res["normality_test"][opts["ds_01_name"]] = normality_test(opts["ds_01"], opts["ds_01_name"])
    res["overview"][opts["ds_01_name"]] = dataset_overview(opts["ds_01"])
    
    # Si existe segundo dataset, realizar análisis comparativo
    if "ds_02" in opts and opts["ds_02"] is not None:
        res["normality_test"][opts["ds_02_name"]] = normality_test(opts["ds_02"], opts["ds_02_name"])
        res["overview"][opts["ds_02_name"]] = dataset_overview(opts["ds_02"])
        res["variance_test"] = variance_test(opts, res)
        res["difference_test"] = statistical_difference_test(opts, res)
    
    # Guardar resultados
    save_results(res)
    return res

def normality_test(ds, ds_name=""):
    """
    Test de normalidad Shapiro-Wilk con gráficos
    """
    try:
        # Test Shapiro-Wilk
        stat, p_value = stats.shapiro(ds)
        
        # Crear gráficos con estilo simple
        plt.figure(figsize=(12, 4))
        
        # QQ-plot
        plt.subplot(1, 2, 1)
        stats.probplot(ds, dist="norm", plot=plt)
        plt.title("QQ-Plot")
        
        # Boxplot
        plt.subplot(1, 2, 2)
        plt.boxplot(ds)
        plt.title("Boxplot")
        
        plt.tight_layout()
        plt.savefig(f"normality_plots_{ds_name}.png", dpi=100, bbox_inches='tight')
        plt.close()
        
        return {
            "statistic": float(stat),
            "p_value": float(p_value),
            "method": "Shapiro-Wilk",
            "normal": p_value >= 0.05
        }
    except Exception as e:
        print(f"Error in normality_test: {e}")
        return {
            "statistic": None,
            "p_value": None,
            "method": "Shapiro-Wilk",
            "normal": None,
            "error": str(e)
        }

def dataset_overview(ds):
    """
    Resumen estadístico del dataset
    """
    return {
        "samples": len(ds),
        "mean": float(np.mean(ds)),
        "median": float(np.median(ds)),
        "std": float(np.std(ds)),
        "min": float(np.min(ds)),
        "max": float(np.max(ds))
    }

def variance_test(opts, res):
    """
    Test de igualdad de varianzas
    """
    try:
        ds1_normal = res["normality_test"][opts["ds_01_name"]]["normal"]
        ds2_normal = res["normality_test"][opts["ds_02_name"]]["normal"]
        
        if ds1_normal and ds2_normal:
            # F-test para varianzas
            var1 = np.var(opts["ds_01"], ddof=1)
            var2 = np.var(opts["ds_02"], ddof=1)
            f_value = max(var1, var2) / min(var1, var2)
            df1 = len(opts["ds_01"]) - 1
            df2 = len(opts["ds_02"]) - 1
            p_value = 2 * min(stats.f.cdf(f_value, df1, df2), 
                            1 - stats.f.cdf(f_value, df1, df2))
            
            result = {
                "statistic": float(f_value),
                "p_value": float(p_value),
                "method": "F-Test",
                "equal": p_value >= 0.05
            }
        else:
            # Test de Levene
            stat, p_value = stats.levene(opts["ds_01"], opts["ds_02"])
            result = {
                "statistic": float(stat),
                "p_value": float(p_value),
                "method": "Levene",
                "equal": p_value >= 0.05
            }
        
        return result
    except Exception as e:
        print(f"Error in variance_test: {e}")
        return {
            "statistic": None,
            "p_value": None,
            "method": "Unknown",
            "equal": None,
            "error": str(e)
        }

def statistical_difference_test(opts, res):
    """
    Test de diferencia estadística entre dos grupos
    """
    try:
        ds1_normal = res["normality_test"][opts["ds_01_name"]]["normal"]
        ds2_normal = res["normality_test"][opts["ds_02_name"]]["normal"]
        equal_variance = res.get("variance_test", {}).get("equal", False)
        
        if ds1_normal and ds2_normal:
            # T-test paramétrico
            alpha_grid = alpha_optimizer(opts["ds_01"], opts["ds_02"], opts)
            best_alpha = alpha_grid["best_alpha"]
            
            stat, p_value = stats.ttest_ind(opts["ds_01"], opts["ds_02"], equal_var=equal_variance)
            
            result = {
                "statistic": float(stat),
                "p_value": float(p_value),
                "method": "T-Test",
                "alpha_used": float(best_alpha),
                "equal": p_value >= best_alpha
            }
        else:
            # Test de Wilcoxon (no paramétrico)
            stat, p_value = stats.mannwhitneyu(opts["ds_01"], opts["ds_02"])
            result = {
                "statistic": float(stat),
                "p_value": float(p_value),
                "method": "Wilcoxon",
                "equal": p_value >= 0.05
            }
        
        return result
    except Exception as e:
        print(f"Error in statistical_difference_test: {e}")
        return {
            "statistic": None,
            "p_value": None,
            "method": "Unknown",
            "equal": None,
            "error": str(e)
        }

def alpha_optimizer(x, y, opts):
    """
    Optimizador del valor alpha para tests estadísticos
    """
    try:
        # Parámetros por defecto
        alpha_params = opts.get("alpha", {})
        alpha_list = alpha_params.get("values", [0.01, 0.05, 0.1])
        P = alpha_params.get("P", 0.4)
        k = alpha_params.get("k", 1)
        sample_size = alpha_params.get("sample_size", 6)
        replacement = alpha_params.get("sampling_with_replacement", True)
        iterations = alpha_params.get("iterations", 500)
        
        alpha_results = []
        beta_list = []
        expected_losses_list = []
        
        for alpha in alpha_list:
            rejection_count = []
            
            for i in range(iterations):
                # Muestreo
                sample_x = np.random.choice(x, size=sample_size, replace=replacement)
                sample_y = np.random.choice(y, size=sample_size, replace=replacement)
                
                # Test t
                _, p_value = stats.ttest_ind(sample_x, sample_y)
                
                # Contar rechazos
                rejection_count.append(1 if p_value < alpha else 0)
            
            # Cálculo de potencia y error tipo II
            rejected = np.sum(rejection_count)
            power = rejected / iterations
            beta = 1 - power
            
            # Pérdida esperada
            expected_loss = (P * alpha) + ((1 - P) * k * beta)
            
            beta_list.append(beta)
            expected_losses_list.append(expected_loss)
            alpha_results.append({
                "alpha": float(alpha),
                "beta": float(beta),
                "expected_loss": float(expected_loss),
                "power": float(power)
            })
        
        # Encontrar mejor alpha (menor pérdida esperada)
        best_idx = np.argmin(expected_losses_list)
        best_alpha = alpha_list[best_idx]
        
        return {
            "alpha_grid": alpha_results,
            "best_alpha": float(best_alpha),
            "min_expected_loss": float(expected_losses_list[best_idx])
        }
    except Exception as e:
        print(f"Error in alpha_optimizer: {e}")
        return {
            "alpha_grid": [],
            "best_alpha": 0.05,
            "min_expected_loss": None,
            "error": str(e)
        }

def safe_serialize(obj):
    """
    Función segura para serializar objetos a JSON
    """
    if obj is None:
        return None
    elif isinstance(obj, (bool, int, float, str)):
        return obj
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: safe_serialize(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [safe_serialize(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        return safe_serialize(obj.__dict__)
    else:
        return str(obj)

def save_results(res):
    """
    Guardar resultados en archivo JSON de forma segura
    """
    try:
        # Serializar de forma segura
        serializable_res = safe_serialize(res)
        
        with open("statistical_analysis_results.json", "w") as f:
            json.dump(serializable_res, f, indent=2, ensure_ascii=False)
        print("Results successfully saved to statistical_analysis_results.json")
    except Exception as e:
        print(f"Error saving results: {e}")
        # Fallback: save basic information
        basic_results = {
            "overview": res.get("overview", {}),
            "normality_summary": {
                name: {
                    "normal": test.get("normal"),
                    "p_value": test.get("p_value")
                }
                for name, test in res.get("normality_test", {}).items()
            }
        }
        with open("statistical_analysis_basic_results.json", "w") as f:
            json.dump(safe_serialize(basic_results), f, indent=2)

# Ejemplo de uso
if __name__ == "__main__":
    # Datos de ejemplo
    np.random.seed(42)
    data1 = np.random.normal(0, 1, 100)
    data2 = np.random.normal(0.5, 1.2, 100)
    
    opts = {
        "ds_01": data1,
        "ds_01_name": "grupo_control",
        "ds_02": data2,
        "ds_02_name": "grupo_tratamiento",
        "alpha": {
            "values": [0.01, 0.05, 0.1],
            "P": 0.4,
            "k": 1,
            "sample_size": 10,
            "iterations": 100
        }
    }
    
    try:
        resultados = statistical_analysis(opts)
        print("Análisis completado exitosamente!")
    except Exception as e:
        print(f"Error durante el análisis: {e}")
