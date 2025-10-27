from pathlib import Path
import sys

# Import all required classifiers
from sklearn.tree import DecisionTreeClassifier #type: ignore
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier #type: ignore
from sklearn.neighbors import KNeighborsClassifier #type: ignore
from sklearn.naive_bayes import GaussianNB #type: ignore
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis #type: ignore
from sklearn.svm import SVC #type: ignore
from sklearn.linear_model import LogisticRegression, Perceptron #type: ignore
from sklearn.neural_network import MLPClassifier #type: ignore

instances = {}

# Instance 1: Tree-Based Classifiers
instances["Tree_Based"] = {
    'classifiers': {
        'DecisionTree': DecisionTreeClassifier(),
        'RandomForest': RandomForestClassifier(),
        'ExtraTrees': ExtraTreesClassifier()
    },
    'grid_params': {
        'DecisionTree': {
            'clf__max_depth': [3, 5, 7, 10, 20],
            'clf__min_samples_split': [2, 5, 10],
            'clf__criterion': ['gini', 'entropy']
        },
        'RandomForest': {
            'clf__n_estimators': [50, 100, 200],
            'clf__max_depth': [3, 5, 7, 10],
            'clf__min_samples_split': [2, 5, 10]
        },
        'ExtraTrees': {
            'clf__n_estimators': [50, 100, 200],
            'clf__max_depth': [3, 5, 7, 10, 20],
            'clf__min_samples_split': [2, 5, 10]
        }
    }
}

# Instance 2: KNN Variants
instances["KNN_Variants"] = {
    'classifiers': {
        'KNN_Coarse': KNeighborsClassifier(),
        'KNN_Medium': KNeighborsClassifier(),
        'KNN_Fine': KNeighborsClassifier(),
        'KNN_Weighted': KNeighborsClassifier(),
        'KNN_Cosine': KNeighborsClassifier(metric='cosine')
    },
    'grid_params': {
        'KNN_Coarse': {
            'clf__n_neighbors': [10, 30, 50, 70, 90],
            'clf__weights': ['uniform', 'distance']
        },
        'KNN_Medium': {
            'clf__n_neighbors': [10, 15, 20, 25],
            'clf__weights': ['uniform', 'distance']
        },
        'KNN_Fine': {
            'clf__n_neighbors': [1, 2, 3],
            'clf__weights': ['uniform', 'distance']
        },
        'KNN_Weighted': {
            'clf__n_neighbors': [5, 10, 15],
            'clf__weights': ['distance']
        },
        'KNN_Cosine': {
            'clf__n_neighbors': [5, 10, 15],
            'clf__weights': ['uniform', 'distance']
        }
    }
}

# Instance 3: Probabilistic Classifiers
instances["Probabilistic"] = {
    'classifiers': {
        'GaussianNB': GaussianNB(),
        'LDA': LinearDiscriminantAnalysis()
    },
    'grid_params': {
        'GaussianNB': {
            'clf__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
        },
        'LDA': {
            'clf__solver': ['svd', 'lsqr', 'eigen'],
            'clf__shrinkage': [None, 'auto', 0.1, 0.5, 0.9]
        }
    }
}

# Instance 4: SVM Variants
instances["SVM_Variants"] = {
    'classifiers': {
        'LinearSVM': SVC(kernel='linear', probability=True),
        'QuadraticSVM': SVC(kernel='poly', degree=2, probability=True),
        'CubicSVM': SVC(kernel='poly', degree=3, probability=True),
        'RbfSVM': SVC(kernel='rbf', probability=True)
    },
    'grid_params': {
        'LinearSVM': {
            'clf__C': [0.1, 1, 10, 100]
        },
        'QuadraticSVM': {
            'clf__C': [0.1, 1, 10],
            'clf__gamma': ['scale', 'auto']
        },
        'CubicSVM': {
            'clf__C': [0.1, 1, 10],
            'clf__gamma': ['scale', 'auto']
        },
        'RbfSVM': {
            'clf__C': [0.1, 1, 10, 100],
            'clf__gamma': ['scale', 'auto']
        }
    }
}

# Instance 5: Logistic Regression variants
instances["Logistic_Regression"] = {
    'classifiers': {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'LogisticRegression_L1': LogisticRegression(penalty='l1', solver='saga', max_iter=1000),
        'LogisticRegression_ElasticNet': LogisticRegression(penalty='elasticnet', solver='saga', max_iter=1000)
    },
    'grid_params': {
        'LogisticRegression': {
            'clf__C': [0.001, 0.01, 0.1, 1, 10],
            'clf__penalty': ['l2']
        },
        'LogisticRegression_L1': {
            'clf__C': [0.001, 0.01, 0.1, 1, 10]
        },
        'LogisticRegression_ElasticNet': {
            'clf__C': [0.001, 0.01, 0.1, 1, 10],
            'clf__l1_ratio': [0.1, 0.5, 0.9]
        }
    }
}

# Instance 6: Perceptron and Neural Networks (NEW)
instances["Neural_Networks"] = {
    'classifiers': {
        'Perceptron': Perceptron(),
        'MLP_1Layer': MLPClassifier(max_iter=1000),
        'MLP_2Layer': MLPClassifier(max_iter=1000)
    },
    'grid_params': {
        'Perceptron': {
            'clf__alpha': [0.0001, 0.001, 0.01],
            'clf__penalty': [None, 'l2', 'l1', 'elasticnet']
        },
        'MLP_1Layer': {
            'clf__hidden_layer_sizes': [(50,), (100,), (200,)],
            'clf__alpha': [0.0001, 0.001, 0.01],
            'clf__learning_rate': ['constant', 'adaptive']
        },
        'MLP_2Layer': {
            'clf__hidden_layer_sizes': [(50, 25), (100, 50), (200, 100)],
            'clf__alpha': [0.0001, 0.001, 0.01],
            'clf__learning_rate': ['constant', 'adaptive']
        }
    }
}
