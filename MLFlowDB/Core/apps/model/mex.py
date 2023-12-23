from enum import Enum
from typing import List

from mongoengine import DoesNotExist

from apps.model.mldr_model.basic import VersionedEntity
from apps.model.prov_dm import Entity
import mongoengine as me

from utils.utils import have_intersection

class Language(Enum):
    C = "C"
    CPLUSPLUS = "C++"
    CENTURA = "Centura"
    DOTNET = "DotNet"
    JAVA = "Java"
    JavaScript = "JavaScript"
    NodeJS = "NodeJS"
    Other = "Other"
    PHP = "PHP"
    Python = "Python"
    Ruby = "Ruby"
    R = "R"


class MachineLearningProblemType(Enum):
    UNKNOWN = "unknown"
    CLASSIFICATION = "classification"
    STATISTICAL = "statistical"
    CLUSTERING = "clustering"
    REGRESSION = "regression"
    ASSOCIATION = "association"
    METAHEURISTIC = "metaheuristic"

    @property
    def namespace(self):
        if self is MachineLearningProblemType.ASSOCIATION or self is MachineLearningProblemType.METAHEURISTIC:
            return f"{MachineLearningProblemType.STATISTICAL.value}::"
        return f"{self.value}::"

    @classmethod
    def get_values(cls):
        return [member.value for member in cls]


class MachineLearningMethodType(Enum):
    UNKNOWN = "Unknown"
    REINFORCEMENT = "Reinforcement"
    SEMI_SUPERVISED = "SemiSupervised"
    SUPERVISED = "Supervised"
    UNSUPERVISED = "Unsupervised"

    @classmethod
    def get_values(cls):
        return [member.value for member in cls]



class MachineLearningEvaluationMeasuring(Enum):
    '''
    Export from http://mex.aksw.org/mex-algo
    '''

    UNKNOWN = "Unknown"

    '''
    Classification
    '''
    ACCURACY = "accuracy"
    ERROR = "error"
    F1_MEASURE = "f1Measure"
    FALSE_NEGATIVE = "falseNegative"
    FALSE_NEGATIVE_RATE = "falseNegativeRate"
    FALSE_POSITIVE = "falsePositive"
    FALSE_POSITIVE_RATE = "falsePositiveRate"
    TRUE_NEGATIVE = "trueNegative"
    TRUE_NEGATIVE_RATE = "trueNegativeRate"
    TRUE_POSITIVE = "truePositive"
    TRUE_POSITIVE_RATE = "truePositiveRate"
    PRECISION = "precision"
    RECALL = "recall"
    ROC = "ROC"
    SENSITIVITY = "sensitivity"

    '''
    Statistical
    '''
    BONFERRONI_DUNN = "bonferroniDunn"
    FRIEDMAN = "friedman"
    KAPPA_STATISTICS = "kappaStatistics"
    KOLMOGOROV_SMIRNOV = "kolmogorovSmirnov"
    L1_NORM = "L1norm"
    L2_NORM = "L2norm"
    LINF_NORM = "Linfnorm"
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    NEMENYI = "nemenyi"
    PEARSON_CORRELATION = "pearsonCorrelation"
    EXAMPLE = "example"
    SPMI = "spmi"
    STANDARD_DEVIATION = "standardDeviation"
    VARIANCE = "variance"
    WILCOXON = "wilcoxon"

    '''
    Clustering
    '''
    CHEBYSHEV_DISTANCE = "chebyschevDistance"
    COSINE = "cosine"
    EUCLIDEAN_DISTANCE = "euclideanDistance"
    GEN_SIMILARITY_COEFFICIENT = "genSimilarityCoefficient"
    HAMMING_DISTANCE = "hammingDistance"
    MANHATTAN_DISTANCE = "manhattanDistance"

    '''
    Regression
    '''
    CORRELATION_COEFFICIENT = "correlationCoeffcient"
    MEAN_ABSOLUTE_DEVIATION = "meanAbsoluteDeviation"
    MEAN_SQUARED_ERROR = "meanSquaredError"
    MEDIAN_ABSOLUTE_DEVIATION = "medianAbsoluteDeviation"
    RELATIVE_ABSOLUTE_ERROR = "relativeAbsoluteError"
    RESIDUAL = "residual"
    ROOT_MEAN_SQUARED_ERROR = "rootMeanSquaredError"
    ROOT_RELATIVE_SQUARED_ERROR = "rootRelativeSquaredError"
    TOTAL_ERROR = "totalError"
    R2 = "r2"

    USER_CLASS = "UserDefined"

    @staticmethod
    def classification_measuring():
        return [MachineLearningEvaluationMeasuring.ACCURACY,
                MachineLearningEvaluationMeasuring.ERROR,
                MachineLearningEvaluationMeasuring.F1_MEASURE,
                MachineLearningEvaluationMeasuring.FALSE_NEGATIVE,
                MachineLearningEvaluationMeasuring.FALSE_NEGATIVE_RATE,
                MachineLearningEvaluationMeasuring.FALSE_POSITIVE,
                MachineLearningEvaluationMeasuring.FALSE_POSITIVE_RATE,
                MachineLearningEvaluationMeasuring.TRUE_NEGATIVE,
                MachineLearningEvaluationMeasuring.TRUE_NEGATIVE_RATE,
                MachineLearningEvaluationMeasuring.TRUE_POSITIVE,
                MachineLearningEvaluationMeasuring.TRUE_POSITIVE_RATE,
                MachineLearningEvaluationMeasuring.PRECISION,
                MachineLearningEvaluationMeasuring.RECALL,
                MachineLearningEvaluationMeasuring.ROC,
                MachineLearningEvaluationMeasuring.SENSITIVITY]

    @staticmethod
    def statistical_measuring():
        return [MachineLearningEvaluationMeasuring.BONFERRONI_DUNN,
                MachineLearningEvaluationMeasuring.FRIEDMAN,
                MachineLearningEvaluationMeasuring.KAPPA_STATISTICS,
                MachineLearningEvaluationMeasuring.KOLMOGOROV_SMIRNOV,
                MachineLearningEvaluationMeasuring.L1_NORM,
                MachineLearningEvaluationMeasuring.L2_NORM,
                MachineLearningEvaluationMeasuring.LINF_NORM,
                MachineLearningEvaluationMeasuring.MEAN,
                MachineLearningEvaluationMeasuring.MEDIAN,
                MachineLearningEvaluationMeasuring.MODE,
                MachineLearningEvaluationMeasuring.NEMENYI,
                MachineLearningEvaluationMeasuring.PEARSON_CORRELATION,
                MachineLearningEvaluationMeasuring.EXAMPLE,
                MachineLearningEvaluationMeasuring.SPMI,
                MachineLearningEvaluationMeasuring.STANDARD_DEVIATION,
                MachineLearningEvaluationMeasuring.VARIANCE,
                MachineLearningEvaluationMeasuring.WILCOXON]

    @staticmethod
    def clustering_measuring():
        return [MachineLearningEvaluationMeasuring.CHEBYSHEV_DISTANCE,
                MachineLearningEvaluationMeasuring.COSINE,
                MachineLearningEvaluationMeasuring.EUCLIDEAN_DISTANCE,
                MachineLearningEvaluationMeasuring.GEN_SIMILARITY_COEFFICIENT,
                MachineLearningEvaluationMeasuring.HAMMING_DISTANCE,
                MachineLearningEvaluationMeasuring.MANHATTAN_DISTANCE]

    @staticmethod
    def regression_measuring():
        return [MachineLearningEvaluationMeasuring.CORRELATION_COEFFICIENT,
                MachineLearningEvaluationMeasuring.MEAN_ABSOLUTE_DEVIATION,
                MachineLearningEvaluationMeasuring.MEAN_SQUARED_ERROR,
                MachineLearningEvaluationMeasuring.MEDIAN_ABSOLUTE_DEVIATION,
                MachineLearningEvaluationMeasuring.RELATIVE_ABSOLUTE_ERROR,
                MachineLearningEvaluationMeasuring.RESIDUAL,
                MachineLearningEvaluationMeasuring.ROOT_MEAN_SQUARED_ERROR,
                MachineLearningEvaluationMeasuring.ROOT_RELATIVE_SQUARED_ERROR,
                MachineLearningEvaluationMeasuring.TOTAL_ERROR,
                MachineLearningEvaluationMeasuring.R2]

    @property
    def value_with_namespace(self):
        type = ""
        if self in MachineLearningEvaluationMeasuring.classification_measuring():
            type = MachineLearningProblemType.CLASSIFICATION.namespace
        elif self in MachineLearningEvaluationMeasuring.regression_measuring():
            type = MachineLearningProblemType.REGRESSION.namespace
        elif self in MachineLearningEvaluationMeasuring.clustering_measuring():
            type = MachineLearningProblemType.CLUSTERING.namespace
        elif self in MachineLearningEvaluationMeasuring.clustering_measuring():
            type = MachineLearningProblemType.STATISTICAL.namespace
        return type + self.value()

    @staticmethod
    def measuring_class(name:str):
        if not MachineLearningEvaluationMeasuring.is_member(name):
            raise ValueError(f"{name} is not a MachineLearningEvaluationMeasuring. You can defined one.")
        type = 'Unknown'
        if not MachineLearningEvaluationMeasuring.is_user_defined(name):
            mea = MachineLearningEvaluationMeasuring(name)
            if mea in MachineLearningEvaluationMeasuring.classification_measuring():
                type = MachineLearningProblemType.CLASSIFICATION.value
            elif mea in MachineLearningEvaluationMeasuring.regression_measuring():
                type = MachineLearningProblemType.REGRESSION.value
            elif mea in MachineLearningEvaluationMeasuring.clustering_measuring():
                type = MachineLearningProblemType.CLUSTERING.value
            elif mea in MachineLearningEvaluationMeasuring.clustering_measuring():
                type = MachineLearningProblemType.STATISTICAL.value
        else:
            name_class = name.split("::")[-1]
            udm = UserDefinedMeasure.get(name_class)
            return udm.measuring_class
        return type

    @staticmethod
    def is_member(name:str):
        for member in MachineLearningEvaluationMeasuring:
            if member.value == name:
                return True
            else:
                if MachineLearningEvaluationMeasuring.is_user_defined(name):
                    name_class = name.split("::")[-1]
                    if UserDefinedMeasure.get(name_class) is not None:
                        return True
        return False

    @staticmethod
    def is_user_defined(name:str):
        if name.startswith(MachineLearningEvaluationMeasuring.USER_CLASS.value):
            return True
        return False


class UserDefinedMeasure(VersionedEntity):
    name = me.StringField()
    measuring_class = me.StringField(default="Unknown")
    formula = me.StringField(null=False)

    @staticmethod
    def get(name):
        try:
            return UserDefinedMeasure.objects.get(name=name)
        except DoesNotExist:
            return None

    @staticmethod
    def get_or_generate(name,formula=None):
        try:
            return UserDefinedMeasure.objects.get(name=name)
        except DoesNotExist:
            return UserDefinedMeasure.objects.create(name=name,formula=formula)

class UserDefinedAlgorithmClass(VersionedEntity):
    name = me.StringField()

    @staticmethod
    def get(name):
        try:
            return UserDefinedAlgorithmClass.objects.get(name=name)
        except DoesNotExist:
            return None

    @staticmethod
    def get_or_generate(name):
        ud = UserDefinedAlgorithmClass.get(name)
        if ud is None:
            return UserDefinedAlgorithmClass.objects.create(name=name)

class UserDefinedTool(VersionedEntity):
    name = me.StringField()

    @staticmethod
    def get(name):
        try:
            return UserDefinedTool.objects.get(name=name)
        except DoesNotExist:
            return None

    @staticmethod
    def get_or_generate(name):
        ud = UserDefinedTool.get(name)
        if ud is None:
            return UserDefinedTool.objects.create(name=name)

class MachineLearningAlgorithmClassType(Enum):
    UNKNOWN = "Unknown"

    AD_TREE = "ADTree"
    AQ = "AQ"
    ADAPTATIVE_BOOST = "AdaptativeBoost"
    APRIORI = "Apriori"
    ARTIFICIAL_NEURAL_NETWORK = "ArtificialNeuralNetwork"
    AUTOREGRESSIVE_INTEGRATED_MOVINGAVERAGE = "AutoregressiveIntegratedMovingAverage"
    AUTOREGRESSIVE_MOVING_AVERAGE = "AutoregressiveMovingAverage"
    AVERAGE_ONE_DEPENDENCE_ESTIMATORS = "AverageOneDependenceEstimators"
    BF_TREE = "BFTree"
    BIRCH = "BIRCH"
    BACK_PROPAGATION = "BackPropagation"
    BAGGING = "Bagging"
    BASELINE = "Baseline"
    BAYES_THEORY = "BayesTheory"
    BOOSTING = "Boosting"
    CART = "CART"
    CHAID = "CHAID"
    CHAMELEON = "CHAMELEON"
    CLARA = "CLARA"
    CLARANS = "CLARANS"
    CURE = "CURE"
    CLUSTERING = "Clustering"
    DECISION_STUMP = "DecisionStump"
    DECISION_TABLE = "DecisionTable"
    DECISION_TREES = "DecisionTrees"
    C45 = "C45"
    J48 = "J48"
    J48GRAFT = "J48Graft"
    RANDOM_FOREST = "RandomForest"
    RANDOM_TREE = "RandomTree"
    ELTL = "ELTL"
    FP = "FP"
    GENETIC_ALGORITHMS = "GeneticAlgorithms"
    HYBRID_ALGORITHM = "HybridAlgorithm"
    ID3 = "ID3"
    INDUCE = "INDUCE"
    INDUCTIVE_LOGIC_PROGRAMMING = "InductiveLogicProgramming"
    KMEANS = "Kmeans"
    LAD_TREE = "LADTree"
    LMT = "LMT"
    LOGICAL_REPRESENTATIONS = "LogicalRepresentations"
    MARS = "MARS"
    MARKOV = "Markov"
    MULTILAYER_PERCEPTRON = "MultilayerPerceptron"
    NB_TREE = "NBTree"
    NEAREST_NEIGBOUR = "NearestNeigbour"
    OPTICS = "OPTICS"
    PROBABILISTIC_MODEL = "ProbabilisticModel"
    NAIVEBAYES = "NaiveBayes"
    PROBABILISTIC_SOFT_LOGIC = "ProbabilisticSoftLogic"
    REP_TREE = "REPTree"
    REGRESSION_ANALYSIS = "RegressionAnalysis"
    LINE_ARREGRESSION = "LinearRegression"
    LOGISTIC_REGRESSION = "LogisticRegression"
    REGRESSION_FUNCTIONS = "RegressionFunctions"
    RULES = "Rules"
    SEQUENTIAL_MINIMAL_OPTIMIZATION = "SequentialMinimalOptimization"
    LINEAR_SMO = "LinearSMO"
    SIMPLE_CART = "SimpleCart"
    SUPPORT_VECTOR_MACHINES = "SupportVectorMachines"
    C_SVM = "C-SVM"
    LINEAR_SVM = "Linear-SVM"
    POLYNOMIAL_SVM = "Polynomial-SVM"
    R_SVM = "R-SVM"
    RBF_SVM = "RBF-SVM"
    SIGMOID_SVM = "Sigmoid-SVM"
    SUPPORT_VECTOR_NETWORKS = "SupportVectorNetworks"

    USER_CLASS = "UserDefined"

    @classmethod
    def get_values(cls):
        return [member.value for member in cls]

    @staticmethod
    def is_member(name:str):
        for member in MachineLearningAlgorithmClassType:
            if member.value == name:
                return True
            else:
                if name.startswith(MachineLearningAlgorithmClassType.USER_CLASS.value):
                    name_class = name.split("::")[-1]
                    if UserDefinedAlgorithmClass.get(name_class) is not None:
                        return True
        return False


class MachineLearningAlgorithmClass():
    '''
    Export from http://mex.aksw.org/mex-algo
    '''
    relations = {
        'AlgorithmClass': {'disjoint': ['HyperParameter', 'LearningMethod', 'LearningProblem'], 'subclass_of': []},
        'ADTree': {'disjoint': [], 'subclass_of': []}, 'AQ': {'disjoint': [], 'subclass_of': []},
        'AdaptativeBoost': {'disjoint': [], 'subclass_of': []}, 'Apriori': {'disjoint': [], 'subclass_of': []},
        'ArtificialNeuralNetwork': {'disjoint': [], 'subclass_of': []},
        'AutoregressiveIntegratedMovingAverage': {'disjoint': [], 'subclass_of': []},
        'AutoregressiveMovingAverage': {'disjoint': [], 'subclass_of': []},
        'AverageOneDependenceEstimators': {'disjoint': [], 'subclass_of': []},
        'BFTree': {'disjoint': [], 'subclass_of': []}, 'BIRCH': {'disjoint': [], 'subclass_of': []},
        'BackPropagation': {
            'disjoint': ['C45', 'LogisticRegression', 'NaiveBayes', 'Kmeans', 'ELTL', 'SupportVectorMachines',
                         'RegressionAnalysis', 'RandomForest'], 'subclass_of': []},
        'Bagging': {'disjoint': [], 'subclass_of': []}, 'Baseline': {'disjoint': [], 'subclass_of': []},
        'BayesTheory': {'disjoint': [], 'subclass_of': []}, 'Boosting': {'disjoint': [], 'subclass_of': []},
        'CART': {'disjoint': [], 'subclass_of': []}, 'CHAID': {'disjoint': [], 'subclass_of': []},
        'CHAMELEON': {'disjoint': [], 'subclass_of': []}, 'CLARA': {'disjoint': [], 'subclass_of': []},
        'CLARANS': {'disjoint': [], 'subclass_of': []}, 'CURE': {'disjoint': [], 'subclass_of': []},
        'Clustering': {'disjoint': [], 'subclass_of': []}, 'DecisionStump': {'disjoint': [], 'subclass_of': []},
        'DecisionTable': {'disjoint': [], 'subclass_of': []}, 'DecisionTrees': {'disjoint': [], 'subclass_of': []},
        'C45': {
            'disjoint': ['LogisticRegression', 'NaiveBayes', 'Kmeans', 'SupportVectorMachines', 'RegressionAnalysis',
                         'RandomForest'], 'subclass_of': ['DecisionTrees']},
        'J48': {'disjoint': [], 'subclass_of': ['DecisionTrees']},
        'J48Graft': {'disjoint': [], 'subclass_of': ['DecisionTrees']},
        'RandomForest': {'disjoint': ['RegressionAnalysis', 'SupportVectorMachines'], 'subclass_of': ['DecisionTrees']},
        'RandomTree': {'disjoint': ['RegressionAnalysis', 'SupportVectorMachines'], 'subclass_of': ['DecisionTrees']},
        'ELTL': {
            'disjoint': ['LogisticRegression', 'NaiveBayes', 'Kmeans', 'SupportVectorMachines', 'RegressionAnalysis',
                         'RandomForest'], 'subclass_of': []}, 'FP': {'disjoint': [], 'subclass_of': []},
        'GeneticAlgorithms': {'disjoint': [], 'subclass_of': []},
        'HybridAlgorithm': {'disjoint': [], 'subclass_of': []},
        'ID3': {'disjoint': [], 'subclass_of': []}, 'INDUCE': {'disjoint': [], 'subclass_of': []},
        'InductiveLogicProgramming': {'disjoint': [], 'subclass_of': []}, 'Kmeans': {
            'disjoint': ['LogisticRegression', 'NaiveBayes', 'SupportVectorMachines', 'RegressionAnalysis',
                         'RandomForest'],
            'subclass_of': []}, 'LADTree': {'disjoint': [], 'subclass_of': []},
        'LMT': {'disjoint': [], 'subclass_of': []},
        'LogicalRepresentations': {'disjoint': [], 'subclass_of': []}, 'MARS': {'disjoint': [], 'subclass_of': []},
        'Markov': {'disjoint': [], 'subclass_of': []}, 'MultilayerPerceptron': {'disjoint': [], 'subclass_of': []},
        'NBTree': {'disjoint': [], 'subclass_of': []}, 'NearestNeigbour': {'disjoint': [], 'subclass_of': []},
        'OPTICS': {'disjoint': [], 'subclass_of': []}, 'ProbabilisticModel': {'disjoint': [], 'subclass_of': []},
        'NaiveBayes': {'disjoint': ['RegressionAnalysis', 'SupportVectorMachines', 'RandomForest'],
                       'subclass_of': ['ProbabilisticModel']},
        'ProbabilisticSoftLogic': {'disjoint': [], 'subclass_of': []}, 'REPTree': {'disjoint': [], 'subclass_of': []},
        'RegressionAnalysis': {'disjoint': ['SupportVectorMachines'], 'subclass_of': []},
        'LinearRegression': {'disjoint': [], 'subclass_of': ['RegressionAnalysis']},
        'LogisticRegression': {'disjoint': ['SupportVectorMachines', 'NaiveBayes', 'RandomForest'],
                               'subclass_of': ['RegressionAnalysis']},
        'RegressionFunctions': {'disjoint': [], 'subclass_of': []}, 'Rules': {'disjoint': [], 'subclass_of': []},
        'SequentialMinimalOptimization': {'disjoint': [], 'subclass_of': []},
        'LinearSMO': {'disjoint': [], 'subclass_of': ['SequentialMinimalOptimization']},
        'SimpleCart': {'disjoint': [], 'subclass_of': []}, 'SupportVectorMachines': {'disjoint': [], 'subclass_of': []},
        'C-SVM': {'disjoint': [], 'subclass_of': ['SupportVectorMachines']},
        'Linear-SVM': {'disjoint': [], 'subclass_of': ['SupportVectorMachines']},
        'Polynomial-SVM': {'disjoint': [], 'subclass_of': ['SupportVectorMachines']},
        'R-SVM': {'disjoint': [], 'subclass_of': ['SupportVectorMachines']},
        'RBF-SVM': {'disjoint': [], 'subclass_of': ['SupportVectorMachines']},
        'Sigmoid-SVM': {'disjoint': [], 'subclass_of': ['SupportVectorMachines']},
        'SupportVectorNetworks': {'disjoint': [], 'subclass_of': []},
        'UserClassifier': {'disjoint': [], 'subclass_of': []}}

    @staticmethod
    def check_disjoint(lists: List[MachineLearningAlgorithmClassType]):
        lists_value = [l.value for l in lists]
        for c in lists_value:
            disjoints = MachineLearningAlgorithmClass.relations.get(c, {'disjoint': [], 'subclass_of': []})["disjoint"]
            if have_intersection(disjoints, lists_value) and c not in disjoints:
                return False,f"{c} has disjoints: {str(disjoints)}, but you have {str(have_intersection(disjoints, lists_value))}."
        return True,""

class MachineLearningAlgorithmToolType(Enum):
    APACHE_MAHOUT = "Apache Mahout"
    DL_LEARNER = "DLLearner"
    ELKI = "ELKI"
    E_VIEWS = "EViews"
    ENCOG = "Encog"
    FAMA = "FAMa"
    H2O = "H2O"
    IBM_MINER = "IBM Miner"
    JULIA = "Julia"
    KNIME = "KNIME"
    KXEN = "KXEN"
    LION_SOLVER = "LIONsolver"
    LIB_LINEAR = "LibLinear"
    LIB_SVM = "LibSVM"
    ML_PACK = "MLPACK"
    MASSIVE_ONLINE_ANALYSIS = "Massive Online Analysis"
    MATHEMATICA = "Mathematica"
    MATLAB = "Matlab"
    MICROSOFT_AZURE_MACHINE_LEARNING = "Microsoft Azure Machine Learning"
    MONTE_CARLO_MACHINE_LEARNING_LIBRARY = "Monte Carlo Machine Learning Library"
    NEURO_SOLUTIONS = "NeuroSolutions"
    OCTAVE = "Octave"
    OPENCV = "OpenCV"
    OPENNN = "OpenNN"
    ORACLE_DATA_MINING = "Oracle Data Mining"
    ORANGE = "Orange"
    RCASE = "RCASE"
    RAPID_MINER = "RapidMiner"
    SAP = "SAP"
    SAS_ENTERPRISE_MINER = "SAS Enterprise Miner"
    SPSS = "SPSS"
    SQL_SERVER_ANALYSIS_SERVICES = "SQL Server Analysis Services"
    STATISTICA_DATA_MINER = "STATISTICA Data Miner"
    SHOGUN = "Shogun"
    STATA = "Stata"
    WEKA = "Weka"
    YALE = "YALE"
    YOOREEKA = "Yooreeka"
    MLPY = "mlpy"
    SCIKIT_LEARN = "scikit-learn"
    PYTORCH = "pytorch"
    TENSORFLOW = "Tensorflow"
    KERAS = "Keras"
    MX_NET = "MXNet"
    ML_NET = "ML.NET"
    XGBOOST = "XGBoost"
    LIGHT_GBM = "LightGBM"
    CAT_BOOST = "CatBoost"
    THEANO = "Theano"
    DL4J = "Deeplearning4j"
    FASTAI = "fastai"
    PROPHET = "Prophet"
    GLUONCV = "GluonCV"
    ONNX = "ONNX"

    @staticmethod
    def is_member(name):
        for member in MachineLearningAlgorithmToolType:
            if member.value == name:
                return True
        return False

class MachineLearningAlgorithmTool():
    relation = {'SoftwareAgent': {'disjoint': [], 'subclass_of': []},
                'Tool': {'disjoint': [], 'subclass_of': ['Entity', 'SoftwareAgent', 'Version', 'Project']},
                'ApacheMahout': {'disjoint': [], 'subclass_of': []}, 'DLLearner': {'disjoint': [], 'subclass_of': []},
                'ELKI': {'disjoint': [], 'subclass_of': []}, 'EViews': {'disjoint': [], 'subclass_of': []},
                'Encog': {'disjoint': [], 'subclass_of': []}, 'FAMa': {'disjoint': [], 'subclass_of': []},
                'H2O': {'disjoint': [], 'subclass_of': []}, 'IBMMiner': {'disjoint': [], 'subclass_of': []},
                'Julia': {'disjoint': [], 'subclass_of': []}, 'KNIME': {'disjoint': [], 'subclass_of': []},
                'KXEN': {'disjoint': [], 'subclass_of': []}, 'LIONsolver': {'disjoint': [], 'subclass_of': []},
                'LibLinear': {'disjoint': [], 'subclass_of': []}, 'LibSVM': {'disjoint': [], 'subclass_of': []},
                'Library': {'disjoint': [], 'subclass_of': []}, 'C': {'disjoint': [], 'subclass_of': ['Library']},
                'CPlusPlus': {'disjoint': [], 'subclass_of': ['Library']},
                'Centura': {'disjoint': [], 'subclass_of': ['Library']},
                'DotNet': {'disjoint': [], 'subclass_of': ['Library']},
                'Java': {'disjoint': [], 'subclass_of': ['Library']},
                'JavaScript': {'disjoint': [], 'subclass_of': ['Library']},
                'NodeJS': {'disjoint': [], 'subclass_of': ['Library']},
                'Other': {'disjoint': [], 'subclass_of': ['Library']},
                'PHP': {'disjoint': [], 'subclass_of': ['Library']},
                'Python': {'disjoint': [], 'subclass_of': ['Library']},
                'Ruby': {'disjoint': [], 'subclass_of': ['Library']}, 'MLPACK': {'disjoint': [], 'subclass_of': []},
                'MassiveOnlineAnalysis': {'disjoint': [], 'subclass_of': []},
                'Mathematica': {'disjoint': [], 'subclass_of': []}, 'Matlab': {'disjoint': [], 'subclass_of': []},
                'MicrosoftAzureMachineLearning': {'disjoint': [], 'subclass_of': []},
                'MonteCarloMachineLearningLibrary': {'disjoint': [], 'subclass_of': []},
                'NeuroSolutions': {'disjoint': [], 'subclass_of': []}, 'Octave': {'disjoint': [], 'subclass_of': []},
                'OpenCV': {'disjoint': [], 'subclass_of': []}, 'OpenNN': {'disjoint': [], 'subclass_of': []},
                'OracleDataMining': {'disjoint': [], 'subclass_of': []}, 'Orange': {'disjoint': [], 'subclass_of': []},
                'R': {'disjoint': [], 'subclass_of': []}, 'RCASE': {'disjoint': [], 'subclass_of': []},
                'RapidMiner': {'disjoint': [], 'subclass_of': []}, 'SAP': {'disjoint': [], 'subclass_of': []},
                'SASEnterpriseMiner': {'disjoint': [], 'subclass_of': []}, 'SPSS': {'disjoint': [], 'subclass_of': []},
                'SQLServerAnalysisServices': {'disjoint': [], 'subclass_of': []},
                'STATISTICADataMiner': {'disjoint': [], 'subclass_of': []},
                'Shogun': {'disjoint': [], 'subclass_of': []}, 'Stata': {'disjoint': [], 'subclass_of': []},
                'Weka': {'disjoint': [], 'subclass_of': []}, 'YALE': {'disjoint': [], 'subclass_of': []},
                'Yooreeka': {'disjoint': [], 'subclass_of': []}, 'mlpy': {'disjoint': [], 'subclass_of': []},
                'scikit-learn': {'disjoint': [], 'subclass_of': []}}