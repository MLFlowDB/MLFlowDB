import os
from datetime import datetime
from enum import Enum
from typing import List, Dict

from rest_framework.decorators import action

from MLDRSys.settings import HDFS_MODEL_DIR, MODEL_LOCAL_FILE, HDFS_CLIENT
from apps.model.mex import Language, MachineLearningProblemType, MachineLearningMethodType, \
    MachineLearningAlgorithmClassType, \
    MachineLearningEvaluationMeasuring, UserDefinedMeasure
from apps.model.mldr_model.basic import Environment, SourceCode, Framework, VersionedActivity, VersionedEntity
from apps.model.mldr_model.data_transforming import Dataset
from apps.model.prov_dm import Activity, Entity, Relation, PROVRelation
import mongoengine as me

from utils.utils import auto_versioning, get_versioning


class FeatureSelection(VersionedActivity):
    name = me.StringField()
    input_datasets = me.ListField()
    output_datasets = me.ListField()
    selected_features = me.ListField()

    @staticmethod
    def get(name_origin, version=None):
        return get_versioning(name_origin, version, FeatureSelection)

    @classmethod
    def generate(cls, name, input_datasets_class: List[Dataset], output_datasets_class: List[Dataset],
                 started_time, ended_time, selected_features=None):
        if selected_features is None:
            selected_features = []
        return FeatureSelection.objects.save(name=name, input_datasets=[i.id for i in input_datasets_class],
                                             output_datasets=[i.id for i in output_datasets_class],
                                             started_at_time=started_time, ended_at_time=ended_time,
                                             selected_features=selected_features)

    def save(self, *args, **kwargs):
        input_datasets_class = [Dataset.objects.get(id=id) for id in self.input_datasets]
        output_datasets_class = [Dataset.objects.get(id=id) for id in self.output_datasets]

        for input_dataset in input_datasets_class:
            Relation.generate(start_point=self.id, start_point_type=self._cls, end_point=input_dataset.id,
                              end_point_type=input_dataset._cls,
                              relation_type=PROVRelation.USE)

        for output_dataset in output_datasets_class:
            Relation.generate(start_point=output_dataset.id, start_point_type=output_dataset._cls, end_point=self.id,
                              end_point_type=self._cls,
                              relation_type=PROVRelation.GENERATION)

        return super(type(self), self).save(*args, **kwargs)

    def to_dict(self):

        return {
            "id": self.id,
            "version": self.version,
            "name": self.name,
            "input_datasets": [Dataset.objects.get(id=id).to_dict() for id in self.input_datasets],
            "output_datasets": [Dataset.objects.get(id=id).to_dict() for id in self.output_datasets],
            "selected_feature": self.selected_features
        }


class MachineLearningSourceCode(SourceCode):
    @classmethod
    def generate(cls, name, codes=None, file=None):
        return SourceCode.generate(cls, name, codes, file)

    def save(self, *args, **kwargs):
        return super(type(self), self).save(*args, **kwargs)


class MachineLearningFramework(Framework):
    framework_name = me.StringField()
    framework_language = me.EnumField(Language, null=True)
    framework_version = me.StringField(null=True)

    @classmethod
    def generate(cls, framework_name, framework_language=None, framework_version=None):
        return Framework.generate(cls, framework_name, framework_language, framework_version)

    @classmethod
    def get(cls, name, version=None, lang=None):
        return Framework.get(cls, name, version, lang)

    @classmethod
    def generate_or_get(cls, framework_name, framework_language=None, framework_version=None):
        return Framework.generate_or_get(cls, framework_name, framework_language, framework_version)

    def to_dict(self):
        return super(MachineLearningFramework, self).to_dict()


class MachineLearningExecutionEnvironment(Environment):
    @classmethod
    def generate_from_dict(cls, d):
        return Environment.generate_from_dict(cls, d)

    def to_dict(self):
        return super(MachineLearningExecutionEnvironment, self).to_dict()


'''
MachineLearningImplementation
机器学习算法实现
'''


class MachineLearningImplementation(VersionedActivity):
    name = me.StringField()

    learning_method = me.EnumField(MachineLearningMethodType, null=True)
    learning_problem = me.EnumField(MachineLearningProblemType, null=True)
    algorithm_class = me.ListField(default=[])

    machine_learning_platform = me.ObjectIdField(null=True)
    machine_learning_algorithm_source = me.ObjectIdField(null=True)

    @staticmethod
    def get(name_origin, version=None):
        return get_versioning(name_origin, version, MachineLearningImplementation)

    @staticmethod
    def generate(name, learning_problem: str = None, learning_method: str = None, algorithm_class=None,
                 framework: MachineLearningFramework = None, **kwargs):
        return MachineLearningImplementation.objects.create(
            name=name,
            learning_problem=MachineLearningProblemType(learning_problem) if learning_problem is not None else None,
            algorithm_class=algorithm_class if algorithm_class is not None else None,
            learning_method=MachineLearningMethodType(learning_method) if learning_method is not None else None,
            machine_learning_platform=framework.id if framework is not None else None,
            **kwargs)

    def save(self, *args, **kwargs):
        if self.machine_learning_algorithm_source is not None:
            sc = MachineLearningSourceCode.objects.get(id=self.machine_learning_algorithm_source)
            Relation.generate(start_point=self.id, start_point_type=self._cls,
                              end_point=self.machine_learning_algorithm_source,
                              end_point_type=sc._cls, relation_type=PROVRelation.QUOTE)
        if self.machine_learning_platform is not None:
            fw = MachineLearningFramework.objects.get(id=self.machine_learning_platform)
            Relation.generate(start_point=self.id, start_point_type=self._cls, end_point=self.machine_learning_platform,
                              end_point_type=fw._cls,
                              relation_type=PROVRelation.ATTRIBUTE)
        return super(type(self), self).save(*args, **kwargs)

    def to_dict(self):
        user_dict = {}
        for k in self._dynamic_fields.to_dict().keys():
            user_dict[k] = self._data.get(k)

        sc = MachineLearningSourceCode.objects.get(
            id=self.machine_learning_algorithm_source) if self.machine_learning_algorithm_source is not None else None
        fw = MachineLearningFramework.objects.get(
            id=self.machine_learning_platform) if self.machine_learning_platform is not None else None
        defined_dict = {
            'id': str(self.id),
            'version': self.version,
            'name': self.name,
            'learning_method': self.learning_method.value,
            'learning_problem': self.learning_problem.value,
            'algorithm_class': [str(i) for i in self.algorithm_class],
            'machine_learning_algorithm_source': sc.to_dict() if sc is not None else None,
            'framework': fw.to_dict() if fw is not None else None
        }

        result_dict = {**defined_dict, **user_dict}

        return result_dict


# class HyperParametersSchema(me.DynamicEmbeddedDocument):
#     name = me.StringField()

class HyperParameters(me.DynamicEmbeddedDocument):

    @classmethod
    def generate_from_dict(cls, d):
        base_list = []
        for b in base_list:
            if b not in d:
                raise ValueError(f"parameter '{str(b)}' should be provided")
        return cls(**d)

    def to_dict(self):
        user_dict = {}
        for k in self._dynamic_fields.to_dict().keys():
            user_dict[k] = self._data.get(k)

        return user_dict


class MachineLearningTask(VersionedEntity):
    name = me.StringField()
    description = me.StringField()

    @staticmethod
    def get(name_origin, version=None):
        return get_versioning(name_origin, version, MachineLearningTask)

    def to_dict(self):
        user_dict = {}
        for k in self._dynamic_fields.to_dict().keys():
            user_dict[k] = self._data.get(k)

        defined_dict = {
            "id": str(self.id),
            "version": self.version,
            "name": self.name,
            "description": self.description,
        }
        return {**user_dict, **defined_dict}


class MachineLearningTraining(VersionedEntity):
    versioning_unique = ['name_tmp']

    name_tmp = me.StringField()
    hyper_parameter = me.EmbeddedDocumentField(HyperParameters)
    task = me.ObjectIdField()
    environment = me.ObjectIdField(null=True)
    used_datasets = me.ListField()
    implementation = me.ObjectIdField()

    @staticmethod
    def get(name_origin, version=None):
        return get_versioning(name_origin, version, MachineLearningTraining, field="name_tmp")

    @classmethod
    def generate(cls, hyper_parameters, task_name, imp_name, env: MachineLearningExecutionEnvironment,
                 used_datasets: List[Dataset], started_time, ended_time, **kwargs):
        hp = HyperParameters.generate_from_dict(hyper_parameters)
        imp = MachineLearningImplementation.get(imp_name)
        task = MachineLearningTask.get(task_name)
        return MachineLearningTraining.objects.create(name_tmp=imp_name, hyper_parameter=hp, task=task.id,
                                                      environment=env.id,
                                                      used_datasets=[dataset.id for dataset in used_datasets],
                                                      started_at_time=started_time, ended_at_time=ended_time,
                                                      implementation=imp.id, **kwargs)

    def save(self, *args, **kwargs):
        used_datasets_class = [Dataset.objects.get(id=dataset_id) for dataset_id in self.used_datasets]
        print(self.id)
        for dataset in used_datasets_class:
            Relation.generate(start_point=self.id, start_point_type=self._cls, end_point=dataset.id,
                              end_point_type=dataset._cls,
                              relation_type=PROVRelation.USE)
        if self.environment is not None:
            env = MachineLearningExecutionEnvironment.objects.get(id=self.environment)
            Relation.generate(start_point=self.id, start_point_type=self._cls, end_point=env.id,
                              end_point_type=env._cls,
                              relation_type=PROVRelation.ATTRIBUTE)
        task = MachineLearningTask.objects.get(id=self.task)
        Relation.generate(start_point=self.id, start_point_type=self._cls, end_point=self.task,
                          end_point_type=task._cls,
                          relation_type=PROVRelation.DERIVATION)
        imp = MachineLearningImplementation.objects.get(id=self.implementation)
        Relation.generate(start_point=self.id, start_point_type=self._cls, end_point=self.implementation,
                          end_point_type=imp._cls,
                          relation_type=PROVRelation.DERIVATION)
        return super(MachineLearningTraining, self).save(*args, **kwargs)

    def to_dict(self):
        user_dict = {}
        for k in self._dynamic_fields.to_dict().keys():
            user_dict[k] = self._data.get(k)

        env = None
        if self.environment is not None:
            env = MachineLearningExecutionEnvironment.objects.get(id=self.environment)
        task = MachineLearningTask.objects.get(id=self.task)
        imp = MachineLearningImplementation.objects.get(id=self.implementation)
        used_datasets_class = [Dataset.objects.get(id=id) for id in self.used_datasets]
        defined_dict = {
            "id": str(self.id),
            "version": self.version,
            "name": self.name_tmp,
            "implementation": imp.to_dict(),
            "hyper_parameters": self.hyper_parameter.to_dict(),
            "datasets": [used_datasets.to_dict() for used_datasets in used_datasets_class],
            "task": task.to_dict(),
            "env": env.to_dict() if env is not None else None
        }
        return {**user_dict, **defined_dict}


class MachineLearningModelInstance(Entity):
    instance_path = me.StringField(db_field="instance_path")
    generated_from = me.ObjectIdField()

    @staticmethod
    def _upload_model(filename):
        local_src = os.path.join(MODEL_LOCAL_FILE, filename)
        dest = os.path.join(HDFS_MODEL_DIR, filename)
        if not os.path.isfile(local_src):
            raise FileExistsError(f"找不到文件:{str(local_src)}")
        HDFS_CLIENT.copy_from_local(local_src, dest)
        return dest

    @staticmethod
    def _download_model(filename, pos=MODEL_LOCAL_FILE):
        src = os.path.join(HDFS_MODEL_DIR, filename)
        local_dest = os.path.join(pos, filename)
        print(local_dest)
        if not os.path.isfile(local_dest):
            if not HDFS_CLIENT.exists(src):
                raise FileExistsError(f"找不到文件:{str(src)}")
            HDFS_CLIENT.copy_to_local(src, local_dest)
        return local_dest

    @classmethod
    def generate(cls, name, file, ml_learning_id):
        file_name = f"{name}_{str(datetime.now()).replace(':', '_')}_{file.name}"
        local_path = os.path.join(MODEL_LOCAL_FILE, file_name)
        with open(local_path, 'wb+') as destination:
            for chunk in file.chunks():
                destination.write(chunk)
        instance_path = local_path
        # instance_path = MachineLearningModelInstance._upload_dataset(file_name)
        model_instance = MachineLearningModelInstance.objects.create(instance_path=instance_path,
                                                                     generated_from=ml_learning_id)
        return model_instance

    def to_dict(self):
        ml_training_instance = MachineLearningTraining.objects.get(id=self.generated_from)
        return {
            'id': str(self.id),
            'instance_path': self.instance_path,
            'generated_from': ml_training_instance.to_dict()
        }


class MachineLearningModel(VersionedEntity):
    name = me.StringField()
    instance = me.ObjectIdField(null=True)
    generated_from = me.ObjectIdField()

    @staticmethod
    def get(name_origin, version=None):
        return get_versioning(name_origin, version, MachineLearningModel)

    @classmethod
    def generate(cls, name, ml_training_name, **kwargs):
        ml_training_instance = MachineLearningTraining.get(ml_training_name)
        return MachineLearningModel.objects.create(name=name, generated_from=ml_training_instance.id, **kwargs)

    def save(self, *args, **kwargs):
        ml_training_instance = MachineLearningTraining.objects.get(id=self.generated_from)
        Relation.generate(start_point=self.id, start_point_type=self._cls, end_point=self.generated_from,
                          end_point_type=ml_training_instance._cls,
                          relation_type=PROVRelation.GENERATION)
        if self.instance is not None:
            mi = MachineLearningModelInstance.objects.get(id=self.instance)
            Relation.generate(start_point=self.id, start_point_type=self._cls, end_point=self.instance,
                              end_point_type=mi._cls,
                              relation_type=PROVRelation.QUOTE)
        return super(MachineLearningModel, self).save(*args, **kwargs)

    def to_dict(self):
        user_dict = {}
        for k in self._dynamic_fields.to_dict().keys():
            user_dict[k] = self._data.get(k)

        ml_training_instance = MachineLearningTraining.objects.get(id=self.generated_from)

        model_instance = None
        if self.instance is not None:
            model_instance = MachineLearningModelInstance.objects.get(id=self.instance)

        return {
            "id": str(self.id),
            "version": self.version,
            "name": self.name,
            "generated_from": ml_training_instance.to_dict(),
            "model_instance": model_instance.to_dict() if model_instance is not None else None
        }


class MachineLearningEvaluationSingleMeasureEmbedding(me.DynamicEmbeddedDocument):
    measuring_class = me.StringField()
    measuring_name = me.StringField()
    user_defined = me.BooleanField(default=False)

    def to_dict(self):
        user_dict = {}
        for k in self._dynamic_fields.to_dict().keys():
            user_dict[k] = self._data.get(k)

        return {
            'measuring_class': self.measuring_class,
            'measuring_name': self.measuring_name,
            'user_defined': self.user_defined,
            **user_dict
        }


class MachineLearningEvaluation(VersionedActivity):
    name = me.StringField()
    measuring_collection = me.EmbeddedDocumentListField(MachineLearningEvaluationSingleMeasureEmbedding)

    @staticmethod
    def get(name_origin, version=None):
        return get_versioning(name_origin, version, MachineLearningEvaluation)

    @classmethod
    def generate(cls, name, measures: List[Dict], **kwargs):
        c = []
        for m in measures:
            measuring_name = m.get("name", MachineLearningEvaluationMeasuring.UNKNOWN.value)
            other_infos = {**m}
            del other_infos["name"]
            if not MachineLearningEvaluationMeasuring.is_member(measuring_name):
                raise ValueError(f"{measuring_name} is not a MachineLearningEvaluationMeasuring. You can defined one.")
            measuring_class = MachineLearningEvaluationMeasuring.measuring_class(measuring_name)
            user_defined = MachineLearningEvaluationMeasuring.is_user_defined(measuring_name)
            c.append(MachineLearningEvaluationSingleMeasureEmbedding(measuring_class=measuring_class,
                                                                     measuring_name=measuring_name,
                                                                     user_defined=user_defined, **other_infos))
        return MachineLearningEvaluation.objects.create(name=name, measuring_collection=c, **kwargs)

    def save(self, *args, **kwargs):
        return super(MachineLearningEvaluation, self).save(*args, **kwargs)

    def to_dict(self):
        user_dict = {}
        for k in self._dynamic_fields.to_dict().keys():
            user_dict[k] = self._data.get(k)

        return {
            'id': str(self.id),
            'version': self.version,
            'name': self.name,
            'measures': [i.to_dict() for i in self.measuring_collection],
            **user_dict
        }


class MachineLearningEvaluationResult(Entity):
    versioning_unique = ['name_tmp']
    name_tmp = me.StringField()
    result = me.DictField()

    @classmethod
    def generate(cls, name, result: Dict, eval: MachineLearningEvaluation, **kwargs):
        measure_names = [i.to_dict().get("measuring_name") for i in eval.measuring_collection]
        for k, v in result.items():
            if k in measure_names:
                continue
            if "::" in k:
                name_class = name.split("::")[-1]
                if UserDefinedMeasure.get(name_class) is not None:
                    continue
            raise ValueError(f"Measure {str(k)} is not in MachineLearningEvaluation {str(eval.name)}")

        return MachineLearningEvaluationResult.objects.create(name_tmp=name, result=result, from_exe=eval.id, **kwargs)

    def save(self, *args, **kwargs):
        return super(MachineLearningEvaluationResult, self).save(*args, **kwargs)

    def to_dict(self):
        return {
            "id" : str(self.id),
            "version" : str(self.version),
            "evaluation_name":self.name_tmp,
            "result":self.result,
        }


class MachineLearningEvaluationExecution(Activity):
    versioning_unique = ['model_name', 'eval_name']

    evaluation = me.ObjectIdField()
    model = me.ObjectIdField()
    eval_result = me.ObjectIdField(null=True)

    eval_name = me.StringField()
    model_name = me.StringField()

    @classmethod
    def generate(cls, eval_name, model_name, model: MachineLearningModel, mle: MachineLearningEvaluation,
                 result: MachineLearningEvaluationResult, **kwargs):
        return MachineLearningEvaluationExecution.objects.create(eval_name=eval_name, model_name=model_name,
                                                                 evaluation=mle.id, model=model.id,
                                                                 eval_result=result.id, **kwargs)

    def save(self, *args, **kwargs):
        model_instance = MachineLearningModel.objects.get(id=self.model)
        eval_instance = MachineLearningEvaluation.objects.get(id=self.evaluation)
        result_instance = MachineLearningEvaluationResult.objects.get(id=self.eval_result)
        Relation.generate(start_point=self.id, start_point_type=self._cls, end_point=self.evaluation,
                          end_point_type=eval_instance._cls,
                          relation_type=PROVRelation.DERIVATION)
        Relation.generate(start_point=self.id, start_point_type=self._cls, end_point=self.model,
                          end_point_type=model_instance._cls,
                          relation_type=PROVRelation.USE)
        Relation.generate(end_point=self.id, end_point_type=self._cls,
                          start_point=self.eval_result,start_point_type=result_instance._cls,
                          relation_type=PROVRelation.GENERATION)
        return super(MachineLearningEvaluationExecution, self).save(*args, **kwargs)

    def to_dict(self):
        user_dict = {}
        for k in self._dynamic_fields.to_dict().keys():
            user_dict[k] = self._data.get(k)

        model_instance = MachineLearningModel.objects.get(id=self.model)
        eval_instance = MachineLearningEvaluation.objects.get(id=self.evaluation)
        result_instance = MachineLearningEvaluationResult.objects.get(id=self.eval_result)
        return {
            'id': str(self.id),
            'version': self.version,
            'model': model_instance.to_dict(),
            'evaluation': eval_instance.to_dict(),
            'result': result_instance.to_dict(),
            **user_dict
        }
