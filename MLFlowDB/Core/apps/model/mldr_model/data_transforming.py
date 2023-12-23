import copy
import datetime
import os
from enum import Enum

import mongoengine as me
import numpy as np
import pandas as pd
from django.conf import settings
from mongoengine import DoesNotExist
from pandas import DataFrame

from MLDRSys.settings import DATASET_LOCAL_FILE, HDFS_DATASET_DIR, HDFS_CLIENT
from apps.model.prov_dm import Activity, Entity, Agent, Counter, Relation, PROVRelation
from apps.model.mldr_model.data_gathering import OriginDataCollection
from apps.model.mldr_model.basic import SourceCode, Framework, Environment, VersionedEntity, VersionedActivity
from utils.utils import auto_versioning, get_versioning


class DataPreparationComponentType(Enum):
    UNKNOWN = "unknown"

    # For General
    SPLIT = "split"
    RENAME = "rename"

    # For JSON
    DOC_TO_ROW = "doc_to_row"
    DOC_TO_COL = "doc_to_col"
    DOC_TO_DOC = "doc_to_doc"
    DOC_EXPAND = "doc_expand"
    DOC_REDUCE = "doc_reduce"

    # For DataFrame
    ROW_TO_ROW = "row_to_row"
    COL_TO_COL = "col_to_col"
    ROW_EXPAND = "row_expand"
    COL_EXPAND = "col_expand"
    ROW_REDUCE = "row_reduce"
    COL_REDUCE = "col_reduce"
    DATASET_AGGREGATE = "aggregate"

    @staticmethod
    def is_member(name):
        for member in DataPreparationComponentType:
            if member.value == name:
                return True
        return False


'''
DataPreparationComponentExecutionEnvironment
数据预处理元件的执行环境
'''


class DataPreparationComponentExecutionEnvironment(Environment):
    @classmethod
    def generate_from_dict(cls, d):
        return Environment.generate_from_dict(cls, d)

    def to_dict(self):
        return super(DataPreparationComponentExecutionEnvironment, self).to_dict()


'''
DataPreparationPipeline
数据预处理流水线
'''


# 将继承自Activity -> VersionedActivity
# class DataPreparationPipeline(Activity):
class DataPreparationPipeline(VersionedActivity):
    name = me.StringField()
    nodes = me.ListField(me.ObjectIdField())
    edges = me.ListField()

    def save(self, *args, **kwargs):
        for node_id in self.nodes:
            instance = DataPreparationComponent.objects.get(id=node_id)
            Relation.generate(start_point=self.id, start_point_type=self._cls, end_point=node_id,
                              end_point_type=instance._cls,
                              relation_type=PROVRelation.INFORM)
        self.version = auto_versioning(["name"], self, type(self))
        return super(DataPreparationPipeline, self).save(*args, **kwargs)

    @staticmethod
    def get(name_origin, version=None):
        return get_versioning(name_origin, version, DataPreparationPipeline)

    def to_dict(self):
        return {
            'id': str(self.id),
            'version': self.version,
            'name': self.name,
            'nodes': [DataPreparationComponent.objects.get(id=node_id).to_dict() for node_id in self.nodes],
            'edges': [[str(edge_id) for edge_id in edge] for edge in self.edges]
        }


'''
DataPreparationPipelineExecution
数据预处理流水线执行结果
'''


# 继承修改
# class DataPreparationPipelineExecution(Activity):
class DataPreparationPipelineExecution(VersionedActivity):
    name_tmp = me.StringField()
    versioning_unique = ['name_tmp']
    from_pipeline = me.ObjectIdField()
    used_collection = me.ObjectIdField()
    nodes = me.ListField(default=[])
    edges = me.ListField(default=[])

    def save(self, *args, **kwargs):
        dpp = DataPreparationPipeline.objects.get(id=self.from_pipeline)
        Relation.generate(start_point=self.id,
                          start_point_type=self._cls,
                          end_point=self.from_pipeline,
                          end_point_type=dpp._cls,
                          relation_type=PROVRelation.DERIVATION)

        oc = OriginDataCollection.objects.get(id=self.used_collection)
        Relation.generate(start_point=self.id,
                          start_point_type=self._cls,
                          end_point=self.used_collection,
                          end_point_type=oc._cls,
                          relation_type=PROVRelation.USE)

        for node_id in self.nodes:
            instance = DataPreparationComponentExecution.objects.get(id=node_id)
            Relation.generate(start_point=self.id,
                              start_point_type=self._cls,
                              end_point=node_id,
                              end_point_type=instance._cls,
                              relation_type=PROVRelation.INFORM)
        for p in self.edges:
            n1 = DataPreparationComponentExecution.objects.get(id=p[0])
            n2 = DataPreparationComponentExecution.objects.get(id=p[1])
            Relation.generate(start_point=p[1],
                              start_point_type=n2._cls,
                              end_point=p[0],
                              end_point_type=n1._cls,
                              relation_type=PROVRelation.INFORM)
        self.version = auto_versioning(["name_tmp"], self, type(self))
        return super(DataPreparationPipelineExecution, self).save(*args, **kwargs)

    @staticmethod
    def get(name_origin, version=None):
        return get_versioning(name_origin, version, DataPreparationPipelineExecution, field="name_tmp")

    @classmethod
    def generate(cls, started_time, ended_time, dpp: DataPreparationPipeline,oc:OriginDataCollection, nodes, edges):
        return cls.objects.create(started_at_time=started_time, ended_at_time=ended_time, name_tmp=dpp.name,
                                  from_pipeline=dpp.id, nodes=nodes, edges=edges,used_collection=oc.id)

    def to_dict(self):
        dpp = DataPreparationPipeline.objects.get(id=self.from_pipeline)
        return {
            'id': str(self.id),
            'version': self.version,
            'name': dpp.name,
            'nodes': [DataPreparationComponentExecution.objects.get(id=node_id).to_dict() for node_id in self.nodes],
            'edges': [[str(edge_id) for edge_id in edge] for edge in self.edges],
            'used_collection':OriginDataCollection.objects.get(id=self.used_collection).to_dict()
        }


'''
DataPreparationComponent
数据预处理元件
'''


# 继承修改
# class DataPreparationComponent(Activity):
class DataPreparationComponent(VersionedActivity):
    name = me.StringField()
    component_type = me.EnumField(DataPreparationComponentType, default=DataPreparationComponentType.UNKNOWN)
    component_parameters_list = me.ListField(default=[])
    source_code = me.ObjectIdField()
    framework = me.ObjectIdField(null=True)

    def to_dict(self):
        user_dict = {}
        for k in self._dynamic_fields.to_dict().keys():
            user_dict[k] = self._data.get(k)

        sc = DataPreparationSourceCode.objects.get(id=self.source_code)
        fw = DataPreparationComponentFramework.objects.get(id=self.framework) if self.framework is not None else None
        return {
            'id': str(self.id),
            'version': self.version,
            'name': self.name,
            'component_type': self.component_type.value,  # 获取枚举的值
            'component_parameters_list': self.component_parameters_list,
            'source_code': sc.to_dict(),
            'framework': fw.to_dict() if self.framework else None,  # 处理可能为 None 的情况.
            **user_dict
        }

    @staticmethod
    def get(name_origin, version=None):
        return get_versioning(name_origin, version, DataPreparationComponent)

    def save(self, *args, **kwargs):
        sc = DataPreparationSourceCode.objects.get(id=self.source_code)
        Relation.generate(start_point=self.id, start_point_type=self._cls, end_point=self.source_code,
                          end_point_type=sc._cls, relation_type=PROVRelation.QUOTE)
        if self.framework is not None:
            fw = DataPreparationComponentFramework.objects.get(
                id=self.framework) if self.framework is not None else None
            Relation.generate(start_point=self.id, start_point_type=self._cls, end_point=self.framework,
                              end_point_type=fw._cls,
                              relation_type=PROVRelation.ATTRIBUTE)
        self.version = auto_versioning(["name"], self, type(self))
        return super(DataPreparationComponent, self).save(*args, **kwargs)


'''
DataPreparationComponentExecution
数据预处理元件执行结果
'''


# 继承修改
# class DataPreparationComponentExecution(Activity):
class DataPreparationComponentExecution(VersionedActivity):
    name_tmp = me.StringField()
    versioning_unique = ['name_tmp']
    from_component = me.ObjectIdField()
    input_args = me.DictField(mnull=True)
    outputs = me.StringField(null=True)
    environment = me.ObjectIdField(null=True)

    def save(self, *args, **kwargs):
        c = DataPreparationComponent.objects.get(id=self.from_component)
        env = DataPreparationComponentExecutionEnvironment.objects.get(
            id=self.environment) if self.environment is not None else None
        Relation.generate(start_point=self.id,
                          start_point_type=self._cls,
                          end_point=self.from_component,
                          end_point_type=c._cls,
                          relation_type=PROVRelation.DERIVATION)
        if env:
            Relation.generate(start_point=self.id,
                              start_point_type=self._cls,
                              end_point=self.environment,
                              end_point_type=env._cls,
                              relation_type=PROVRelation.ATTRIBUTE)
        self.version = auto_versioning(["name_tmp"], self, type(self))
        return super(DataPreparationComponentExecution, self).save(*args, **kwargs)

    @classmethod
    def generate(cls, started_time, ended_time, dp: DataPreparationComponent, input_args, outputs,
                 env: DataPreparationComponentExecutionEnvironment, **kwargs):
        return cls.objects.create(started_at_time=started_time, ended_at_time=ended_time, name_tmp=dp.name,
                                  from_component=dp.id, input_args=input_args, outputs=outputs, environment=env.id,
                                  **kwargs)

    def to_dict(self):
        user_dict = {}
        for k in self._dynamic_fields.to_dict().keys():
            user_dict[k] = self._data.get(k)

        c = DataPreparationComponent.objects.get(id=self.from_component)
        env = DataPreparationComponentExecutionEnvironment.objects.get(
            id=self.environment) if self.environment else None

        return {
            'id': str(self.id),
            'version': str(self.version),
            'from_component': c.to_dict(),
            'environment': env.to_dict() if env else None,
            **user_dict
        }


'''
SourceCode
源代码
'''


class DataPreparationSourceCode(SourceCode):
    @classmethod
    def generate(cls, name, codes=None, file=None):
        return SourceCode.generate(cls, name, codes, file)

    def save(self, *args, **kwargs):
        return super(type(self), self).save(*args, **kwargs)


'''
DataPreparationComponentFramework
数据预处理元件执行框架
'''


class DataPreparationComponentFramework(Framework):
    framework_name = me.StringField()
    framework_language = me.StringField(null=True)
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
        return super(DataPreparationComponentFramework, self).to_dict()


class DatasetInstanceType(Enum):
    CSV = "csv"
    JSON = "json"
    PARQUET = "parquet"
    NPY = "npy"
    H5 = "hdf5"
    EXCEL = "excel"
    UNKNOWN = "unknown"

    # DATAFRAME = "pandas_dataframe"

    @classmethod
    def check_file_name(cls, filename):
        ext = str(filename).split(".")[-1]
        if ext == "csv":
            return DatasetInstanceType.CSV
        elif ext in ["xlsx", "xls"]:
            return DatasetInstanceType.EXCEL
        elif ext == "json":
            return DatasetInstanceType.JSON
        elif ext in ["np", "npy"]:
            return DatasetInstanceType.NPY
        elif ext in ["h5", "hdf5"]:
            return DatasetInstanceType.H5
        elif ext == "parquet":
            return DatasetInstanceType.PARQUET
        else:
            return DatasetInstanceType.UNKNOWN

    @staticmethod
    def is_member(name):
        for member in DatasetInstanceType:
            if member.value == name:
                return True
        return False


class DatasetEvaluationResultType(Enum):
    CSV = "csv"
    JSON = "json"


'''
DatasetInstance
'''


# 继承修改
# class DatasetInstance(Entity):
class DatasetInstance(VersionedEntity):
    instance_path = me.StringField(db_field="instance_path")
    instance_type = me.EnumField(DatasetInstanceType, default=DatasetInstanceType.PARQUET)
    generated_from = me.ObjectIdField()
    name_tmp = me.StringField()
    versioning_unique = ['name_tmp']

    @staticmethod
    def _upload_dataset(filename):
        local_src = os.path.join(DATASET_LOCAL_FILE, filename)
        dest = os.path.join(HDFS_DATASET_DIR, filename)
        if not os.path.isfile(local_src):
            raise FileExistsError(f"找不到文件:{str(local_src)}")
        HDFS_CLIENT.copy_from_local(local_src, dest)
        return dest

    @staticmethod
    def _download_dataset(filename, pos=DATASET_LOCAL_FILE):
        src = os.path.join(HDFS_DATASET_DIR, filename)
        local_dest = os.path.join(pos, filename)
        print(local_dest)
        if not os.path.isfile(local_dest):
            if not HDFS_CLIENT.exists(src):
                raise FileExistsError(f"找不到文件:{str(src)}")
            HDFS_CLIENT.copy_to_local(src, local_dest)
        return local_dest

    @staticmethod
    def _save_file(name, data, file_type):
        def _export_csv(df: DataFrame):
            filename = f"{name}_{str(datetime.datetime.now()).replace(':', '_')}_{name}.csv"
            print("导出为csv")
            path = os.path.join(DATASET_LOCAL_FILE, filename)
            df.to_csv(path, encoding='utf_8_sig')
            return filename

        def _export_npy(df: DataFrame):
            filename = f"{name}_{str(datetime.datetime.now()).replace(':', '_')}_{name}.npy"
            path = os.path.join(DATASET_LOCAL_FILE, filename)
            nparray = df.to_numpy()
            np.save(path, nparray)
            return filename

        def _export_h5(df: DataFrame):
            filename = f"{name}_{str(datetime.datetime.now()).replace(':', '_')}_{name}.hdf5"
            path = os.path.join(DATASET_LOCAL_FILE, filename)
            df.to_hdf(path, key=name, encoding='utf_8_sig')
            return filename

        def _export_excel(df: DataFrame):
            filename = f"{name}_{str(datetime.datetime.now()).replace(':', '_')}_{name}.xlsx"
            path = os.path.join(DATASET_LOCAL_FILE, filename)
            df.to_excel(path, encoding='utf_8_sig')
            return filename

        def _export_json(df: DataFrame):
            filename = f"{name}_{str(datetime.datetime.now()).replace(':', '_')}_{name}.json"
            path = os.path.join(DATASET_LOCAL_FILE, filename)
            df.to_json(path, encoding='utf_8_sig')
            return filename

        def _export_parquet(df: DataFrame):
            raise NotImplementedError

        df = pd.DataFrame.from_records(data)
        handle_list = {
            DatasetInstanceType.PARQUET: _export_parquet,
            DatasetInstanceType.CSV: _export_csv,
            DatasetInstanceType.NPY: _export_npy,
            DatasetInstanceType.H5: _export_h5,
            DatasetInstanceType.EXCEL: _export_excel,
            DatasetInstanceType.JSON: _export_json
        }
        return handle_list[file_type](df)

    @staticmethod
    def _get_feature_from_data(data):
        df = pd.DataFrame.from_records(data)
        return df.columns.tolist()

    @staticmethod
    def _get_feature_from_file(path, file_type: DatasetInstanceType):
        def _import_csv(path):
            df = pd.read_csv(path)
            return df

        def _import_npy(path):
            nparray = np.load(path, encoding='utf_8_sig')
            df = pd.DataFrame(nparray)
            return df

        def _import_h5(path):
            raise NotImplementedError()

        def _import_excel(path):
            df = pd.read_excel(path)
            return df

        def _import_json(path):
            df = pd.read_json(path)
            return df

        def _import_parquet(path):
            raise NotImplementedError

        handle_list = {
            DatasetInstanceType.PARQUET: _import_parquet,
            DatasetInstanceType.CSV: _import_csv,
            DatasetInstanceType.NPY: _import_npy,
            DatasetInstanceType.H5: _import_h5,
            DatasetInstanceType.EXCEL: _import_excel,
            DatasetInstanceType.JSON: _import_json
        }
        return handle_list[file_type](path).columns.tolist()

    @classmethod
    def generate(cls, name, file_type, pipeline_exe: DataPreparationPipelineExecution, data=None, file=None,
                 auto_feature=False, **kwargs):
        di = None
        features = []
        if file is not None:
            file_name = file.name
            if file_type == DatasetInstanceType.UNKNOWN:
                new_file_type = DatasetInstanceType.check_file_name(file_name)
            else:
                new_file_type = copy.deepcopy(file_type)
            file_name = f"{name}_{str(datetime.datetime.now()).replace(':', '_')}_{file_name}"
            path = os.path.join(DATASET_LOCAL_FILE, file_name)
            with open(path, 'wb+') as destination:
                for chunk in file.chunks():
                    destination.write(chunk)
            instance_path = path
            # instance_path = DatasetInstance._upload_dataset(file_name)
            di = DatasetInstance.objects.create(instance_path=instance_path, instance_type=new_file_type,
                                                generated_from=pipeline_exe.id, **kwargs)
            if auto_feature:
                features = DatasetInstance._get_feature_from_file(path, new_file_type)
        if data is not None:
            if file == DatasetInstanceType.UNKNOWN:
                new_file_type = DatasetInstanceType.JSON
            else:
                new_file_type = copy.deepcopy(file_type)
            file_name = DatasetInstance._save_file(name, data, new_file_type)
            local_path = os.path.join(DATASET_LOCAL_FILE, file_name)
            instance_path = local_path
            # instance_path = DatasetInstance._upload_dataset(file_name)
            di = DatasetInstance.objects.create(instance_path=instance_path, instance_type=new_file_type,
                                                generated_from=pipeline_exe.id, **kwargs)
            if auto_feature:
                features = DatasetInstance._get_feature_from_data(data)
        return di, features

    def save(self, *args, **kwargs):
        pe = DataPreparationPipelineExecution.objects.get(id=self.generated_from)
        Relation.generate(start_point=self.id,
                          start_point_type=self._cls,
                          end_point=self.generated_from,
                          end_point_type=pe._cls,
                          relation_type=PROVRelation.GENERATION)
        return super(DatasetInstance, self).save(*args, **kwargs)

    def to_dict(self):
        pe = DataPreparationPipelineExecution.objects.get(id=self.generated_from)
        return {
            'id': str(self.id),
            'instance_path': self.instance_path,
            'instance_type': self.instance_type.value,
        }


'''
Dataset
数据集
'''


class DatasetType(Enum):
    VAL = "val"
    TRAIN = "train"
    TEST = "test"

    @staticmethod
    def is_member(name):
        for member in DatasetType:
            if member.value == name:
                return True
        return False


class Dataset(VersionedEntity):
    name = me.StringField()
    instance = me.ObjectIdField()
    generated_from = me.ObjectIdField()
    usage = me.EnumField(DatasetType)
    features = me.ListField(default=[])

    @classmethod
    def generate(cls, name, file_type, pipeline_exe: DataPreparationPipelineExecution, data=None, file=None,
                 usage=DatasetType.TRAIN, features=None, **kwargs):
        if features is None:
            features = []
        if data is None and file is None:
            return None

        if len(features) <= 0:
            is_auto_feature = True

        instance, auto_feature = DatasetInstance.generate(name=name, data=data, file=file, file_type=file_type,
                                                          pipeline_exe=pipeline_exe, auto_feature=True)
        if len(features) > 0:
            true_features = copy.deepcopy(features)
        else:
            true_features = copy.deepcopy(auto_feature)
        ds = Dataset.objects.create(name=name, instance=instance.id, generated_from=pipeline_exe.id, usage=usage,
                                    features=true_features, **kwargs)
        return ds

    @staticmethod
    def get(name_origin, version=None):
        return get_versioning(name_origin, version, Dataset)

    def save(self, *args, **kwargs):
        instance = DatasetInstance.objects.get(id=self.instance)
        Relation.generate(start_point=self.id,
                          start_point_type=self._cls,
                          end_point=self.instance,
                          end_point_type=instance._cls,
                          relation_type=PROVRelation.QUOTE)
        pe = DataPreparationPipelineExecution.objects.get(id=self.generated_from)
        Relation.generate(start_point=self.id,
                          start_point_type=self._cls,
                          end_point=self.generated_from,
                          end_point_type=pe._cls,
                          relation_type=PROVRelation.GENERATION)
        return super(Dataset, self).save(*args, **kwargs)

    def to_dict(self):
        user_dict = {}
        for k in self._dynamic_fields.to_dict().keys():
            user_dict[k] = self._data.get(k)

        instance = DatasetInstance.objects.get(id=self.instance)
        pe = DataPreparationPipelineExecution.objects.get(id=self.generated_from)
        return {
            'id': str(self.id),
            'name': self.name,
            'version': self.version,
            'instance': instance.to_dict(),
            'generated_from': pe.to_dict(),
            'features': self.features,
            **user_dict
        }


'''
DatasetEvaluationResult
数据集评估结果
'''


class DatasetEvaluationResult(Entity):
    path = me.StringField(db_field="instance_path")
    type = me.EnumField(DatasetEvaluationResultType, default=DatasetEvaluationResultType.CSV)
