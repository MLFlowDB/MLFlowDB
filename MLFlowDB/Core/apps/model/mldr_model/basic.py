import copy
import datetime
import os
from enum import Enum

import mongoengine as me
from mongoengine import DoesNotExist

from MLDRSys import settings
from apps.model.prov_dm import Entity, Agent, Collection, Activity
from utils.utils import auto_versioning, get_versioning

'''
Versioned
'''


class VersionedEntity(Entity):
    versioning_unique = ['name']
    def save(self, *args, **kwargs):
        ignore_version = kwargs.get("ignore_version",False)
        print(ignore_version)
        new_kwargs = copy.deepcopy(kwargs)
        if "ignore_version" in new_kwargs:
            del new_kwargs["ignore_version"]
        if not ignore_version:
            version = auto_versioning(self.versioning_unique, self, type(self))
            # if version == -1:
            #     # destory
            #     return copy_artifact
            self.version = version
        return super(VersionedEntity, self).save(*args, **new_kwargs)


class VersionedAgent(Agent):
    versioning_unique = ['name']

    def save(self, *args, **kwargs):
        ignore_version = kwargs.get("ignore_version",False)
        new_kwargs = copy.deepcopy(kwargs)
        if "ignore_version" in new_kwargs:
            del new_kwargs["ignore_version"]
        if not ignore_version:
            self.version = auto_versioning(self.versioning_unique, self, type(self))
        return super(VersionedAgent, self).save(*args, **new_kwargs)


class VersionedCollection(Collection):
    versioning_unique = ['name']

    def save(self, *args, **kwargs):
        ignore_version = kwargs.get("ignore_version",False)
        new_kwargs = copy.deepcopy(kwargs)
        if "ignore_version" in new_kwargs:
            del new_kwargs["ignore_version"]
        if not ignore_version:
            self.version = auto_versioning(self.versioning_unique, self, type(self))
        return super(VersionedCollection, self).save(*args, **new_kwargs)

class VersionedActivity(Activity):
    versioning_unique = ['name']

    def save(self, *args, **kwargs):
        self.version = auto_versioning(self.versioning_unique, self, type(self))
        return super(VersionedActivity, self).save(*args, **kwargs)

class SourceCodeLocation(Enum):
    DB = "db"
    FS = "fs"

class SourceCode(VersionedEntity):
    name = me.StringField()
    location = me.EnumField(SourceCodeLocation, default=SourceCodeLocation.DB)
    codes = me.StringField(null=True)
    path = me.StringField(null=True)

    @staticmethod
    def generate(class_type, name, codes=None, file=None):
        if file is not None:
            print("Source codes in file")
            file_path = os.path.join(os.path.join(settings.MEDIA_ROOT, "source_code"),
                                     f"{name}_{str(datetime.datetime.now()).replace(':', '_')}_{file.name}")

            with open(file_path, 'wb+') as destination:
                for chunk in file.chunks():
                    destination.write(chunk)

            sc = class_type.objects.create(name=name, location=SourceCodeLocation.FS, path=file_path)
            return sc
        elif codes is not None:
            print(f"Source codes in db:{codes}")
            sc = class_type.objects.create(name=name, location=SourceCodeLocation.DB, codes=codes)
            return sc
        else:
            return None

    def to_dict(self):
        # 将实例转换为字典
        return {
            'id': str(self.id),
            'version': self.version,
            'name': self.name,
            'location': self.location.value,  # 获取枚举的值
            'codes': self.codes,
            'path': self.path
        }


class Framework(Agent):
    framework_name = me.StringField()
    framework_language = me.StringField(null=True)
    framework_version = me.StringField(null=True)

    @staticmethod
    def generate(class_type, framework_name, framework_language=None, framework_version=None):
        if framework_name is None:
            return None
        args = {"framework_name": framework_name}
        if framework_language is not None:
            args["framework_language"] = framework_language
        if framework_version is not None:
            args["framework_version"] = framework_version
        return class_type.objects.create(**args)

    @staticmethod
    def get(class_type,name, version=None, lang=None):
        try:
            fw = class_type.objects.get(framework_name=name, framework_version=version,
                                                               framework_language=lang)
            return fw
        except DoesNotExist:
            return None

    @staticmethod
    def generate_or_get(class_type, framework_name, framework_language=None, framework_version=None):
        if framework_name is None:
            return None
        fw = class_type.get(framework_name)
        if fw is None:
            return class_type.generate(framework_name, framework_language, framework_version)
        else:
            return fw

    def to_dict(self):
        # 将实例转换为字典
        return {
            'id': str(self.id),
            'framework_name': self.framework_name,
            'framework_language': str(self.framework_language),
            'framework_version': self.framework_version
        }

class Environment(Agent):
    os_name = me.StringField(null=True)
    os_version = me.StringField(null=True)
    os_bits = me.StringField(null=True)
    os_arch = me.StringField(null=True)
    os_node = me.StringField(null=True)

    python_build_type = me.StringField(null=True)
    python_build_time = me.StringField(null=True)
    python_compiler = me.StringField(null=True)
    python_branch = me.StringField(null=True)
    python_implementation = me.StringField(null=True)
    python_revision = me.StringField(null=True)
    python_version = me.StringField(null=True)

    machine_type = me.StringField(null=True)
    machine_processor = me.StringField(null=True)
    machine_gpu_number = me.IntField()
    machine_gpu_infos = me.ListField(me.DictField(null=True))
    machine_memory_total = me.StringField(null=True)
    machine_disk_number = me.IntField(null=True)
    machine_disk_infos = me.ListField(me.DictField(null=True))

    runtime_random_status = me.StringField(null=True, db_field="runtime_random_status")

    @staticmethod
    def generate_from_dict(cls, d):
        if d == {}:
            return None
        env = cls()
        if "os" in d:
            os_d = d.get("os", {})
            env.os_name = os_d.get("os_name", None)
            env.os_version = os_d.get("os_version", None)
            env.os_bits = os_d.get("os_bits", None)
            env.os_arch = os_d.get("os_arch", None)
            env.os_node = os_d.get("os_node", None)
        if "python" in d:
            python_d = d.get("python", {})
            env.python_build_type = python_d.get("python_build_type", None)
            env.python_build_time = python_d.get("python_build_time", None)
            env.python_compiler = python_d.get("python_compiler", None)
            env.python_branch = python_d.get("python_branch", None)
            env.python_implementation = python_d.get("python_implementation", None)
            env.python_revision = python_d.get("python_revision", None)
            env.python_version = python_d.get("python_version", None)
        if "machine" in d:
            machine_d = d.get("machine", {})
            env.machine_type = machine_d.get("machine_type", None)
            env.machine_processor = machine_d.get("machine_processor", None)
            env.machine_gpu_number = machine_d.get("machine_gpu_number", None)
            env.machine_gpu_infos = machine_d.get("machine_gpu_infos", None)
            env.machine_memory_total = machine_d.get("machine_memory_total", None)
            env.machine_disk_number = machine_d.get("machine_disk_number", None)
            env.machine_disk_infos = machine_d.get("machine_disk_infos", None)
        if "runtime" in d:
            runtime_d = d.get("runtime", {})
            env.runtime_random_status = runtime_d.get("runtime_random_status", None)
        env.save()
        return env

    def to_dict(self):
        os_data = {
            'os_name': self.os_name,
            'os_version': self.os_version,
            'os_bits': self.os_bits,
            'os_arch': self.os_arch,
            'os_node': self.os_node
        }

        python_data = {
            'python_build_type': self.python_build_type,
            'python_build_time': self.python_build_time,
            'python_compiler': self.python_compiler,
            'python_branch': self.python_branch,
            'python_implementation': self.python_implementation,
            'python_revision': self.python_revision,
            'python_version': self.python_version
        }

        machine_data = {
            'machine_type': self.machine_type,
            'machine_processor': self.machine_processor,
            'machine_gpu_number': self.machine_gpu_number,
            'machine_gpu_infos': self.machine_gpu_infos,
            'machine_memory_total': self.machine_memory_total,
            'machine_disk_number': self.machine_disk_number,
            'machine_disk_infos': self.machine_disk_infos
        }

        runtime_data = {
            'runtime_random_status': self.runtime_random_status
        }

        return {
            'os': os_data,
            'python': python_data,
            'machine': machine_data,
            'runtime': runtime_data
        }

