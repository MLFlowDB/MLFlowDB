import copy
import json
from enum import auto, Enum

from bson import ObjectId
from django.http import JsonResponse
from mongoengine import DoesNotExist

from apps.model.prov_dm import Counter, Relation


def _handle_data(data, key, allow_none=False, allowed=None, default=None, allow_type=None):
    if data:
        if allowed and isinstance(allowed, list):
            if data not in allowed:
                raise ValueError(f"value is not allowed. allowed value is {str(str(allowed))}，but get{str(data)}")
    else:
        if not allow_none:
            raise ValueError(f"{str(key)} is not allowed to be empty.")
        else:
            data = default
    if data and allow_type and not isinstance(data, allow_type):
        raise TypeError(f"{str(key)} should be {str(allow_type)}, but get {str(type(data))}")
    return data

def get_request(request,key:str,default=None,allow_none=True,allowed=None, allow_type=None):
    data = request.data.get(key, None)
    return _handle_data(data=data, key=key, allow_none=allow_none, allowed=allowed, default=default,
                        allow_type=allow_type)
    # data = request.data.get(key, None)
    # print(request.data)
    # if data is None:
    #     if allow_none is False:
    #         raise ValueError(f"{key} is not allowed to be empty.")
    #     data = default
    # return data

def get_json_r(request, key: str, allow_none=True, allowed=None, default=None, allow_type=None,enum=None):
    data = json.loads(request.body).get(key, None)
    return _handle_data(data=data, key=key, allow_none=allow_none, allowed=allowed, default=default,
                        allow_type=allow_type)

def json_response(msg):
    return JsonResponse({
        'status': '200',
        'data': msg
    })

def copy_relation(source_artifact, target_artifact, except_list=None):
    # copy endpoint that value in target
    start_rs = Relation.objects.filter(start_point=source_artifact.id)
    target_value = [str(i) for i in target_artifact._data.values()]
    source_value = [str(i) for i in source_artifact._data.values()]
    for r in start_rs:
        if str(r.end_point) in target_value:
            Relation.generate(start_point=target_artifact.id,
                              end_point=r.end_point,
                              start_point_type = target_artifact._cls,
                              end_point_type=r.end_point_type,
                              relation_type=r.relation_type
                              )
        if str(r.end_point) not in source_value:
            r.delete()

    # change startpoint to target_artifact
    end_rs = Relation.objects.filter(end_point=source_artifact.id)
    for r in end_rs:
        r.start_point = target_artifact.id
        r.start_point_type = target_artifact._cls
        r.save()


def auto_versioning(unique_keys,artifact,classtype):
    args = {}
    for key in unique_keys:
        args[key] = getattr(artifact,key)
    result = classtype.objects(**args).aggregate([
            {"$group": {
                "_id": None,
                "max_version": {"$max": "$version"}
            }}
        ])
    max_version = None
    for doc in result:
        max_version = doc.get("max_version")
        break

    if max_version is not None:
        version =  max_version + 1
    else:
        version = 1

    # check if it is needed to copy?
    if version == 1:
        return version
    lastest = classtype.objects.get(**args,version=max_version)
    if lastest.id == artifact.id:
        copy_artifact = classtype(**lastest._data)
        copy_artifact.id = ObjectId()
        copy_artifact.version = version - 1  if version>1 else 1
        copy_artifact.save(ignore_version=True)
        # return -1,copy_artifact
        copy_relation(artifact,copy_artifact)

    return version

def get_versioning(name_origin,version = None,classtype=None,field="name"):
    if "$" in name_origin:
        spl = str(name_origin).split("$")
        name = spl[0]
        version = int(spl[1])
    else:
        name = copy.deepcopy(name_origin)
    try:
        qr = {field:name}
        if version is None:
            qs = classtype.objects.filter(**qr)
            max_version = classtype.objects(**qr).order_by('-version').first()
            if max_version is None:
                raise ValueError(f"{name_origin} of {str(classtype)} does not exist(max version).")
            return max_version
        else:
            m = classtype.objects.get(**qr, version=version)
            if m is None:
                raise ValueError(f"{name_origin} does not exist.")
            return m
    except DoesNotExist:
        raise ValueError(f"{name_origin} does not exist.")

def have_intersection(list1, list2):
    set1 = set(list1)
    set2 = set(list2)

    intersection = set1 & set2

    return len(intersection) > 0

class RequestDataType(Enum):
    FORM_DATA = auto()
    JSON = auto()

def get_user_defined_property(request, property_list=None, request_type=RequestDataType.FORM_DATA):
    if property_list is None:
        property_list = []

    if request_type == RequestDataType.FORM_DATA:
        data = request.data
    elif request_type == RequestDataType.JSON:
        data = json.loads(request.body)
    else:
        raise NotImplementedError

    user_defined_property = {key: value for key, value in data.items() if key not in property_list}

    return user_defined_property

