from bson import ObjectId
from rest_framework_mongoengine import serializers
from apps.model.mldr_model.data_transforming import DataPreparationComponent, DataPreparationComponentFramework, \
    DataPreparationPipeline, DataPreparationComponentExecution, DataPreparationPipelineExecution, Dataset
from apps.model.mex import MachineLearningProblemType, MachineLearningMethodType
from apps.model.mldr_model.data_gathering import OriginData, OriginDataCollection
from apps.model.mldr_model.ml_activity import MachineLearningImplementation, MachineLearningTask, MachineLearningTraining, \
    FeatureSelection, MachineLearningModel, MachineLearningModelInstance, MachineLearningEvaluation, \
    MachineLearningEvaluationExecution

'''
DATA GATHERING
'''
class OriginDataSerializer(serializers.DynamicDocumentSerializer):
    class Meta:
        model = OriginData
        fields = '__all__'

    def create(self, validated_data):
        return OriginData.objects.create(**validated_data)

class OriginDataCollectionSerializer(serializers.DynamicDocumentSerializer):
    class Meta:
        model = OriginDataCollection
        fields = '__all__'

    def create(self, validated_data):
        return OriginDataCollection.objects.create(**validated_data)

'''
DATA TRANSFORMING
'''
class DataPreparationComponentSerializer(serializers.DynamicDocumentSerializer):
    class Meta:
        model = DataPreparationComponent
        fields = '__all__'

    def create(self, validated_data):
        return DataPreparationComponent.objects.create(**validated_data)

class DataPreparationPipelineExecutionSerializer(serializers.DynamicDocumentSerializer):
    class Meta:
        model = DataPreparationPipelineExecution
        fields = '__all__'

    def create(self, validated_data):
        return DataPreparationPipelineExecution.objects.create(**validated_data)

class DataPreparationComponentExecutionSerializer(serializers.DynamicDocumentSerializer):
    class Meta:
        model = DataPreparationComponentExecution
        fields = '__all__'

    def create(self, validated_data):
        return DataPreparationComponentExecution.objects.create(**validated_data)

class DataPreparationPipelineSerializer(serializers.DynamicDocumentSerializer):
    class Meta:
        model = DataPreparationPipeline
        fields = '__all__'

    def create(self, validated_data):
        return DataPreparationPipeline.objects.create(**validated_data)

class DataPreparationComponentFrameworkSerializer(serializers.DynamicDocumentSerializer):
    class Meta:
        model = DataPreparationComponentFramework
        fields = '__all__'

    def create(self, validated_data):
        return DataPreparationComponentFramework.generate_or_get(**validated_data)

class DatasetSerializer(serializers.DynamicDocumentSerializer):
    class Meta:
        model = Dataset
        fields = '__all__'

class MachineLearningImplementationSerializer(serializers.DynamicDocumentSerializer):
    class Meta:
        model = MachineLearningImplementation
        fields = '__all__'

    def create(self, validated_data):
        print(validated_data)
        return MachineLearningImplementation.generate(**validated_data).to_dict()


class MachineLearningTrainingSerializer(serializers.DynamicDocumentSerializer):
    class Meta:
        model = MachineLearningTraining
        fields = '__all__'

class FeatureSelectionSerializer(serializers.DynamicDocumentSerializer):
    class Meta:
        model = FeatureSelection
        fields = '__all__'

class MachineLearningModelSerializer(serializers.DynamicDocumentSerializer):
    class Meta:
        model = MachineLearningModel
        fields = '__all__'

class MachineLearningModelInstanceSerializer(serializers.DynamicDocumentSerializer):
    class Meta:
        model = MachineLearningModelInstance
        fields = '__all__'

class MachineLearningEvaluationSerializer(serializers.DynamicDocumentSerializer):
    class Meta:
        model = MachineLearningEvaluation
        fields = '__all__'

class MachineLearningEvaluationExecutionSerializer(serializers.DynamicDocumentSerializer):
    class Meta:
        model = MachineLearningEvaluationExecution
        fields = '__all__'

class MachineLearningTaskSerializer(serializers.DynamicDocumentSerializer):
    name = serializers.serializers.CharField()
    description = serializers.serializers.CharField(allow_null=True)

    class Meta:
        model = MachineLearningTask
        fields = '__all__'

    def create(self, validated_data):
        return MachineLearningTask.objects.create(**validated_data)
