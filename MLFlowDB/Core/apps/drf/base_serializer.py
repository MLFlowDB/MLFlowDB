from rest_framework_mongoengine import serializers

from apps.model.mex import UserDefinedMeasure, UserDefinedAlgorithmClass, UserDefinedTool


class UserDefinedMeasureSerializer(serializers.DynamicDocumentSerializer):
    name = serializers.serializers.CharField()
    formula = serializers.serializers.CharField()
    measuring_class = serializers.serializers.CharField()

    class Meta:
        model = UserDefinedMeasure
        fields = '__all__'

    def create(self, validated_data):
        return UserDefinedMeasure.objects.create(**validated_data)

class UserDefinedAlgorithmClassSerializer(serializers.DynamicDocumentSerializer):
    name = serializers.serializers.CharField()

    class Meta:
        model = UserDefinedAlgorithmClass
        fields = '__all__'

    def create(self, validated_data):
        return UserDefinedAlgorithmClass.objects.create(**validated_data)

class UserDefinedToolSerializer(serializers.DynamicDocumentSerializer):
    name = serializers.serializers.CharField()

    class Meta:
        model = UserDefinedTool
        fields = '__all__'

    def create(self, validated_data):
        return UserDefinedTool.objects.create(**validated_data)