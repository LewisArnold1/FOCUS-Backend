from rest_framework import serializers
from django.contrib.auth.models import User
from django.contrib.auth.hashers import make_password
from .models import CalibrationData


class RegisterUserSerializer(serializers.ModelSerializer):
    """Creating New Users"""
    class Meta:
        model = User
        fields = ['username', 'password']
        extra_kwargs = {'password': {'write_only': True}}

    def create(self, validated_data):
        validated_data['password'] = make_password(validated_data['password'])
        return super(RegisterUserSerializer, self).create(validated_data)

class CalibrationSerializer(serializers.ModelSerializer):
    class Meta:
        model = CalibrationData
        data = serializers.JSONField()
        timestamp = serializers.DateTimeField()