# Generated by Django 5.1.2 on 2025-02-18 15:02

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('eye_processing', '0018_simpleeyemetrics_left_iris_velocity_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='simpleeyemetrics',
            name='movement_type',
            field=models.CharField(default='fixation', max_length=10),
        ),
    ]
