# Generated by Django 5.1.2 on 2024-11-26 14:01

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('eye_processing', '0004_simpleeyemetrics_user'),
    ]

    operations = [
        migrations.AlterField(
            model_name='simpleeyemetrics',
            name='eye_aspect_ratio',
            field=models.FloatField(blank=True, null=True),
        ),
    ]
