# Generated by Django 5.1.2 on 2024-11-25 11:36

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('eye_processing', '0004_simpleeyemetrics_user'),
    ]

    operations = [
        migrations.AddField(
            model_name='simpleeyemetrics',
            name='session_id',
            field=models.IntegerField(default=0),
        ),
    ]