from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("camera", "0006_analyticsrule_coverage"),
    ]

    operations = [
        migrations.AlterField(
            model_name="alarm",
            name="created_at",
            field=models.DateTimeField(auto_now_add=True, db_index=True),
        ),
        migrations.AlterField(
            model_name="facecapture",
            name="created_at",
            field=models.DateTimeField(auto_now_add=True, db_index=True),
        ),
    ]
