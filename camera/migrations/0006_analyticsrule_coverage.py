from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("camera", "0005_analyticsrule"),
    ]

    operations = [
        migrations.AddField(
            model_name="analyticsrule",
            name="coverage",
            field=models.CharField(
                choices=[
                    ("touch", "Any part in the area"),
                    ("inside", "Whole object inside the area"),
                ],
                default="touch",
                max_length=16,
            ),
        ),
    ]
