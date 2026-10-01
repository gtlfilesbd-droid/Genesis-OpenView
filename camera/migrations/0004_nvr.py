from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("camera", "0003_personsample"),
    ]

    operations = [
        migrations.CreateModel(
            name="Nvr",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("host", models.CharField(max_length=64)),
                ("username", models.CharField(max_length=128)),
                ("password", models.CharField(max_length=128)),
            ],
        ),
    ]
