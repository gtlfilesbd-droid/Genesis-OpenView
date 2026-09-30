import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("camera", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="person",
            name="list_status",
            field=models.CharField(default="whitelist", max_length=16),
        ),
        migrations.CreateModel(
            name="FaceCapture",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("camera_number", models.PositiveIntegerField()),
                ("track_id", models.IntegerField()),
                ("face_crop", models.ImageField(upload_to="captures/")),
                ("embedding", models.BinaryField()),
                ("det_score", models.FloatField(default=0)),
                ("quality", models.FloatField(default=0)),
                ("match_score", models.FloatField(blank=True, null=True)),
                ("matched_name", models.CharField(blank=True, default="", max_length=120)),
                (
                    "person",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        to="camera.person",
                    ),
                ),
            ],
            options={
                "ordering": ["-created_at"],
            },
        ),
    ]
