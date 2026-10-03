from django.db import migrations, models


def copy_zones(apps, schema_editor):
    Zone = apps.get_model("camera", "Zone")
    Rule = apps.get_model("camera", "AnalyticsRule")
    for zone in Zone.objects.all():
        Rule.objects.get_or_create(
            camera_number=zone.camera_number,
            kind="zone",
            defaults={
                "points": zone.points,
                "duration_seconds": zone.dwell_seconds,
                "active": zone.active,
                "direction": "any",
                "max_people": 5,
            },
        )


class Migration(migrations.Migration):

    dependencies = [
        ("camera", "0004_nvr"),
    ]

    operations = [
        migrations.CreateModel(
            name="AnalyticsRule",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("camera_number", models.PositiveIntegerField()),
                (
                    "kind",
                    models.CharField(
                        choices=[
                            ("zone", "Zone"),
                            ("line_cross", "Line crossing"),
                            ("object_in", "Object in field"),
                            ("object_removed", "Remove object"),
                            ("people_count", "People counting"),
                            ("crowd", "Crowd detection"),
                        ],
                        max_length=32,
                    ),
                ),
                ("points", models.JSONField(default=list)),
                (
                    "direction",
                    models.CharField(
                        choices=[("forward", "Forward"), ("backward", "Backward"), ("any", "Any")],
                        default="any",
                        max_length=16,
                    ),
                ),
                ("duration_seconds", models.PositiveIntegerField(default=60)),
                ("max_people", models.PositiveIntegerField(default=5)),
                ("active", models.BooleanField(default=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
        ),
        migrations.AddConstraint(
            model_name="analyticsrule",
            constraint=models.UniqueConstraint(fields=("camera_number", "kind"), name="uniq_analytics_camera_kind"),
        ),
        migrations.AddField(
            model_name="alarm",
            name="kind",
            field=models.CharField(default="zone", max_length=32),
        ),
        migrations.RunPython(copy_zones, migrations.RunPython.noop),
    ]
