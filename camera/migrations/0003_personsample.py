from django.db import migrations, models
import django.db.models.deletion


def copy_existing_photos(apps, schema_editor):
    Person = apps.get_model("camera", "Person")
    PersonSample = apps.get_model("camera", "PersonSample")
    for person in Person.objects.all():
        if not person.embedding or not person.photo:
            continue
        PersonSample.objects.create(
            person=person,
            photo=person.photo.name,
            embedding=person.embedding,
        )


class Migration(migrations.Migration):

    dependencies = [
        ("camera", "0002_facecapture_list_status"),
    ]

    operations = [
        migrations.CreateModel(
            name="PersonSample",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("photo", models.ImageField(upload_to="people/samples/")),
                ("embedding", models.BinaryField()),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                (
                    "person",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="samples",
                        to="camera.person",
                    ),
                ),
            ],
            options={
                "ordering": ["created_at"],
            },
        ),
        migrations.RunPython(copy_existing_photos, migrations.RunPython.noop),
    ]
