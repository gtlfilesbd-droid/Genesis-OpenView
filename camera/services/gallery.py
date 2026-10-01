from django.core.files.base import ContentFile

from camera.models import PersonSample

from .faces import SAMPLE_LIMIT, SAMPLE_SAME, current_vector, prepare_embedding, similarity, to_bytes


def _photo_bytes(field) -> bytes:
    if not field:
        return b""
    try:
        field.open("rb")
        return field.read()
    except Exception:
        return b""
    finally:
        try:
            field.close()
        except Exception:
            pass


def _ready_vector(blob, photo_field, save):
    ready = current_vector(bytes(blob) if blob else None)
    if ready is not None:
        return ready
    try:
        updated, changed = prepare_embedding(bytes(blob) if blob else None, _photo_bytes(photo_field) or None)
    except Exception:
        print("face embed failed")
        return None
    if changed and updated:
        save(updated)
        return current_vector(updated)
    return None


def load_person_vector(person):
    """Return one buffalo_l embedding, rebuilding the stored vector when it is the wrong length."""
    vectors = load_person_vectors(person)
    return vectors[0] if vectors else None


def load_person_vectors(person):
    """Return every sample embedding for this person."""
    samples = list(person.samples.all())
    if samples:
        vectors = []
        for sample in samples:
            ready = _ready_vector(
                sample.embedding,
                sample.photo,
                lambda updated, sample=sample: _store_sample_vector(sample, updated),
            )
            if ready is not None:
                vectors.append(ready)
        if vectors:
            return vectors
    ready = _ready_vector(
        person.embedding,
        person.photo,
        lambda updated: _store_person_vector(person, updated),
    )
    return [ready] if ready is not None else []


def _store_sample_vector(sample, updated: bytes) -> None:
    sample.embedding = updated
    sample.save(update_fields=["embedding"])


def _store_person_vector(person, updated: bytes) -> None:
    person.embedding = updated
    person.save(update_fields=["embedding"])


def sample_skip_reason(person, embedding) -> str:
    """Return why this face cannot be stored: invalid, full, duplicate, or empty."""
    vector = current_vector(to_bytes(embedding))
    if vector is None:
        return "invalid"
    samples = list(person.samples.all())
    if len(samples) >= SAMPLE_LIMIT:
        return "full"
    for sample in samples:
        stored = current_vector(bytes(sample.embedding) if sample.embedding else None)
        if stored is not None and similarity(vector, stored) >= SAMPLE_SAME:
            return "duplicate"
    return ""


def add_sample(person, photo_bytes: bytes, embedding, filename: str) -> bool:
    """Store another angle unless it repeats a saved sample or the profile is full."""
    vector = current_vector(to_bytes(embedding))
    if vector is None or not photo_bytes or sample_skip_reason(person, embedding):
        return False
    sample = PersonSample(person=person, embedding=to_bytes(vector))
    sample.photo.save(filename, ContentFile(photo_bytes), save=False)
    sample.save()
    changed = []
    if not person.photo:
        person.photo.save(filename, ContentFile(photo_bytes), save=False)
        changed.append("photo")
    if current_vector(bytes(person.embedding) if person.embedding else None) is None:
        person.embedding = to_bytes(vector)
        changed.append("embedding")
    if changed:
        person.save(update_fields=changed)
    return True
