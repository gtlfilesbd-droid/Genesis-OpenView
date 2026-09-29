from .faces import current_vector, prepare_embedding


def _photo_bytes(person) -> bytes:
    if not person.photo:
        return b""
    try:
        person.photo.open("rb")
        return person.photo.read()
    except Exception:
        return b""
    finally:
        try:
            person.photo.close()
        except Exception:
            pass


def load_person_vector(person):
    """Return a buffalo_l embedding, rebuilding it from the saved photo when the stored vector is old."""
    blob = bytes(person.embedding) if person.embedding else None
    ready = current_vector(blob)
    if ready is not None:
        return ready
    try:
        updated, changed = prepare_embedding(blob, _photo_bytes(person) or None)
    except Exception:
        print("face embed failed:", getattr(person, "pk", ""))
        return None
    if changed and updated:
        person.embedding = updated
        person.save(update_fields=["embedding"])
        return current_vector(updated)
    return None
