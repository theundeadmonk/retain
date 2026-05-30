"""Unit tests for cold-path utility functions."""

import pytest
from sqlalchemy import select as sa_select

from retain.cold_path import _fail_event, _get_transcript_id_from_event, _to_sparse_format
from retain.hot_path import _new_id, _utcnow
from retain.models import Event as EventModel


@pytest.mark.unit
class TestToSparseFormat:
    def test_normal(self):
        result = _to_sparse_format({"indices": [1, 3, 5], "values": [0.7, 0.3, 0.9]})
        assert result == "{1:0.7,3:0.3,5:0.9}/30522"

    def test_empty(self):
        result = _to_sparse_format({"indices": [], "values": []})
        assert result == "{}/30522"

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="length mismatch"):
            _to_sparse_format({"indices": [1, 2], "values": [0.5]})

    def test_missing_indices_key(self):
        with pytest.raises(ValueError, match="length mismatch"):
            _to_sparse_format({"values": [0.5]})

    def test_missing_values_key(self):
        with pytest.raises(ValueError, match="length mismatch"):
            _to_sparse_format({"indices": [1]})


@pytest.mark.unit
class TestFailEvent:
    async def test_marks_event_as_failed(self, engine):
        event_id = _new_id()
        async with engine.begin() as conn:
            await conn.execute(
                EventModel.__table__.insert().values(
                    id=event_id,
                    event_type="extraction",
                    status="pending",
                    created_at=_utcnow(),
                    updated_at=_utcnow(),
                )
            )

        await _fail_event(engine, event_id, "something broke")

        async with engine.begin() as conn:
            result = await conn.execute(
                sa_select(EventModel).where(EventModel.id == event_id)
            )
            row = result.fetchone()
        assert row.status == "failed"
        assert row.result["error"] == "something broke"
        assert row.completed_at is not None

    async def test_fail_nonexistent_event_noop(self, engine):
        await _fail_event(engine, "nonexistent_event_id_12345", "error")


@pytest.mark.unit
class TestGetTranscriptId:
    async def test_returns_transcript_id_from_event(self, engine):
        event_id = _new_id()
        transcript_id = _new_id()
        async with engine.begin() as conn:
            await conn.execute(
                EventModel.__table__.insert().values(
                    id=event_id,
                    event_type="extraction",
                    status="pending",
                    payload={"transcript_id": transcript_id},
                    created_at=_utcnow(),
                    updated_at=_utcnow(),
                )
            )

        result = await _get_transcript_id_from_event(engine, event_id)
        assert result == transcript_id

    async def test_returns_none_for_nonexistent_event(self, engine):
        result = await _get_transcript_id_from_event(
            engine, "nonexistent_event_id_12345"
        )
        assert result is None

    async def test_returns_none_when_payload_missing_transcript_id(self, engine):
        event_id = _new_id()
        async with engine.begin() as conn:
            await conn.execute(
                EventModel.__table__.insert().values(
                    id=event_id,
                    event_type="extraction",
                    status="pending",
                    payload={"other": "data"},
                    created_at=_utcnow(),
                    updated_at=_utcnow(),
                )
            )

        result = await _get_transcript_id_from_event(engine, event_id)
        assert result is None
