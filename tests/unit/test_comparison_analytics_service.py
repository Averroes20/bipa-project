import pytest
import json
from unittest.mock import MagicMock
from datetime import datetime, timedelta
from app.services.comparison_analytics_service import ComparisonAnalyticsService
from app.models.audio_models import AudioAnalysis

def test_get_comparison_analytics_empty(mock_db_session):
    mock_query = mock_db_session.query.return_value
    mock_filter = mock_query.filter.return_value
    mock_order = mock_filter.order_by.return_value
    mock_order.all.return_value = []
    
    service = ComparisonAnalyticsService(mock_db_session)
    result = service.get_comparison_analytics("user_1")
    
    assert "Complete an analysis" in result["aiInsights"]
    assert result["pronunciationComparison"][0]["You"] == 0

def test_get_comparison_analytics_with_data(mock_db_session):
    analysis_detail = {
        "dimensions": {"pronunciation": 80, "fluency": 75, "intonation": 85, "clarity": 90, "accent": 88},
        "analysisMetadata": {"fluency_basis": {"Speech Rate": "invalid WPM", "Pause Duration Avg": "invalid s"}},
        "errors": [
            {"type": "Substitution", "expected": "buku", "detected": "buko"},
            {"type": "Substitution", "expected": "a", "detected": "u"},
            {"type": "Substitution", "expected": "t", "detected": "d"}
        ]
    }
    
    mock_record1 = MagicMock(spec=AudioAnalysis)
    mock_record1.analysis_detail = json.dumps(analysis_detail)
    mock_record1.created_at = datetime.utcnow()
    
    mock_record2 = MagicMock(spec=AudioAnalysis)
    mock_record2.analysis_detail = "invalid json"
    mock_record2.created_at = datetime.utcnow() - timedelta(days=10)
    
    mock_query = mock_db_session.query.return_value
    mock_filter = mock_query.filter.return_value
    mock_order = mock_filter.order_by.return_value
    mock_order.all.return_value = [mock_record1, mock_record2]
    
    service = ComparisonAnalyticsService(mock_db_session)
    result = service.get_comparison_analytics("user_1")
    
    assert result["pronunciationComparison"][0]["You"] == 80.0
    assert result["speakingStatistics"]["wpm"] == 0.0
    assert result["speakingStatistics"]["avg_pause_duration"] == 0.0
    assert len(result["wordStatistics"]) > 0
    assert result["wordStatistics"][0]["word"] == "buku"
