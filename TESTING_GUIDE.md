# Complete Testing Guide for Geopolitical Forecasting System

## Overview

This guide covers end-to-end testing and training of the hybrid geopolitical forecasting system, which combines:
- GDELT event data ingestion
- Temporal Knowledge Graph (TKG) construction
- LLM reasoning (Gemini API)
- Ensemble predictions with calibration

## Prerequisites

### 1. API Keys Setup

```bash
# Required: Gemini API key for LLM reasoning
export GEMINI_API_KEY="your-gemini-api-key-here"

# Optional: GDELT API (uses public endpoint by default)
export GDELT_API_KEY="your-gdelt-key-if-available"
```

To get a Gemini API key:
1. Visit https://makersuite.google.com/app/apikey
2. Create a new API key
3. Set it in your environment or .env file

### 2. Install Dependencies

All dependencies are already installed via uv:
```bash
uv sync  # If needed
```

## Part 1: Data Ingestion & Knowledge Graph

### Step 1: Ingest GDELT Events

First, populate your database with recent GDELT events:

```bash
# Ingest last 7 days of conflict and diplomatic events
uv run python -c "
from src.data.gdelt_client import GDELTClient
from src.data.event_ingestion import EventIngestionPipeline
from datetime import datetime, timedelta

# Initialize
client = GDELTClient()
pipeline = EventIngestionPipeline('data/events.db')

# Ingest recent events (QuadClass 1=diplomatic, 4=conflict)
end_date = datetime.now()
start_date = end_date - timedelta(days=7)

events = client.fetch_events(
    start_date=start_date.strftime('%Y%m%d%H%M%S'),
    end_date=end_date.strftime('%Y%m%d%H%M%S'),
    quad_classes=[1, 4],  # Diplomatic + Conflict events
    max_events=5000
)

# Store in database
pipeline.ingest_batch(events)
print(f'Ingested {len(events)} events')
"
```

### Step 2: Build Temporal Knowledge Graph

```bash
# Construct TKG from events
uv run python -c "
from src.knowledge_graph.tkg_constructor import TKGConstructor
from src.knowledge_graph.entity_resolver import EntityResolver
from src.data.storage import EventStorage

# Load events
storage = EventStorage('data/events.db')
events = storage.get_events_in_range(days_back=7)

# Build graph
resolver = EntityResolver()
constructor = TKGConstructor(resolver)
tkg = constructor.build_from_events(events)

print(f'Graph built with {tkg.number_of_nodes()} nodes, {tkg.number_of_edges()} edges')

# Save for training
import pickle
with open('data/tkg.pkl', 'wb') as f:
    pickle.dump(tkg, f)
"
```

### Step 3: Generate Embeddings

```bash
# Create vector embeddings for entities
uv run python -c "
from src.knowledge_graph.embedding_generator import EmbeddingGenerator
import pickle

# Load TKG
with open('data/tkg.pkl', 'rb') as f:
    tkg = pickle.load(f)

# Generate embeddings
generator = EmbeddingGenerator(
    embedding_dim=100,
    learning_rate=0.01,
    model_type='TransE'
)

embeddings = generator.train(
    tkg,
    epochs=100,
    batch_size=32
)

# Save embeddings
embeddings.save('data/embeddings/')
print('Embeddings generated and saved')
"
```

## Part 2: Training the TKG Predictor

### Step 4: Prepare Training Data

```bash
# Create training dataset from historical events
uv run python -c "
from src.knowledge_graph.tkg_predictor import TKGPredictor
from src.data.storage import EventStorage
import numpy as np

storage = EventStorage('data/events.db')
predictor = TKGPredictor()

# Get historical events for training
train_events = storage.get_events_in_range(
    days_back=30,
    end_date='2024-01-01'  # Historical cutoff
)

# Extract patterns for training
patterns = []
for event in train_events:
    pattern = {
        'actors': [event.actor1_code, event.actor2_code],
        'action': event.event_code,
        'timestamp': event.date_added,
        'goldstein': event.goldstein_scale,
        'quad_class': event.quad_class
    }
    patterns.append(pattern)

# Train frequency-based baseline (RE-GCN requires DGL)
predictor.train_baseline(patterns)
predictor.save('checkpoints/tkg/')
print(f'Trained on {len(patterns)} historical patterns')
"
```

### Step 5: Validate TKG Predictions

```bash
# Test TKG predictor on recent events
uv run python -c "
from src.knowledge_graph.tkg_predictor import TKGPredictor

predictor = TKGPredictor()
predictor.load('checkpoints/tkg/')

# Test prediction
test_query = {
    'actor1': 'USA',
    'actor2': 'CHN',
    'relation': 'diplomatic_engagement',
    'horizon': 30  # days
}

prediction = predictor.predict(
    test_query['actor1'],
    test_query['actor2'],
    test_query['relation'],
    horizon_days=test_query['horizon']
)

print(f'Prediction: {prediction}')
print(f'Probability: {prediction.get(\"probability\", 0):.2%}')
"
```

## Part 3: End-to-End Forecasting

### Step 6: Basic Forecast Test

```bash
# Simple forecast without GDELT data
uv run python forecast.py "Will Russia escalate military operations in Ukraine in Q1 2024?"
```

### Step 7: Full System Test with All Components

```bash
# Forecast with verbose output showing all components
uv run python forecast.py \
  "Will China increase military activity near Taiwan in 2024?" \
  --verbose \
  --format json
```

### Step 8: Batch Testing with Multiple Questions

```bash
# Create test questions file
cat > test_questions.txt << 'EOF'
Will there be a major diplomatic breakthrough between USA and China in 2024?
Will Russia escalate military operations in Ukraine in Q1 2024?
Will tensions in the South China Sea increase by mid-2024?
Will Iran and Saudi Arabia maintain diplomatic normalization through 2024?
Will NATO expand to include new members in 2024?
EOF

# Run batch predictions
uv run python -c "
from src.forecasting.forecast_engine import ForecastEngine
import json

engine = ForecastEngine()

with open('test_questions.txt', 'r') as f:
    questions = f.readlines()

results = []
for question in questions:
    question = question.strip()
    if question:
        print(f'\\nForecasting: {question}')
        try:
            forecast = engine.forecast(question)
            results.append({
                'question': question,
                'probability': forecast.probability,
                'confidence': forecast.confidence,
                'reasoning': forecast.reasoning_chain[0] if forecast.reasoning_chain else ''
            })
            print(f'  Probability: {forecast.probability:.2%}')
        except Exception as e:
            print(f'  Error: {e}')

# Save results
with open('forecast_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\\nResults saved to forecast_results.json')
"
```

## Part 4: Calibration and Evaluation

### Step 9: Store Predictions for Calibration

```bash
# Store predictions in database
uv run python -c "
from src.calibration.prediction_store import PredictionStore
from datetime import datetime, timedelta

store = PredictionStore('data/predictions.db')

# Add test predictions
test_predictions = [
    {
        'question': 'Will Russia escalate in Ukraine by March 2024?',
        'probability': 0.65,
        'confidence': 0.80,
        'resolution_date': datetime.now() + timedelta(days=30)
    },
    {
        'question': 'Will China invade Taiwan in 2024?',
        'probability': 0.15,
        'confidence': 0.75,
        'resolution_date': datetime.now() + timedelta(days=365)
    }
]

for pred in test_predictions:
    store.add_prediction(
        question=pred['question'],
        raw_probability=pred['probability'],
        confidence=pred['confidence'],
        resolution_date=pred['resolution_date']
    )

print(f'Stored {len(test_predictions)} predictions')
"
```

### Step 10: Calibrate Probabilities

```bash
# Apply calibration to predictions
uv run python -c "
from src.calibration.isotonic_calibrator import IsotonicCalibrator
from src.calibration.prediction_store import PredictionStore

store = PredictionStore('data/predictions.db')
calibrator = IsotonicCalibrator()

# Get uncalibrated predictions
predictions = store.get_uncalibrated_predictions(limit=100)

if len(predictions) >= 10:  # Need minimum for calibration
    # Extract data for calibration
    probs = [p.raw_probability for p in predictions]

    # Fit calibrator (would normally use resolved predictions)
    # For demo, using synthetic outcomes
    import numpy as np
    synthetic_outcomes = (np.array(probs) > 0.5).astype(int)

    calibrator.fit(probs, synthetic_outcomes)

    # Apply calibration
    for pred in predictions:
        calibrated = calibrator.calibrate(pred.raw_probability)
        store.update_calibrated_probability(pred.id, calibrated)

    print(f'Calibrated {len(predictions)} predictions')
else:
    print('Need more predictions for calibration')
"
```

### Step 11: Evaluate Performance

```bash
# Run evaluation metrics
uv run python evaluate.py score

# Generate performance report
uv run python evaluate.py report --output evaluation_report.html

# Check calibration drift
uv run python evaluate.py calibrate

# View performance trend
uv run python evaluate.py trend --days 30
```

## Part 5: Production Testing

### Step 12: Full System Health Check

```bash
# Comprehensive system test
uv run python -c "
from src.forecasting.forecast_engine import ForecastEngine
import sys

def health_check():
    engine = ForecastEngine()

    # Check components
    checks = {
        'Gemini API': False,
        'RAG Pipeline': False,
        'TKG Predictor': False,
        'Calibration': False,
        'Database': False
    }

    # Test Gemini
    try:
        from src.forecasting.gemini_client import GeminiClient
        client = GeminiClient()
        checks['Gemini API'] = client.client is not None
    except:
        pass

    # Test RAG
    try:
        checks['RAG Pipeline'] = engine.rag_pipeline is not None
    except:
        pass

    # Test TKG
    try:
        checks['TKG Predictor'] = engine.ensemble_predictor.tkg_predictor is not None
    except:
        pass

    # Test Database
    try:
        from src.data.storage import EventStorage
        storage = EventStorage('data/events.db')
        checks['Database'] = True
    except:
        pass

    # Test Calibration
    try:
        from src.calibration.prediction_store import PredictionStore
        store = PredictionStore('data/predictions.db')
        checks['Calibration'] = True
    except:
        pass

    # Report
    print('System Health Check:')
    print('=' * 40)
    for component, status in checks.items():
        status_str = '✓ OK' if status else '✗ FAIL'
        print(f'{component:20} {status_str}')

    # Overall status
    all_ok = all(checks.values())
    print('=' * 40)
    if all_ok:
        print('System Status: ✓ ALL SYSTEMS OPERATIONAL')
    else:
        print('System Status: ⚠ DEGRADED - Some components unavailable')
        print('\\nThe system can still make predictions using available components.')

    return all_ok

if __name__ == '__main__':
    success = health_check()
    sys.exit(0 if success else 1)
"
```

### Step 13: Performance Benchmark

```bash
# Benchmark prediction speed
uv run python -c "
import time
from src.forecasting.forecast_engine import ForecastEngine

engine = ForecastEngine()
questions = [
    'Will Russia escalate in Ukraine?',
    'Will China invade Taiwan?',
    'Will Iran develop nuclear weapons?'
]

print('Performance Benchmark:')
print('=' * 40)

for q in questions:
    start = time.time()
    try:
        forecast = engine.forecast(q)
        elapsed = time.time() - start
        print(f'Question: {q[:40]}...')
        print(f'  Time: {elapsed:.2f}s')
        print(f'  Probability: {forecast.probability:.2%}')
    except Exception as e:
        print(f'  Error: {e}')

print('=' * 40)
"
```

## Troubleshooting

### Common Issues and Solutions

1. **No GEMINI_API_KEY set**
   ```bash
   export GEMINI_API_KEY="your-key-here"
   # Or create .env file
   echo "GEMINI_API_KEY=your-key-here" > .env
   ```

2. **TKG not trained warning**
   - This is expected initially. The system works with LLM-only mode.
   - To enable TKG, follow steps 1-5 to ingest data and train.

3. **DGL not available warning**
   - The system falls back to frequency-based TKG predictions.
   - For full RE-GCN support, install DGL (optional, GPU recommended).

4. **No predictions in database**
   - The evaluation system needs stored predictions.
   - Run some forecasts first, then wait for resolutions.

5. **Torch/torchvision errors**
   ```bash
   uv pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```

## Quick Start Commands

```bash
# 1. Set API key
export GEMINI_API_KEY="your-gemini-api-key"

# 2. Quick test
uv run python forecast.py "Will China invade Taiwan in 2024?"

# 3. Verbose test with all components
uv run python forecast.py "Will Russia escalate in Ukraine?" --verbose

# 4. JSON output for parsing
uv run python forecast.py "Will Iran develop nuclear weapons by 2025?" --format json

# 5. Check system health
uv run python evaluate.py score
```

## Next Steps

1. **Collect Real GDELT Data**: Run data ingestion daily to build historical dataset
2. **Train on Your Events**: Use your specific geopolitical focus areas
3. **Calibrate with Outcomes**: As predictions resolve, update calibration
4. **Monitor Drift**: Use evaluation tools to track performance over time
5. **Fine-tune Weights**: Adjust ensemble weights based on component performance

## Architecture Summary

```
Input Question
     ↓
[Forecast Engine]
     ├── RAG Pipeline → Historical Context
     ├── Gemini LLM → Scenario Tree (generate → validate → refine)
     │        └── P_LLM = sum(affirmative scenario probabilities)
     ├── TKG Predictor → Graph Patterns → P_TKG
     ↓
[Ensemble Predictor] (α=0.6 LLM, β=0.4 TKG)
     ↓
[Calibration] (Isotonic + Temperature Scaling)
     ↓
Final Forecast (Probability + Confidence + Reasoning)
```

The system is designed to work even with components missing, gracefully degrading to use available data sources.