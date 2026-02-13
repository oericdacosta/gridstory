# API Documentation

🚧 **Under Development** - API not yet implemented

## Planned Endpoints

### Race Data

#### Get Race Timeline
```
GET /race/{year}/{round}
```

Returns the structured race timeline (events, strategy, key moments).

**Response**:
```json
{
  "year": 2025,
  "round": 1,
  "event": "Australian Grand Prix",
  "events": [
    {
      "type": "undercut",
      "lap": 23,
      "driver": "VER",
      "target": "HAM",
      "time_gained": 1.2
    }
  ]
}
```

#### Get Race Report
```
GET /race/{year}/{round}/report
```

Returns a generated race report in narrative form.

**Response**:
```json
{
  "report": "Max Verstappen executed a perfect undercut on lap 23..."
}
```

### Interactive Chat

#### Chat with Race Data
```
POST /chat
```

Interactive Q&A about race data.

**Request**:
```json
{
  "year": 2025,
  "round": 1,
  "message": "Who was fastest in sector 2?"
}
```

**Response**:
```json
{
  "answer": "Max Verstappen was fastest in sector 2 with a time of 24.123s..."
}
```

## Authentication

TBD

## Rate Limiting

TBD
