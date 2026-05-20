# QuantEdge v2.1

AI-powered trading strategy backtester with Flask backend and Render deployment support.

## Project Structure

```
.
├── api/
│   ├── backtest_engine.py    ← Vercel-compatible API helper
│   └── index.py             ← Vercel Flask entrypoint for /api/* routes
├── public/
│   └── index.html           ← Frontend UI
├── app.py                   ← Flask web app entrypoint for local/Render
├── backtest_engine.py       ← Core backtest logic for app.py
├── requirements.txt
├── runtime.txt
├── render.yaml              ← Render deployment manifest
├── README.md
└── .gitignore
```

## Render Deployment

### 1. Push your code to GitHub

If this repository is not yet on GitHub, add the remote and push:

```bash
git add .
git commit -m "Add Render deployment manifest and docs"
git push origin main
```

### 2. Connect to Render

1. Go to https://dashboard.render.com.
2. Create a new **Web Service**.
3. Connect your GitHub account and select this repository.
4. Choose branch `main`.
5. Set the environment to **Python**.
6. Use the default build command or set:

```bash
pip install -r requirements.txt
```

7. Set the start command:

```bash
gunicorn app:app
```

8. Add the required environment variable:

```
GROQ_API_KEY = your_api_key_here
```

9. Deploy.

> If Render detects `render.yaml`, it will use the service settings from that file automatically.

### 3. Access your app

After deployment, your app will be available at the Render URL shown in the dashboard.
The API endpoints will be available at `/api/*` and the frontend will load from the root URL.

---

## Local Development

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
python app.py
# Open http://localhost:5000
```

## Helpful Tests

Run these to verify the backtest logic before pushing:

```bash
python test_backtest.py
python test_api_call.py
```

---

## Fixes in v2.1

- **Render support**: Added `render.yaml` and updated deployment docs
- **AI analyze fix**: Correct prompt and response format
- **AI optimize fix**: Response now returns `suggestions[]` array matching frontend expectations
- **Parse-strategy fix**: Correct request/response field names
- **`requirements.txt`**: Added so Python dependencies install cleanly
