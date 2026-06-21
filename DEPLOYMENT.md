# MedSight Deployment Guide for Render

This guide walks you through deploying MedSight to Render with Qdrant Cloud as the managed vector database.

---

## What Was Created

The following files were added/modified for production deployment:

| File | Purpose |
|------|---------|
| `Dockerfile` | Production container image with FastEmbed model pre-cached |
| `.dockerignore` | Excludes local data, caches, and dev files from Docker build |
| `render.yaml` | Render Blueprint for one-click deployment |
| `src/config.py` | Updated for container paths and graceful fallbacks |
| `src/main.py` | Added startup validation and health checks |

---

## Prerequisites

1. **GitHub account** with your MedSight code pushed to a repository
2. **Render account** (free tier available at https://render.com)
3. **Qdrant Cloud account** (free tier available at https://cloud.qdrant.io)
4. **Together AI API key** (from https://api.together.xyz)

---

## Step 1: Set Up Qdrant Cloud

### 1.1 Create a Cluster

1. Go to [https://cloud.qdrant.io](https://cloud.qdrant.io) and sign up/log in
2. Click **"Create Cluster"**
3. Choose:
   - **Provider**: Any (AWS, GCP, or Azure)
   - **Region**: Choose closest to your users (e.g., `ap-southeast-1` for India/Asia)
   - **Plan**: Free (1 GB, 1 node)
4. Click **"Create"** and wait ~2 minutes for provisioning

### 1.2 Get Connection Details

Once your cluster is ready:

1. Click on your cluster name
2. Note the **Cluster URL** (looks like `https://xxx.cloud.qdrant.io:6333`)
3. Go to **"Access"** tab → **"Create API Key"**
4. Copy the API key (starts with `eyJ...`)

---

## Step 2: Migrate Your Vector Data

You need to move your `icmr_guidelines` collection from local Docker to Qdrant Cloud.

### Option A: Re-run ETL Pipeline (Recommended)

This is the cleanest approach if your ETL is reproducible:

```bash
# Set environment variables to point to Qdrant Cloud
export QDRANT_URL="https://xxx.cloud.qdrant.io:6333"
export QDRANT_API_KEY="your-api-key-here"

# Run your ETL pipeline
python src/services/etl_pipeline.py
```

### Option B: Snapshot Export/Import

If you have data in local Qdrant that you can't easily regenerate:

1. **Export snapshot from local Qdrant**:
   ```bash
   # Your local Qdrant is running on localhost:6333
   curl -X POST 'http://localhost:6333/collections/icmr_guidelines/snapshots'
   ```

2. **Download the snapshot** from `data/qdrant_storage/snapshots/`

3. **Upload to Qdrant Cloud**:
   - In Qdrant Cloud dashboard, go to **"Collections"**
   - Click **"Upload Snapshot"**
   - Select your snapshot file
   - Name the collection `icmr_guidelines`

### Verify Migration

```bash
curl -H "api-key: your-api-key" \
     https://xxx.cloud.qdrant.io:6333/collections/icmr_guidelines
```

You should see collection info with document count > 0.

---

## Step 3: Deploy to Render

### 3.1 Connect Your Repository

1. Go to [https://dashboard.render.com](https://dashboard.render.com)
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repository (`Sappy2048/medsight`)
4. Click **"Connect"**

### 3.2 Configure the Service

Render should auto-detect the `render.yaml` blueprint. If not, configure manually:

| Setting | Value |
|---------|-------|
| **Name** | `medsight-api` |
| **Runtime** | `Docker` |
| **Branch** | `main` (or your default branch) |
| **Region** | `Singapore` or `Oregon` (choose closest to users) |
| **Plan** | `Starter` (free tier) |

### 3.3 Set Environment Variables

In the Render dashboard for your service, go to **"Environment"** tab and add:

| Variable | Value | Notes |
|----------|-------|-------|
| `QDRANT_URL` | `https://xxx.cloud.qdrant.io:6333` | Your Qdrant Cloud cluster URL |
| `QDRANT_API_KEY` | `eyJ...` | Your Qdrant Cloud API key |
| `TOGETHER_API_KEY` | `tgp_v1_...` | Your Together AI API key |

**Important**: Do NOT commit these to Git. The `.env` file is already in `.dockerignore` and `.gitignore`.

### 3.4 Deploy

Click **"Create Web Service"** or **"Deploy"**.

Render will:
1. Build the Docker image (~5-10 minutes first time)
2. Pre-download FastEmbed model during build
3. Start the container
4. Run startup health checks

Watch the **Logs** tab for startup messages. You should see:

```
============================================================
MedSight API Starting Up
============================================================
FastEmbed cache directory: /app/data/fastembed_cache
Cache directory exists: True
Qdrant URL configured: https://xxx.cloud.qdrant.io...
Together AI API key is configured
Frontend file found: /app/frontend/index.html
Qdrant connected successfully. Collections: ['icmr_guidelines']
Required collection 'icmr_guidelines' exists
============================================================
```

---

## Step 4: Verify Deployment

### 4.1 Health Check

Visit: `https://medsight-api.onrender.com/health`

Expected response:
```json
{
  "status": "healthy",
  "qdrant_connected": true
}
```

### 4.2 Web Interface

Visit: `https://medsight-api.onrender.com/`

You should see the MedSight web interface. Try a test query:

```
Tab Augmentin 625 BD + Dolo 650 TDS for 5 days
Prescription Date: 2021-03-15
Patient Age: 45
```

### 4.3 API Test

```bash
curl -X POST https://medsight-api.onrender.com/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Warfarin and Azithromycin prescribed 2015-05-15",
    "prescription_date": "2015-05-15",
    "patient_age": 67
  }'
```

---

## Troubleshooting

### Issue: "Qdrant connection failed"

**Symptoms**: Health check shows `qdrant_connected: false`

**Solutions**:
1. Verify `QDRANT_URL` and `QDRANT_API_KEY` are set correctly in Render dashboard
2. Ensure Qdrant Cloud cluster is running (not paused)
3. Check that `icmr_guidelines` collection exists in Qdrant Cloud

### Issue: "CRITICAL: TOGETHER_API_KEY is not set"

**Symptoms**: LLM calls fail, timeout errors

**Solution**: Add `TOGETHER_API_KEY` to Render environment variables

### Issue: Build fails with "No space left on device"

**Symptoms**: Docker build fails during dependency installation

**Solution**: 
- Free tier has 512 MB RAM. Reduce workers in Dockerfile CMD from `--workers 1` to no workers flag
- Or upgrade to paid plan

### Issue: "Request timeout" during analysis

**Symptoms**: 504 Gateway Timeout after 100 seconds

**Explanation**: Render free tier has 100-second request timeout. The full 6-agent pipeline can take 30–60s.

**Solutions**:
- Use the `/evaluate/stream` endpoint instead (SSE streaming)
- Upgrade to paid plan for longer timeouts
- Optimize pipeline (already concurrent where possible)

### Issue: Cold start is slow

**Symptoms**: First request after idle period takes 20–30 seconds

**Explanation**: Render free tier spins down after 15 minutes of inactivity

**Solutions**:
- Use a uptime monitoring service (e.g., UptimeRobot) to ping `/health` every 10 minutes
- Upgrade to paid plan (always-on)
- Accept cold start as trade-off for free hosting

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        Render Cloud                         │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              Docker Container (512 MB)                │  │
│  │  ┌─────────────────────────────────────────────────┐  │  │
│  │  │           FastAPI Application                   │  │  │
│  │  │  ┌─────────┐  ┌─────────┐  ┌───────────────┐   │  │  │
│  │  │  │  LLM    │  │ LangGraph│  │  Qdrant Client│   │  │  │
│  │  │  │ Client  │  │ Pipeline │  │               │   │  │  │
│  │  │  └────┬────┘  └────┬────┘  └───────┬───────┘   │  │  │
│  │  │       │            │               │           │  │  │
│  │  │       ▼            ▼               ▼           │  │  │
│  │  │  Together AI    FDA/RxNorm    Qdrant Cloud    │  │  │
│  │  │  (API calls)    (API calls)   (Vector DB)     │  │  │
│  │  │                                                 │  │  │
│  │  │  FastEmbed model (pre-cached in image)         │  │  │
│  │  └─────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## Cost Estimates (Monthly)

| Service | Free Tier | Paid (Recommended) |
|---------|-----------|-------------------|
| Render Web Service | $0 (sleeps after 15min) | $7/month (always-on) |
| Qdrant Cloud | $0 (1 GB) | $25/month (10 GB) |
| Together AI | ~$0.001/request | Pay-as-you-go |
| **Total** | **$0** | **~$32/month** |

---

## Security Checklist

- [x] API keys stored in environment variables (not in code)
- [x] `.env` file excluded from Docker build (`.dockerignore`)
- [x] Non-root user in Docker container (`appuser`)
- [x] Health check endpoint for monitoring
- [x] CORS configured for all origins (restrict in production if needed)

---

## Next Steps

1. **Set up custom domain** (Render dashboard → Settings → Custom Domain)
2. **Add monitoring** (Sentry, Datadog, or simple UptimeRobot)
3. **Configure alerts** for failed health checks
4. **Review logs** regularly for errors
5. **Scale up** if usage grows (upgrade Render plan, Qdrant cluster)

---

## Support

- Render docs: https://render.com/docs
- Qdrant Cloud docs: https://qdrant.tech/documentation/cloud/
- MedSight issues: Check logs in Render dashboard first