# Cyberbullying Detection Flask App

## Deployment options

This is a Flask web app that loads saved models from `saved_models/` and uses a local SQLite database.

### Recommended hosts
- **Render**: easiest for Python web services
- **Fly.io**: good for container-based deployment
- **Railway**: simple deploy flow
- **Heroku**: works with `Procfile`

### Not ideal
- **Vercel**: not suitable for this stateful Flask app

## Deploy on Render
1. Push this repo to GitHub.
2. Create a new Web Service on Render.
3. Connect the GitHub repo.
4. Set the build command:
   ```bash
   pip install -r requirements.txt
   ```
5. Set the start command:
   ```bash
   gunicorn app:app --bind 0.0.0.0:$PORT
   ```
6. Add environment variables as needed.

## Notes
- `requirements.txt` lists dependencies.
- `Procfile` is provided for Heroku-style deployments.
- `runtime.txt` specifies Python 3.14.3.
