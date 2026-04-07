# CodeLens
AI-powered code review agent that analyzes GitHub repositories to verify whether a candidate’s resume accurately reflects their actual projects, skills, and development behavior.

## Prerequisites
- Copy or create a project-root `.env` with your API keys (see app startup checks for required variables).
- Optional: install GitNexus globally for richer graphs: `npm install -g gitnexus`. If GitNexus is missing or fails, CodeLens falls back to a direct git clone and filesystem scan.

## GitHub Login
- To enable login, create a GitHub OAuth App at `github.com/settings/developers` with callback URL set to `{APP_URL}/component/streamlit_oauth.callback` or your Streamlit redirect URI.
