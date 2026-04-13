# CodeLens
AI-powered code review agent that analyzes GitHub repositories to verify whether a candidate’s resume accurately reflects their actual projects, skills, and development behavior.

## Prerequisites
- Copy or create a project-root `.env` with your API keys (see app startup checks for required variables).
- Optional: install GitNexus globally for richer graphs: `npm install -g gitnexus`. If GitNexus is missing or fails, CodeLens falls back to a direct git clone and filesystem scan.

## GitHub Login
- To enable login, create a GitHub OAuth App at `github.com/settings/developers` with callback URL set to `{APP_URL}/component/streamlit_oauth.callback` or your Streamlit redirect URI.

## .env API requirements:
OPENROUTER_API_KEY
PINECONE_API_KEY
GITHUB_TOKEN
VOYAGE_API_KEY
APP_URL=http://localhost:8501
OPENROUTER_MODEL
GITNEXUS_USE_EMBEDDINGS=1
GITNEXUS_CLONE_DEPTH=1
GITNEXUS_FILE_LIMIT=2000
GITHUB_CLIENT_ID
GITHUB_CLIENT_SECRET
VOYAGE_MAX_BATCH_TOKENS
VOYAGE_CHARS_PER_TOKEN_EST
VOYAGE_MAX_BATCH_ITEMS

## UI improvement ideas:
[x] CodeLens logo
[x] Match cluely.com design for inspo
[] Add list of workflow steps
[] Popups with speedometers
[] Make resume analysis simpler
[] Add interactive timeline of commits
[] Add code graph (interactive graph that looks like gitnexus codebase graph or obsidian knowledge to look like)
[] Deploy on Repl.it and/or Vercel

CodeLens logo
Make design more colorful - See cluely.com for inspo
- Overall Quality, AI Usage, Commit Health, etc. to Github style cards (see Latest from our changelog on Github.com)
- Bullet point descriptions for each of the 5 main scores
- Visually show the scores with a spedometer look
- Clicking speedometer will open up a scrollable pop-up to corresponding description + graph
- Proper dividers between description sections

Commit Health:
- Consider timeline of commits (for commit health pop up)

Resume:
- Save resume for easy access later (if possible)
- Click Resume match box to pop up resume with color coded highlights over what matches vs what does not match
- Annotated descriptions pointing to color coded highlighted parts (red, yellow, green)

Job Fit:
- Description will also have github style bullet points (see Latest from our changelog on Github.com)

Overall order:
- 5 result speedometer cards
- Strengths/concerns
- Graph
- Skill map
- Recommendations
